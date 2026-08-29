"""
DataCo Supply Chain Delivery Analytics - Flask application (spec sections 5, 7, 31-45, 50).

Run with:

    python train_models.py     # once, to build the models and precomputed analytics
    python app.py              # then open http://127.0.0.1:5000

The app never retrains and never re-reads the 96 MB CSV per request. At startup it loads
the model artifacts and the precomputed analytics JSON produced by train_models.py; page
routes serve stored aggregates, and the interactive endpoints (threshold slider, simulator,
scenario sliders) do arithmetic on small payloads or score a single constructed row.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

import config
from src import data_dictionary as dd
from src import decision_analysis as da
from src import reporting

app = Flask(__name__)
app.config["SECRET_KEY"] = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

NAV = [
    ("home", "Home", "index"),
    ("data", "Data", "data_page"),
    ("eda", "EDA", "eda_page"),
    ("statistics", "Statistical Tests", "statistics_page"),
    ("classification", "Classification", "classification_page"),
    ("diagnostics", "Model Diagnostics", "diagnostics_page"),
    ("risk_drivers", "Risk Drivers", "risk_drivers_page"),
    ("shipping_mode", "Shipping Mode", "shipping_mode_page"),
    ("decision", "Decision Optimization", "decision_page"),
    ("simulator", "Simulator", "simulator_page"),
    ("insights", "Business Insights", "insights_page"),
    ("presentation", "Presentation Mode", "presentation_page"),
]


# --------------------------------------------------------------------------------------
# Artifact loading
# --------------------------------------------------------------------------------------
class Store:
    """Holds everything the app needs, loaded once at startup."""

    def __init__(self) -> None:
        self.analytics: dict | None = None
        self.metadata: dict | None = None
        self.classifier = None
        self.regressor = None
        self.scored: pd.DataFrame | None = None
        self.error: str | None = None

    @property
    def ready(self) -> bool:
        return self.analytics is not None and self.classifier is not None

    def load(self) -> None:
        self.error = None
        try:
            if not config.ANALYTICS_PATH.exists() or not config.CLASSIFIER_PATH.exists():
                self.error = (
                    "Model not found.\nPlease run:\n\n    python train_models.py"
                )
                return
            with open(config.ANALYTICS_PATH) as fh:
                self.analytics = json.load(fh)
            with open(config.METADATA_PATH) as fh:
                self.metadata = json.load(fh)
            self.classifier = joblib.load(config.CLASSIFIER_PATH)
            if config.REGRESSOR_PATH.exists():
                self.regressor = joblib.load(config.REGRESSOR_PATH)
            if config.SCORED_SAMPLE_PATH.exists():
                self.scored = pd.read_csv(config.SCORED_SAMPLE_PATH)
        except Exception as exc:                            # pragma: no cover - defensive
            self.error = f"Could not load model artifacts: {exc}"


STORE = Store()
STORE.load()


def require_ready():
    """Return an error response if artifacts are missing, otherwise None."""
    if STORE.ready:
        return None
    return render_template(
        "error.html",
        nav=NAV,
        active="",
        title="Model not available",
        message=STORE.error or "Model artifacts have not been generated yet.",
        hint="Run  python train_models.py  from the project directory, then reload.",
    ), 503


def page(template: str, active: str, **context):
    """Render a page with the shared navigation context."""
    guard = require_ready()
    if guard:
        return guard
    return render_template(
        template,
        nav=NAV,
        active=active,
        analytics=STORE.analytics,
        metadata=STORE.metadata,
        **context,
    )


# --------------------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------------------
@app.route("/")
def index():
    guard = require_ready()
    if guard:
        return guard
    a = STORE.analytics
    eda = a["eda"]
    counts = a["counts"]

    profit_stats = eda.get("financial", {}).get("profit_stats", {})
    total_profit = sum(
        r.get("total_profit", 0) or 0
        for r in eda.get("target_by", {}).get("Shipping Mode", [])
    )
    value_stats = (
        eda.get("numeric_detail", {}).get("Order Item Total", {}).get("stats", {})
    )
    avg_value = value_stats.get("mean")

    sweep = a.get("threshold_sweep", [])
    at_half = next((s for s in sweep if abs(s["threshold"] - 0.50) < 1e-9), None)

    kpis = [
        {"label": "Orders", "value": f"{eda['n_orders']:,}",
         "sub": f"{counts['n_rows_raw']:,} order line items"},
        {"label": "Late delivery rate", "value": f"{eda['late_rate']:.1f}%",
         "sub": f"{eda['late_count']:,} of {eda['n_orders']:,} orders"},
        {"label": "Avg profit per order", "value": f"${profit_stats.get('mean', 0):,.2f}",
         "sub": f"median ${profit_stats.get('median', 0):,.2f}"},
        {"label": "Total order profit", "value": f"${total_profit:,.0f}",
         "sub": "sum of Order Profit Per Order"},
        {"label": "High-risk orders", "value": f"{at_half['flagged_pct']:.1f}%" if at_half else "n/a",
         "sub": "predicted P(late) above 0.50 on held-out data"},
        {"label": "Avg order value", "value": f"${avg_value:,.2f}" if avg_value else "n/a",
         "sub": "Order Item Total per order"},
    ]
    return page("index.html", "home", kpis=kpis)


@app.route("/data")
def data_page():
    return page("data.html", "data")


@app.route("/eda")
def eda_page():
    return page("eda.html", "eda")


@app.route("/statistics")
def statistics_page():
    return page("statistics.html", "statistics")


@app.route("/classification")
def classification_page():
    return page("classification.html", "classification")


@app.route("/diagnostics")
def diagnostics_page():
    return page("diagnostics.html", "diagnostics")


@app.route("/risk-drivers")
def risk_drivers_page():
    return page("risk_drivers.html", "risk_drivers")


@app.route("/shipping-mode")
def shipping_mode_page():
    return page("shipping_mode.html", "shipping_mode")


@app.route("/decision")
def decision_page():
    return page("decision.html", "decision")


@app.route("/simulator")
def simulator_page():
    guard = require_ready()
    if guard:
        return guard
    md = STORE.metadata
    # Only fields that make sense at decision time are offered as inputs (spec section 31).
    simulator_fields = [
        ("Shipping Mode", "Shipping mode", "decision"),
        ("Market", "Market", "geography"),
        ("Order Region", "Order region", "geography"),
        ("Order Country", "Order country", "geography"),
        ("Customer Segment", "Customer segment", "order"),
        ("Department Name", "Department", "order"),
        ("Category Name", "Category", "order"),
        ("Type", "Payment type", "order"),
    ]
    levels = md.get("categorical_levels", {})
    fields = [
        {"name": name, "label": label, "group": group, "options": levels.get(name, [])}
        for name, label, group in simulator_fields
        if name in levels
    ]
    numeric_fields = [
        {"name": "Order Item Quantity", "label": "Quantity", "min": 1, "max": 5, "step": 1},
        {"name": "Product Price", "label": "Product price ($)", "min": 5, "max": 2000, "step": 5},
        {"name": "Order Item Total", "label": "Order value ($)", "min": 5, "max": 2000, "step": 5},
        {"name": "Order Item Discount Rate", "label": "Discount rate", "min": 0, "max": 0.25, "step": 0.01},
    ]
    numeric_fields = [f for f in numeric_fields if f["name"] in md.get("numeric_defaults", {})]
    return page(
        "simulator.html", "simulator",
        fields=fields, numeric_fields=numeric_fields,
        defaults=md.get("numeric_defaults", {}),
        cat_defaults=md.get("categorical_defaults", {}),
    )


@app.route("/insights")
def insights_page():
    guard = require_ready()
    if guard:
        return guard
    summary = reporting.build_executive_summary(STORE.analytics)
    return page("insights.html", "insights", summary=summary)


@app.route("/presentation")
def presentation_page():
    guard = require_ready()
    if guard:
        return guard
    summary = reporting.build_executive_summary(STORE.analytics)
    return page("presentation_mode.html", "presentation", summary=summary)


@app.route("/pipeline")
def pipeline_page():
    return page("pipeline.html", "")


@app.route("/governance")
def governance_page():
    return page("governance.html", "")


@app.route("/downloads")
def downloads_page():
    return page("downloads.html", "", exports=reporting.EXPORTS)


# --------------------------------------------------------------------------------------
# JSON endpoints
# --------------------------------------------------------------------------------------
@app.route("/api/threshold")
def api_threshold():
    """Serve the precomputed threshold sweep for the section-16 slider."""
    if not STORE.ready:
        return jsonify({"error": "model not available"}), 503
    try:
        t = round(float(request.args.get("t", 0.5)), 2)
    except (TypeError, ValueError):
        return jsonify({"error": "invalid threshold"}), 400
    sweep = STORE.analytics.get("threshold_sweep", [])
    row = min(sweep, key=lambda s: abs(s["threshold"] - t)) if sweep else None
    if row is None:
        return jsonify({"error": "no sweep available"}), 404
    return jsonify(row)


def _build_simulator_row(payload: dict) -> pd.DataFrame:
    """
    Construct a single-row frame with every feature the model expects.

    Fields the user did not supply are filled from the training medians and modal
    categories recorded in model_metadata.json, so the row is always complete and
    prediction never fails on a missing column.
    """
    md = STORE.metadata
    features = md["features"]
    row: dict = {}
    for f in features:
        if f in md.get("numeric_defaults", {}):
            row[f] = md["numeric_defaults"][f]
        else:
            levels = md.get("categorical_levels", {}).get(f, [])
            fallback = levels[0] if levels else "Other"
            row[f] = md.get("categorical_defaults", {}).get(f, fallback)

    for key, value in (payload or {}).items():
        if key not in features or value in (None, ""):
            continue
        if key in md.get("numeric_defaults", {}):
            try:
                row[key] = float(value)
            except (TypeError, ValueError):
                continue
        else:
            row[key] = str(value)

    # Keep derived features consistent with the inputs the user actually moved, so a
    # what-if on price or quantity propagates instead of leaving stale defaults behind.
    qty = float(row.get("Order Item Quantity", 1) or 1)
    price = float(row.get("Product Price", 0) or 0)
    if "Sales" in row:
        row["Sales"] = price * qty
    rate = float(row.get("Order Item Discount Rate", 0) or 0)
    if "Order Item Discount" in row:
        row["Order Item Discount"] = row.get("Sales", 0) * rate
    if "Order Item Total" in row and not (payload or {}).get("Order Item Total"):
        row["Order Item Total"] = row.get("Sales", 0) - row.get("Order Item Discount", 0)
    if "unit_value" in row:
        row["unit_value"] = row.get("Sales", 0) / qty if qty else 0.0
    if "discount_depth" in row:
        row["discount_depth"] = rate
    if "log_product_price" in row:
        row["log_product_price"] = float(np.log1p(max(price, 0)))
    if "log_order_value" in row:
        row["log_order_value"] = float(np.log1p(max(row.get("Order Item Total", 0), 0)))

    return pd.DataFrame([row])[features]


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Live prediction plus counterfactual mode evaluation (spec sections 32-34).

    Returns the predicted late probability for the selected mode, the same prediction
    under every alternative mode, and the decision engine's recommendation under the
    supplied scenario assumptions.
    """
    if not STORE.ready:
        return jsonify({"error": "model not available"}), 503
    try:
        payload = request.get_json(silent=True) or {}
        inputs = payload.get("inputs", {})
        assumptions = da.resolve_assumptions(payload.get("assumptions"))
        constraints = payload.get("constraints") or {}

        base = _build_simulator_row(inputs)
        mode_profile = STORE.analytics["decision"]["mode_profile"]
        mode_col = config.DECISION_VARIABLE

        p_by_mode = {}
        for mode in mode_profile:
            row = base.copy()
            if mode_col in row.columns:
                row[mode_col] = mode
            p_by_mode[mode] = float(
                STORE.classifier.predict_proba(row)[:, 1][0]
            )

        selected = str(inputs.get(mode_col) or base[mode_col].iloc[0])
        p_selected = p_by_mode.get(selected)

        order_value = float(
            inputs.get("Order Item Total")
            or base.get("Order Item Total", pd.Series([0])).iloc[0]
            or 0
        )

        profit = None
        if STORE.regressor is not None:
            try:
                profit = float(STORE.regressor.predict(base)[0])
            except Exception:
                profit = None

        comparison = da.compare_modes(
            p_by_mode, order_value, assumptions, mode_profile,
            constraints=constraints, profit=profit,
        )

        bands = config.RISK_BANDS
        if p_selected is None:
            band = "unknown"
        elif p_selected < bands["low"]:
            band = "Low"
        elif p_selected < bands["high"]:
            band = "Medium"
        else:
            band = "High"

        return jsonify(
            {
                "selected_mode": selected,
                "p_late": p_selected,
                "risk_band": band,
                "p_by_mode": p_by_mode,
                "order_value": order_value,
                "predicted_profit": profit,
                "comparison": comparison,
                "assumptions": assumptions,
                "risk_drivers": _simulator_risk_drivers(selected, p_by_mode, mode_profile),
            }
        )
    except Exception as exc:                                # pragma: no cover - defensive
        app.logger.error("predict failed: %s", traceback.format_exc())
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


def _simulator_risk_drivers(selected: str, p_by_mode: dict, mode_profile: dict) -> list[dict]:
    """Explain the current order's risk in business terms (spec section 34)."""
    drivers = []
    p = p_by_mode.get(selected)
    prof = mode_profile.get(selected, {})

    if p is not None and p >= 0.6:
        drivers.append(
            {
                "title": "High predicted delay risk",
                "detail": f"The model puts this order at {p:.0%} probability of missing "
                          f"its promised date under {selected}.",
            }
        )
    if prof:
        promised = prof.get("promised_days")
        actual = prof.get("mean_transit")
        if promised is not None and actual is not None and actual > promised:
            drivers.append(
                {
                    "title": "The promise is tighter than typical performance",
                    "detail": (
                        f"{selected} promises {promised} day(s) but averages "
                        f"{actual:.2f} days in transit historically, so missing the "
                        "commitment is the normal outcome rather than the exception."
                    ),
                }
            )
    best = min(p_by_mode, key=p_by_mode.get) if p_by_mode else None
    if best and p is not None and p_by_mode[best] < p - 0.05:
        drivers.append(
            {
                "title": "A lower-risk mode is available",
                "detail": (
                    f"{best} carries a model-implied late probability of "
                    f"{p_by_mode[best]:.0%}, against {p:.0%} for {selected}. Whether the "
                    "switch is worthwhile depends on the cost assumptions below."
                ),
            }
        )
    if not drivers:
        drivers.append(
            {
                "title": "No elevated risk indicators",
                "detail": "This order sits within the normal risk range for its mode.",
            }
        )
    return drivers


@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    """
    Re-run the portfolio optimisation under user-set assumptions (spec sections 26-29).

    Recomputes on the stored counterfactual scores rather than re-scoring the model, so
    the sliders respond immediately.
    """
    if not STORE.ready or STORE.scored is None:
        return jsonify({"error": "scored sample not available"}), 503
    try:
        payload = request.get_json(silent=True) or {}
        preset = payload.get("preset")
        assumptions = (
            da.apply_preset(preset) if preset else da.resolve_assumptions(payload.get("assumptions"))
        )
        constraints = payload.get("constraints") or {}

        scored = STORE.scored
        mode_profile = STORE.analytics["decision"]["mode_profile"]
        p_by_mode = {
            m: scored[f"p_late__{m}"].to_numpy()
            for m in mode_profile
            if f"p_late__{m}" in scored.columns
        }
        value_col = "Order Item Total"
        if value_col not in scored.columns:
            return jsonify({"error": "order value column missing"}), 500

        policies = da.evaluate_policies(
            scored, p_by_mode, value_col, assumptions, mode_profile,
            segment_cols={"market": "Market", "region": "Order Region"},
            constraints=constraints,
        )
        median_value = float(pd.to_numeric(scored[value_col]).median())
        break_even = da.break_even_analysis(assumptions, mode_profile, median_value)

        return jsonify(
            {
                "assumptions": assumptions,
                "policies": policies,
                "break_even": break_even,
                "median_order_value": round(median_value, 2),
            }
        )
    except Exception as exc:                                # pragma: no cover - defensive
        app.logger.error("scenario failed: %s", traceback.format_exc())
        return jsonify({"error": f"Scenario evaluation failed: {exc}"}), 500


@app.route("/api/risk-matrix")
def api_risk_matrix():
    """Aggregated risk/value matrix for spec section 36."""
    if not STORE.ready or STORE.scored is None:
        return jsonify({"error": "scored sample not available"}), 503
    scored = STORE.scored
    x_col = request.args.get("x", "Order Item Total")
    if x_col not in scored.columns:
        x_col = "Order Item Total"

    df = scored.copy()
    df["_p"] = df["p_late_current"]
    df = df.dropna(subset=["_p", x_col])
    if df.empty:
        return jsonify({"points": []})

    value_median = float(pd.to_numeric(df[x_col]).median())
    risk_median = float(df["_p"].median())

    def quadrant(row):
        high_value = row[x_col] >= value_median
        high_risk = row["_p"] >= risk_median
        if high_risk and high_value:
            return "Prioritise"
        if high_risk:
            return "Watch / manage"
        if high_value:
            return "Protect"
        return "Routine"

    df["quadrant"] = df.apply(quadrant, axis=1)
    sample = df.sample(min(1500, len(df)), random_state=config.RANDOM_STATE)

    summary = (
        df.groupby("quadrant")
        .agg(orders=("_p", "size"), avg_risk=("_p", "mean"), avg_value=(x_col, "mean"))
        .reset_index()
    )
    summary["avg_risk"] = summary["avg_risk"].round(4)
    summary["avg_value"] = summary["avg_value"].round(2)

    return jsonify(
        {
            "points": [
                {"x": round(float(r[x_col]), 2), "y": round(float(r["_p"]), 4),
                 "quadrant": r["quadrant"]}
                for _, r in sample.iterrows()
            ],
            "value_median": round(value_median, 2),
            "risk_median": round(risk_median, 4),
            "summary": summary.to_dict("records"),
            "x_label": x_col,
        }
    )


@app.route("/download/<name>")
def download(name: str):
    """CSV exports (spec section 40)."""
    if not STORE.ready:
        return require_ready()
    entry = reporting.EXPORTS.get(name)
    if entry is None:
        return render_template(
            "error.html", nav=NAV, active="", title="Unknown export",
            message=f"No export named '{name}'.",
            hint="Return to the downloads page and choose one of the listed exports.",
        ), 404
    filename, builder = entry
    body = builder(STORE.analytics)
    return Response(
        body,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# --------------------------------------------------------------------------------------
# Upload (spec section 50)
# --------------------------------------------------------------------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload_page():
    status, errors = [], []
    if request.method == "POST":
        file = request.files.get("dataset")
        if file is None or not file.filename:
            errors.append("No file was selected.")
        else:
            filename = secure_filename(file.filename)
            if not filename.lower().endswith(".csv"):
                errors.append("Please upload a .csv file.")
            else:
                config.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                target = config.UPLOAD_DIR / config.DATASET_FILENAME
                file.save(target)
                try:
                    raw = dd.load_raw_dataset(target)
                    status.append(f"Dataset loaded ({len(raw):,} rows, {raw.shape[1]} columns)")

                    dictionary = dd.load_dictionary()
                    rec = dd.reconcile_with_dictionary(raw, dictionary)
                    cls = dd.classify_columns(raw)

                    status.append(f"DataCo fields detected ({rec['n_actual']} columns)")
                    status.append(
                        f"Target detected: {cls['target']}" if cls["target"]
                        else "Target NOT found: Late_delivery_risk is missing"
                    )
                    status.append(
                        f"Decision variable detected: {cls['decision']}" if cls["decision"]
                        else "Decision variable NOT found: Shipping Mode is missing"
                    )
                    status.append(
                        f"Economic variables detected: {len(cls['economic'])}"
                    )
                    status.append(
                        f"Leakage screening completed: {len(cls['leakage'])} columns flagged"
                    )
                    status.append(
                        f"PII / identifier screening: {len(cls['pii'])} PII, "
                        f"{len(cls['identifiers'])} identifier columns flagged"
                    )
                    if rec["undocumented"]:
                        errors.append(
                            "Present in the CSV but undocumented in the dictionary: "
                            + ", ".join(rec["undocumented"])
                        )
                    if rec["missing_from_csv"]:
                        errors.append(
                            "Documented in the dictionary but missing from the CSV: "
                            + ", ".join(rec["missing_from_csv"])
                        )
                    if not cls["target"] or not cls["decision"]:
                        errors.append(
                            "The dataset is missing required fields, so analysis cannot run."
                        )
                    else:
                        status.append("Ready for analysis - run  python train_models.py")
                except dd.DatasetError as exc:
                    errors.append(str(exc))
                except Exception as exc:                    # pragma: no cover - defensive
                    errors.append(f"Could not read the uploaded file: {exc}")

    return render_template(
        "upload.html", nav=NAV, active="", status=status, errors=errors,
        analytics=STORE.analytics, metadata=STORE.metadata,
    )


@app.route("/reload")
def reload_artifacts():
    STORE.load()
    return redirect(url_for("index"))


# --------------------------------------------------------------------------------------
# Error handling (spec section 45)
# --------------------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(_error):
    return render_template(
        "error.html", nav=NAV, active="", title="Page not found",
        message="That page does not exist.",
        hint="Use the navigation above to return to the dashboard.",
        analytics=STORE.analytics, metadata=STORE.metadata,
    ), 404


@app.errorhandler(413)
def too_large(_error):
    return render_template(
        "error.html", nav=NAV, active="", title="File too large",
        message="The uploaded file exceeds the size limit.",
        hint=f"The limit is {config.MAX_CONTENT_LENGTH // (1024 * 1024)} MB.",
        analytics=STORE.analytics, metadata=STORE.metadata,
    ), 413


@app.errorhandler(500)
def server_error(error):
    # Never expose a stack trace to the user (spec section 45); log it instead.
    app.logger.error("Unhandled error: %s", traceback.format_exc())
    return render_template(
        "error.html", nav=NAV, active="", title="Something went wrong",
        message="The application encountered an unexpected error.",
        hint="Check the terminal running the app for details.",
        analytics=STORE.analytics, metadata=STORE.metadata,
    ), 500


@app.context_processor
def inject_globals():
    return {
        "app_title": "DataCo Supply Chain Delivery Analytics",
        "app_subtitle": "Predict. Decide. Optimize.",
        "store_ready": STORE.ready,
    }


if __name__ == "__main__":
    if not STORE.ready:
        print("\n" + "=" * 72)
        print(STORE.error or "Model artifacts not found.")
        print("=" * 72 + "\n")
    print(f"Starting on http://{config.HOST}:{config.PORT}")
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
