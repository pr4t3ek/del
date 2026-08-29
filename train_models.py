"""
Training and precomputation pipeline (spec section 43).

Run this once before starting the Flask app:

    python train_models.py

It performs the thirteen steps the brief asks for - load, validate against the dictionary,
clean, engineer features, screen for leakage and timing, split, train, evaluate, save
models, save metrics, save feature importance, save statistical results, and prepare the
decision-analysis metadata - then writes everything to models/ and outputs/.

The Flask app never retrains and never reads the 96 MB CSV per request: it loads the
compact cleaned frame plus these precomputed artifacts at startup.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import config
from src import (
    classification as cl,
    data_dictionary as dd,
    data_preprocessing as dp,
    decision_analysis as da,
    diagnostics as dg,
    eda as eda_mod,
    feature_importance as fi,
    leakage,
    regression as rg,
    statistics_tests as st,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

STEP = 0


def step(message: str) -> None:
    global STEP
    STEP += 1
    print(f"[{STEP:2d}] {message}", flush=True)


def to_jsonable(obj):
    """Convert numpy/pandas scalars so json.dump can handle the payload."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime)):
        return str(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def main() -> int:
    started = time.time()
    for directory in (config.MODEL_DIR, config.RESULT_DIR,
                      config.FIGURE_DIR, config.EXPORT_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    # -- 1. Load -----------------------------------------------------------------------
    step("Loading dataset")
    try:
        raw = dd.load_raw_dataset()
    except dd.DatasetError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        return 1
    print(f"     {raw.shape[0]:,} rows x {raw.shape[1]} columns")

    # -- 2. Validate against the data dictionary ---------------------------------------
    step("Validating against the data dictionary")
    dictionary = dd.load_dictionary()
    reconciliation = dd.reconcile_with_dictionary(raw, dictionary)
    if reconciliation["available"]:
        print(f"     undocumented in dictionary : {reconciliation['undocumented']}")
        print(f"     spelling mismatches        : "
              f"{[m['csv'] for m in reconciliation['spelling_mismatches']]}")

    step("Profiling dataset health")
    health = dp.dataset_health(raw)
    print(f"     {health['n_missing_cells']:,} missing cells, "
          f"{health['n_duplicate_rows']:,} duplicate rows, "
          f"{len(health['dead_columns'])} zero-signal columns")

    # -- 3. Clean ----------------------------------------------------------------------
    step("Cleaning")
    clean = dp.clean_dataset(raw)
    print(f"     {clean.shape[1]} columns kept, "
          f"{clean.memory_usage(deep=True).sum() / 1e6:.0f} MB in memory "
          f"(from {health['memory_mb']:.0f} MB)")

    # -- 4. Feature engineering --------------------------------------------------------
    step("Engineering decision-time features")
    items = dp.engineer_features(clean)
    orders = dp.engineer_features(dp.build_order_frame(clean))
    print(f"     line items {items.shape}, orders {orders.shape}")

    # -- 5. Leakage and timing screen --------------------------------------------------
    step("Screening predictors for leakage and decision-time availability")
    r = dd.ColumnResolver(items.columns)
    target = r.require(config.TARGET)
    mode_col = r.require(config.DECISION_VARIABLE)
    delivery_status = r.resolve("Delivery Status")

    model_pop = items
    n_cancelled = 0
    if config.EXCLUDE_CANCELLED_FROM_MODEL and delivery_status:
        mask = items[delivery_status].astype(str) == config.CANCELLED_DELIVERY_STATUS
        n_cancelled = int(mask.sum())
        model_pop = items.loc[~mask].copy()
    model_pop, bucket_map = dp.bucket_high_cardinality(model_pop, fit=True)

    features = leakage.select_model_features(model_pop, include_decision=True)
    leakage.assert_no_leakage(model_pop, features)
    availability = leakage.availability_table(items, features)
    print(f"     {len(features)} predictors kept; "
          f"{n_cancelled:,} cancelled line items excluded from modelling")

    # -- 6. Split ----------------------------------------------------------------------
    step("Splitting train/test")
    train_idx, test_idx, split_info = cl.make_split(model_pop)
    y_all = model_pop[target].to_numpy().astype(int)
    print(f"     {split_info['strategy']} split: "
          f"{len(train_idx):,} train / {len(test_idx):,} test, "
          f"class balance {y_all.mean():.3f}")

    # -- 7. Train ----------------------------------------------------------------------
    step("Training classifiers (full feature set)")
    t0 = time.time()
    full = cl.train_variant(model_pop, features, target, train_idx, test_idx, run_cv=True)
    for name, res in full["results"].items():
        m = res["metrics"]
        cv = (f"  cv={m['cv_auc_mean']:.4f}+/-{m['cv_auc_std']:.4f}"
              if "cv_auc_mean" in m else "")
        print(f"     {name:22s} AUC={m['roc_auc']:.4f}  F1={m['f1']:.4f}{cv}")
    print(f"     ({time.time() - t0:.0f}s)")

    best_name = cl.select_best(full["results"])
    best = full["results"][best_name]
    print(f"     preferred model: {best_name}")

    # -- 8. Feature ablation: how much does the decision variable carry? ---------------
    step("Running feature ablation (leakage check)")
    ablation = {}
    for label, feats in {
        "mode_only": [mode_col],
        "without_mode": [f for f in features if f != mode_col],
    }.items():
        variant = cl.train_variant(model_pop, feats, target, train_idx, test_idx)
        variant_best = cl.select_best(variant["results"])
        ablation[label] = {
            "n_features": len(feats),
            "best_model": variant_best,
            "metrics": variant["results"][variant_best]["metrics"],
        }
        print(f"     {label:14s} ({len(feats):2d} features) "
              f"AUC={ablation[label]['metrics']['roc_auc']:.4f}")
    ablation["full"] = {
        "n_features": len(features),
        "best_model": best_name,
        "metrics": best["metrics"],
    }

    # -- 9. Evaluate -------------------------------------------------------------------
    step("Evaluating the preferred model")
    y_test = full["y_test"]
    y_prob = best["y_prob"]
    curves = cl.curve_payloads(y_test, y_prob)
    sweep = cl.threshold_sweep(y_test, y_prob)
    calibration = cl.calibration_payload(y_test, y_prob)
    print(f"     AUC={best['metrics']['roc_auc']:.4f}  "
          f"PR-AUC={best['metrics']['pr_auc']:.4f}  "
          f"Brier={calibration['brier']:.4f}  "
          f"Youden J at t={curves['youden']['threshold']:.2f}")

    # -- 10. Statistical diagnostics ---------------------------------------------------
    step("Fitting logistic regression for inference")
    orders_model = orders
    if config.EXCLUDE_CANCELLED_FROM_MODEL and delivery_status:
        om = orders[delivery_status].astype(str) == config.CANCELLED_DELIVERY_STATUS
        orders_model = orders.loc[~om].copy()
    logit = dg.fit_logistic_inference(orders_model)
    vif = dg.compute_vif(orders_model)
    print(f"     AIC={logit['aic']}  BIC={logit['bic']}  "
          f"pseudo-R2={logit['pseudo_r2']}  "
          f"{vif['n_problematic']} predictors with VIF>10 or infinite")

    # -- 11. Feature importance --------------------------------------------------------
    step("Computing feature importance")
    X_test = model_pop[features].iloc[test_idx]
    perm = fi.permutation_view(best["pipeline"], X_test, y_test, n_repeats=3)
    tree_imp = fi.tree_view(best["pipeline"], features)
    logistic_pipe = full["results"]["Logistic Regression"]["pipeline"]
    coef_imp = fi.coefficient_view(logistic_pipe, features)
    print(f"     top permutation driver: {perm[0]['feature']} "
          f"({perm[0]['importance']:.4f})")

    # -- 12. Statistical significance ---------------------------------------------------
    step("Running statistical significance tests")
    stats_payload = st.run_all_tests(orders)
    print(f"     {stats_payload['n_tests']} tests, "
          f"{stats_payload['n_significant']} significant after correction, "
          f"{stats_payload['n_material']} with effect size >= 0.10")

    # -- 13. EDA and economics ----------------------------------------------------------
    step("Precomputing EDA and correlation payloads")
    eda_payload = eda_mod.build_eda(orders)
    corr_payload = eda_mod.build_correlations(orders)

    step("Training the profit regression model")
    profit_model = rg.train_profit_model(model_pop, features, train_idx, test_idx)
    profit_lateness = rg.profit_vs_lateness(orders)
    if profit_model.get("available"):
        print(f"     R2={profit_model['metrics']['r2']:.4f}  "
              f"MAE={profit_model['metrics']['mae']:.2f}")
    print(f"     corr(late, profit) = {profit_lateness.get('correlation')}")

    # -- Decision-analysis metadata ------------------------------------------------------
    step("Preparing decision-analysis metadata")
    real_col = r.require("Days for shipping (real)")
    sched_col = r.require("Days for shipment (scheduled)")
    order_r = dd.ColumnResolver(orders_model.columns)
    mode_profile = da.build_mode_profile(
        orders_model,
        order_r.require(config.TARGET),
        order_r.require(config.DECISION_VARIABLE),
        order_r.require("Days for shipping (real)"),
        order_r.require("Days for shipment (scheduled)"),
    )
    for m, prof in mode_profile.items():
        print(f"     {m:16s} promised {prof['promised_days']}d  "
              f"actual {prof['mean_transit']:.2f}d  late {prof['late_rate']:.1%}")

    # Counterfactual scoring: predict each order under every shipping mode.
    step("Scoring counterfactual shipping modes")
    sample = model_pop.iloc[test_idx]
    if len(sample) > config.SCORED_SAMPLE_SIZE:
        sample = sample.sample(
            config.SCORED_SAMPLE_SIZE, random_state=config.RANDOM_STATE
        )
    p_by_mode = {}
    for m in mode_profile:
        counterfactual = sample.copy()
        counterfactual[mode_col] = m
        p_by_mode[m] = best["pipeline"].predict_proba(counterfactual[features])[:, 1]
        print(f"     {m:16s} mean predicted P(late) = {p_by_mode[m].mean():.4f}")

    value_col = r.require("Order Item Total")
    assumptions = da.resolve_assumptions()
    policies = da.evaluate_policies(
        sample, p_by_mode, value_col, assumptions, mode_profile,
        segment_cols={"market": r.resolve("Market"), "region": r.resolve("Order Region")},
    )
    degeneracy = da.degeneracy_check(sample, p_by_mode, value_col, mode_profile)
    promise = da.promise_redesign_analysis(mode_profile)
    break_even = da.break_even_analysis(
        assumptions, mode_profile, float(pd.to_numeric(sample[value_col]).median())
    )

    sensitivities = {
        "late_penalty_fixed": da.sensitivity_analysis(
            sample, p_by_mode, value_col, mode_profile, "late_penalty_fixed",
            [0, 5, 10, 15, 25, 40, 60, 100, 150, 200],
        ),
        "speed_value_per_day": da.sensitivity_analysis(
            sample, p_by_mode, value_col, mode_profile, "speed_value_per_day",
            [0, 2, 4, 6, 8, 10, 12, 15, 20, 30],
        ),
        "holding_rate_per_day": da.sensitivity_analysis(
            sample, p_by_mode, value_col, mode_profile, "holding_rate_per_day",
            [0.0, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01],
        ),
    }
    print(f"     best policy: {policies['policies'][0]['policy']} "
          f"(avg ${policies['policies'][0]['avg_cost']:.2f}/order)")
    print(f"     degenerate without time terms: {degeneracy['is_degenerate']} "
          f"-> {degeneracy['degenerate_mix']}")

    # -- Save --------------------------------------------------------------------------
    step("Saving models and artifacts")
    joblib.dump(best["pipeline"], config.CLASSIFIER_PATH)
    joblib.dump(best["pipeline"].named_steps["prep"], config.CLASSIFIER_PIPELINE_PATH)
    if profit_model.get("available"):
        joblib.dump(profit_model["pipeline"], config.REGRESSOR_PATH)
        joblib.dump(
            profit_model["pipeline"].named_steps["prep"], config.REGRESSOR_PIPELINE_PATH
        )

    scored = sample[[c for c in (
        r.resolve("Order Id"), mode_col, value_col, r.resolve(config.ECONOMIC_TARGET),
        r.resolve("Market"), r.resolve("Order Region"), r.resolve("Customer Segment"),
        r.resolve("Department Name"), r.resolve("Category Name"), target,
    ) if c]].copy()
    for m, probs in p_by_mode.items():
        scored[f"p_late__{m}"] = probs
    scored["p_late_current"] = [
        p_by_mode[str(m)][i] if str(m) in p_by_mode else np.nan
        for i, m in enumerate(sample[mode_col].astype(str))
    ]
    scored.to_csv(config.SCORED_SAMPLE_PATH, index=False)

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": config.RANDOM_STATE,
        "target": config.TARGET,
        "decision_variable": config.DECISION_VARIABLE,
        "economic_target": config.ECONOMIC_TARGET,
        "best_model": best_name,
        "features": features,
        "bucket_map": bucket_map,
        "n_rows_raw": int(len(raw)),
        "n_rows_model": int(len(model_pop)),
        "n_orders": int(len(orders)),
        "n_cancelled_excluded": n_cancelled,
        "split": split_info,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "class_balance": round(float(y_all.mean()), 4),
        "model_comparison": {
            name: res["metrics"] for name, res in full["results"].items()
        },
        "selection_criterion": (
            "Highest held-out ROC-AUC. AUC rather than accuracy because the decision "
            "layer consumes ranked probabilities, and accuracy at a fixed 0.5 threshold "
            "rewards whichever model happens to match the base rate rather than the one "
            "that best separates late from on-time orders."
        ),
        "mode_profile": mode_profile,
        "categorical_levels": {
            c: sorted(model_pop[c].astype(str).unique().tolist())
            for c in features
            if not pd.api.types.is_numeric_dtype(model_pop[c])
        },
        # Modal level per categorical, used as the simulator's starting value. Opening on the
        # most common real-world choice is more informative than whichever level sorts first.
        "categorical_defaults": {
            c: str(model_pop[c].astype(str).mode().iloc[0])
            for c in features
            if not pd.api.types.is_numeric_dtype(model_pop[c])
            and not model_pop[c].astype(str).mode().empty
        },
        "numeric_defaults": {
            c: round(float(pd.to_numeric(model_pop[c], errors="coerce").median()), 4)
            for c in features
            if pd.api.types.is_numeric_dtype(model_pop[c])
        },
    }
    with open(config.METADATA_PATH, "w") as fh:
        json.dump(to_jsonable(metadata), fh, indent=2)

    analytics = {
        "generated_at": metadata["generated_at"],
        "health": health,
        "reconciliation": reconciliation,
        "availability": availability,
        "eda": eda_payload,
        "correlations": corr_payload,
        "statistics": stats_payload,
        "model_comparison": metadata["model_comparison"],
        "selection_criterion": metadata["selection_criterion"],
        "best_model": best_name,
        "ablation": ablation,
        "curves": curves,
        "threshold_sweep": sweep,
        "calibration": calibration,
        "logistic": logit,
        "vif": vif,
        "importance": {
            "permutation": perm[:25],
            "tree": tree_imp[:25],
            "coefficients": coef_imp[:25],
        },
        "interpretation": fi.build_interpretation(
            perm, stats_payload["results"], eda_payload.get("mode_transit_profile", [])
        ),
        "profit_model": profit_model.get("metrics", {}),
        "profit_vs_lateness": profit_lateness,
        "decision": {
            "assumptions": assumptions,
            "presets": config.SCENARIO_PRESETS,
            "mode_profile": mode_profile,
            "policies": policies,
            "break_even": break_even,
            "sensitivity": sensitivities,
            "degeneracy": degeneracy,
            "promise_redesign": promise,
        },
        "split": split_info,
        "counts": {
            "n_rows_raw": int(len(raw)),
            "n_rows_model": int(len(model_pop)),
            "n_orders": int(len(orders)),
            "n_cancelled_excluded": n_cancelled,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "n_scored_sample": int(len(sample)),
        },
    }
    with open(config.ANALYTICS_PATH, "w") as fh:
        json.dump(to_jsonable(analytics), fh, indent=2)

    print(f"\nDone in {time.time() - started:.0f}s")
    print(f"  models    -> {config.MODEL_DIR}")
    print(f"  analytics -> {config.ANALYTICS_PATH}")
    print(f"  scored    -> {config.SCORED_SAMPLE_PATH}")
    print("\nStart the application with:  python app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
