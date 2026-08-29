"""
Exploratory data analysis and correlation analysis (spec sections 9 and 10).

Every aggregate here is precomputed once by train_models.py and written to JSON. Flask
then serves the stored result rather than recomputing over 180k rows on each request, and
the browser receives aggregated series rather than raw rows.

All EDA runs at the ORDER grain. The delivery outcome is recorded once per order, so
aggregating line items would count a three-line order three times toward a single
delivery result and distort every rate on this page.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data_dictionary import ColumnResolver

# Categorical fields offered in the EDA selector (spec section 9).
CATEGORICAL_CHOICES = [
    "Shipping Mode", "Market", "Order Region", "Customer Segment",
    "Category Name", "Department Name", "Order Status", "Order Country",
    "Type", "Delivery Status",
]

# Numeric fields offered in the EDA selector.
NUMERIC_CHOICES = [
    "Sales", "Order Item Total", "Order Profit Per Order", "Order Item Quantity",
    "Order Item Discount", "Order Item Discount Rate", "Order Item Profit Ratio",
    "Product Price", "Days for shipment (scheduled)", "Days for shipping (real)",
    "n_items", "distance_km", "daily_order_volume",
]

# Numeric fields for the correlation heatmap (spec section 10).
CORRELATION_FIELDS = [
    "Days for shipment (scheduled)", "Order Item Discount", "Order Item Discount Rate",
    "Product Price", "Order Item Quantity", "Sales", "Order Item Total",
    "Order Profit Per Order", "Order Item Profit Ratio", "n_items",
    "distance_km", "daily_order_volume", "Late_delivery_risk",
]


def _rate_table(df: pd.DataFrame, by: str, target: str, profit: str | None) -> list[dict]:
    """Late rate, order count and average profit for each level of `by`."""
    agg = {"orders": (target, "size"), "late_rate": (target, "mean")}
    if profit:
        agg["avg_profit"] = (profit, "mean")
        agg["total_profit"] = (profit, "sum")
    g = df.groupby(by, observed=True).agg(**agg).reset_index()
    g = g.rename(columns={by: "level"})
    g["level"] = g["level"].astype(str)
    g["late_rate"] = (100 * g["late_rate"]).round(2)
    for c in ("avg_profit", "total_profit"):
        if c in g:
            g[c] = g[c].round(2)
    return g.sort_values("orders", ascending=False).to_dict("records")


def _describe(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "count": int(s.size),
        "mean": round(float(s.mean()), 3),
        "median": round(float(s.median()), 3),
        "std": round(float(s.std()), 3),
        "min": round(float(s.min()), 3),
        "max": round(float(s.max()), 3),
        "q1": round(float(s.quantile(0.25)), 3),
        "q3": round(float(s.quantile(0.75)), 3),
    }


def _histogram(series: pd.Series, bins: int = 40) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"counts": [], "edges": []}
    # Clip the extreme tail so the shape stays readable; note it in the payload.
    lo, hi = float(s.quantile(0.001)), float(s.quantile(0.999))
    if lo == hi:
        lo, hi = float(s.min()), float(s.max()) or 1.0
    counts, edges = np.histogram(s.clip(lo, hi), bins=bins)
    return {"counts": counts.tolist(), "edges": [round(e, 3) for e in edges.tolist()]}


def _boxplot_stats(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    q1, med, q3 = (float(s.quantile(q)) for q in (0.25, 0.5, 0.75))
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inliers = s[(s >= lo) & (s <= hi)]
    return {
        "q1": round(q1, 3), "median": round(med, 3), "q3": round(q3, 3),
        "lower": round(float(inliers.min()) if len(inliers) else lo, 3),
        "upper": round(float(inliers.max()) if len(inliers) else hi, 3),
        "n_outliers": int(len(s) - len(inliers)),
    }


def build_eda(orders: pd.DataFrame) -> dict:
    """Precompute the full EDA payload from the order-grain frame."""
    r = ColumnResolver(orders.columns)
    target = r.require(config.TARGET)
    profit = r.resolve(config.ECONOMIC_TARGET)
    mode = r.resolve(config.DECISION_VARIABLE)

    n = len(orders)
    late = orders[target]

    payload: dict = {
        "grain": "order",
        "n_orders": int(n),
        "late_count": int(late.sum()),
        "ontime_count": int(n - late.sum()),
        "late_rate": round(100 * float(late.mean()), 2),
    }

    # ---- Target analysis by each categorical dimension (spec section 9) -------------
    payload["target_by"] = {}
    for name in CATEGORICAL_CHOICES:
        col = r.resolve(name)
        if col is None:
            continue
        payload["target_by"][name] = _rate_table(orders, col, target, profit)

    # ---- Financial analysis (spec section 9) ---------------------------------------
    financial: dict = {}
    if profit:
        financial["profit_distribution"] = _histogram(orders[profit])
        financial["profit_stats"] = _describe(orders[profit])
        financial["profit_by_late"] = [
            {
                "level": "On time" if int(k) == 0 else "Late",
                "avg_profit": round(float(v.mean()), 2),
                "median_profit": round(float(v.median()), 2),
                "orders": int(v.size),
            }
            for k, v in orders.groupby(target, observed=True)[profit]
        ]
        for dim in ("Shipping Mode", "Market", "Order Region"):
            col = r.resolve(dim)
            if col:
                financial[f"profit_by_{dim.replace(' ', '_').lower()}"] = _rate_table(
                    orders, col, target, profit
                )
    ratio = r.resolve("Order Item Profit Ratio")
    if ratio:
        financial["profit_ratio_distribution"] = _histogram(orders[ratio])
        financial["profit_ratio_stats"] = _describe(orders[ratio])
    payload["financial"] = financial

    # ---- Selector payloads ---------------------------------------------------------
    payload["numeric_options"] = [n for n in NUMERIC_CHOICES if r.has(n)]
    payload["categorical_options"] = [c for c in CATEGORICAL_CHOICES if r.has(c)]

    payload["numeric_detail"] = {}
    for name in payload["numeric_options"]:
        col = r.require(name)
        payload["numeric_detail"][name] = {
            "stats": _describe(orders[col]),
            "histogram": _histogram(orders[col]),
            "box": _boxplot_stats(orders[col]),
        }

    payload["categorical_detail"] = {
        name: payload["target_by"][name]
        for name in payload["categorical_options"]
        if name in payload["target_by"]
    }

    # ---- Temporal ------------------------------------------------------------------
    order_date = r.resolve("order date (DateOrders)")
    if order_date is not None and pd.api.types.is_datetime64_any_dtype(orders[order_date]):
        od = orders[order_date]
        monthly = (
            orders.assign(_m=od.dt.to_period("M").astype(str))
            .groupby("_m", observed=True)
            .agg(orders=(target, "size"), late_rate=(target, "mean"),
                 profit=(profit, "sum") if profit else (target, "size"))
            .reset_index()
        )
        monthly["late_rate"] = (100 * monthly["late_rate"]).round(2)
        monthly["profit"] = monthly["profit"].round(2)
        payload["monthly"] = monthly.rename(columns={"_m": "month"}).to_dict("records")
        payload["date_min"] = str(od.min().date())
        payload["date_max"] = str(od.max().date())
        # 2018 is January only - flagged so nobody reads the tail as a real decline.
        year_counts = od.dt.year.value_counts().sort_index()
        payload["year_counts"] = {int(k): int(v) for k, v in year_counts.items()}
        last_year = int(year_counts.index.max())
        payload["partial_final_year"] = {
            "year": last_year,
            "orders": int(year_counts.iloc[-1]),
            "months_present": sorted(
                int(m) for m in od[od.dt.year == last_year].dt.month.unique()
            ),
        }

    # ---- Shipping-mode transit profile ---------------------------------------------
    # This is the mechanism behind every late rate on the page: which modes actually move
    # faster, versus which merely promise to.
    real = r.resolve("Days for shipping (real)")
    sched = r.resolve("Days for shipment (scheduled)")
    if mode and real and sched:
        prof = (
            orders.groupby(mode, observed=True)
            .agg(
                orders=(target, "size"),
                late_rate=(target, "mean"),
                actual_mean=(real, "mean"),
                actual_std=(real, "std"),
                actual_median=(real, "median"),
                promised=(sched, "first"),
            )
            .reset_index()
            .rename(columns={mode: "mode"})
        )
        prof["mode"] = prof["mode"].astype(str)
        prof["late_rate"] = (100 * prof["late_rate"]).round(2)
        for c in ("actual_mean", "actual_std", "actual_median"):
            prof[c] = prof[c].round(3)
        payload["mode_transit_profile"] = prof.sort_values(
            "actual_mean"
        ).to_dict("records")

        # Distribution of realised transit days within each mode.
        dist = (
            orders.groupby([mode, real], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        dist["share"] = dist.groupby(mode, observed=True)["n"].transform(
            lambda s: (100 * s / s.sum()).round(2)
        )
        dist[mode] = dist[mode].astype(str)
        payload["transit_distribution"] = dist.rename(
            columns={mode: "mode", real: "days"}
        ).to_dict("records")

    return payload


def build_correlations(orders: pd.DataFrame) -> dict:
    """
    Correlation matrix plus the strongest relationships with the target (section 10).

    Reported as association only. The page carries the explicit warning the spec requires,
    and names the columns excluded for being outcome-derived.
    """
    r = ColumnResolver(orders.columns)
    target = r.require(config.TARGET)

    cols = r.resolve_many(CORRELATION_FIELDS)
    cols = [c for c in cols if pd.api.types.is_numeric_dtype(orders[c])]
    numeric = orders[cols].apply(pd.to_numeric, errors="coerce")

    corr = numeric.corr(numeric_only=True).round(4)

    with np.errstate(invalid="ignore"):
        target_corr = (
            corr[target].drop(labels=[target], errors="ignore").dropna().sort_values()
        )

    positive = [
        {"variable": k, "correlation": round(float(v), 4)}
        for k, v in target_corr.tail(8).sort_values(ascending=False).items()
    ]
    negative = [
        {"variable": k, "correlation": round(float(v), 4)}
        for k, v in target_corr.head(8).items()
    ]

    return {
        "labels": list(corr.columns),
        "matrix": corr.to_numpy().tolist(),
        "top_positive": positive,
        "top_negative": negative,
        "warning": "Correlation indicates association, not causation.",
        "excluded_note": (
            "Delivery Status and Days for shipping (real) are excluded from this matrix: "
            "both are recorded after delivery and reconstruct the target almost exactly "
            "(100% and 97.6% respectively), so their correlations would be artefacts."
        ),
    }
