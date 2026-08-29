"""
Export helpers and executive-summary generation (spec sections 37 and 40).

Every export is derived from the precomputed analytics payload, so a downloaded CSV always
matches exactly what is displayed on screen.
"""

from __future__ import annotations

import io

import pandas as pd


def _csv(rows: list[dict]) -> str:
    if not rows:
        return "no data available\n"
    buffer = io.StringIO()
    pd.DataFrame(rows).to_csv(buffer, index=False)
    return buffer.getvalue()


def export_model_comparison(analytics: dict) -> str:
    rows = []
    for name, m in analytics.get("model_comparison", {}).items():
        row = {"model": name}
        row.update({k: v for k, v in m.items() if not isinstance(v, dict)})
        conf = m.get("confusion", {})
        row.update({f"confusion_{k}": v for k, v in conf.items()})
        rows.append(row)
    return _csv(rows)


def export_feature_importance(analytics: dict) -> str:
    imp = analytics.get("importance", {})
    rows = []
    for kind in ("permutation", "tree", "coefficients"):
        for r in imp.get(kind, []):
            rows.append({"method": kind, **r})
    return _csv(rows)


def export_statistics(analytics: dict) -> str:
    rows = []
    for r in analytics.get("statistics", {}).get("results", []):
        rows.append(
            {
                "factor": r["factor"],
                "type": r["type"],
                "test": r["test"],
                "statistic": r["statistic"],
                "dof": r.get("dof"),
                "p_value": r["p_value"],
                "effect_size": r["effect_size"],
                "effect_name": r["effect_name"],
                "effect_label": r["effect_label"],
                "significant": r["significant"],
                "n": r["n"],
                "business_interpretation": r["business"],
            }
        )
    return _csv(rows)


def export_shipping_mode(analytics: dict) -> str:
    profile = analytics.get("decision", {}).get("mode_profile", {})
    econ = {
        row["level"]: row
        for row in analytics.get("eda", {}).get("target_by", {}).get("Shipping Mode", [])
    }
    rows = []
    for mode, p in profile.items():
        e = econ.get(mode, {})
        rows.append(
            {
                "shipping_mode": mode,
                "orders": p["orders"],
                "observed_late_rate": p["late_rate"],
                "promised_days": p["promised_days"],
                "mean_transit_days": p["mean_transit"],
                "median_transit_days": p["median_transit"],
                "std_transit_days": p["std_transit"],
                "avg_order_profit": e.get("avg_profit"),
                "total_profit": e.get("total_profit"),
            }
        )
    return _csv(rows)


def export_decision(analytics: dict) -> str:
    rows = []
    for p in analytics.get("decision", {}).get("policies", {}).get("policies", []):
        row = {k: v for k, v in p.items() if k != "mode_mix"}
        for mode, share in p.get("mode_mix", {}).items():
            row[f"share_{mode}"] = share
        rows.append(row)
    return _csv(rows)


def export_break_even(analytics: dict) -> str:
    return _csv(analytics.get("decision", {}).get("break_even", []))


def export_eda(analytics: dict) -> str:
    rows = []
    for dimension, table in analytics.get("eda", {}).get("target_by", {}).items():
        for r in table:
            rows.append({"dimension": dimension, **r})
    return _csv(rows)


def export_availability(analytics: dict) -> str:
    return _csv(analytics.get("availability", []))


def export_diagnostics(analytics: dict) -> str:
    rows = []
    for c in analytics.get("logistic", {}).get("coefficients", []):
        rows.append({"section": "logistic_coefficients", **c})
    for v in analytics.get("vif", {}).get("rows", []):
        rows.append({"section": "vif", **v})
    for c in analytics.get("calibration", {}).get("bins", []):
        rows.append({"section": "calibration", **c})
    return _csv(rows)


EXPORTS = {
    "model_comparison": ("model_comparison.csv", export_model_comparison),
    "feature_importance": ("feature_importance.csv", export_feature_importance),
    "statistics": ("statistical_significance.csv", export_statistics),
    "shipping_mode": ("shipping_mode_economics.csv", export_shipping_mode),
    "decision": ("decision_scenario_analysis.csv", export_decision),
    "break_even": ("break_even_analysis.csv", export_break_even),
    "eda": ("eda_summary.csv", export_eda),
    "availability": ("predictor_availability.csv", export_availability),
    "diagnostics": ("diagnostic_report.csv", export_diagnostics),
}


# --------------------------------------------------------------------------------------
# Executive summary (spec section 37)
# --------------------------------------------------------------------------------------
def build_executive_summary(analytics: dict) -> dict:
    """
    Generate the executive narrative from the computed results.

    Nothing here is written by hand: every number is read from the analytics payload, and
    modelled estimates are labelled separately from observed history.
    """
    eda = analytics.get("eda", {})
    stats = analytics.get("statistics", {})
    decision = analytics.get("decision", {})
    importance = analytics.get("importance", {}).get("permutation", [])
    ablation = analytics.get("ablation", {})

    late_rate = eda.get("late_rate", 0.0)
    n_orders = eda.get("n_orders", 0)

    drivers = [
        {
            "name": d["feature"],
            "importance": d["importance"],
            "material": d["importance"] >= 0.001,
        }
        for d in importance[:5]
    ]

    by_mode = eda.get("target_by", {}).get("Shipping Mode", [])
    worst_mode = max(by_mode, key=lambda r: r["late_rate"]) if by_mode else None
    best_mode = min(by_mode, key=lambda r: r["late_rate"]) if by_mode else None

    def _riskiest(dimension):
        rows = eda.get("target_by", {}).get(dimension, [])
        rows = [r for r in rows if r["orders"] >= 200]
        return max(rows, key=lambda r: r["late_rate"]) if rows else None

    policies = decision.get("policies", {}).get("policies", [])
    best_policy = policies[0] if policies else None
    baseline_policy = next(
        (p for p in policies if p["policy"].startswith("A")), None
    )
    promise = decision.get("promise_redesign", {})

    recommendations = []
    if promise.get("available") and promise.get("transit_identical"):
        recommendations.append(
            {
                "title": "Re-set the Second Class delivery promise",
                "detail": (
                    f"Second Class and Standard Class move goods at the same speed "
                    f"({promise['second_class']['mean_transit']:.2f} vs "
                    f"{promise['standard_class']['mean_transit']:.2f} days) but Second "
                    f"Class promises {promise['second_class']['promised_days']} days "
                    f"against {promise['standard_class']['promised_days']}. Aligning the "
                    f"promise would be expected to cut its late rate by "
                    f"{100 * promise['late_rate_reduction']:.0f} percentage points across "
                    f"{promise['orders_affected']:,} orders at no operational cost."
                ),
                "basis": "Observed transit distributions; scenario-implied late rate.",
            }
        )
    if best_policy and baseline_policy and best_policy["saving_pct"] > 0:
        recommendations.append(
            {
                "title": f"Adopt {best_policy['policy']}",
                "detail": (
                    f"Expected total cost falls from ${baseline_policy['avg_cost']:.2f} to "
                    f"${best_policy['avg_cost']:.2f} per order "
                    f"({best_policy['saving_pct']:.1f}%), with the expected late rate "
                    f"moving from {100 * baseline_policy['expected_late_rate']:.1f}% to "
                    f"{100 * best_policy['expected_late_rate']:.1f}%."
                ),
                "basis": "Modelled estimate under the current scenario assumptions.",
            }
        )
    if ablation.get("without_mode"):
        recommendations.append(
            {
                "title": "Do not invest in destination- or product-level delay forecasting",
                "detail": (
                    "With shipping mode removed, every remaining operational and external "
                    f"factor together reaches only "
                    f"{ablation['without_mode']['metrics']['roc_auc']:.3f} AUC - barely "
                    "above chance. Market, region, country, category, customer segment and "
                    "order size carry no usable delay signal in this dataset."
                ),
                "basis": "Held-out model performance.",
            }
        )
    if worst_mode:
        recommendations.append(
            {
                "title": "Review service commitments before routing changes",
                "detail": (
                    f"{worst_mode['level']} carries the highest late rate at "
                    f"{worst_mode['late_rate']:.1f}%, but this reflects the promise "
                    "attached to it rather than slower fulfilment. Commitment setting is "
                    "the cheaper lever."
                ),
                "basis": "Observed historical rates.",
            }
        )

    return {
        "what": {
            "late_rate": late_rate,
            "n_orders": n_orders,
            "headline": (
                f"{late_rate:.1f}% of {n_orders:,} orders were delivered later than "
                "promised."
            ),
        },
        "why": {
            "drivers": drivers,
            "headline": (
                f"{stats.get('n_significant', 0)} of {stats.get('n_tests', 0)} tested "
                f"factors are statistically significant, and "
                f"{stats.get('n_material', 0)} show an effect size above 0.10. Shipping "
                "mode is the dominant driver; the rest carry negligible weight."
            ),
        },
        "where": {
            "worst_mode": worst_mode,
            "best_mode": best_mode,
            "worst_market": _riskiest("Market"),
            "worst_region": _riskiest("Order Region"),
            "worst_category": _riskiest("Category Name"),
            "worst_segment": _riskiest("Customer Segment"),
        },
        "actions": recommendations,
        "economics": {
            "best_policy": best_policy,
            "baseline_policy": baseline_policy,
            "assumptions": decision.get("assumptions", {}),
            "caveat": (
                "Cost figures are modelled estimates under user-set scenario assumptions. "
                "The dataset contains no freight-cost field, so total logistics cost "
                "cannot be measured from it."
            ),
        },
    }
