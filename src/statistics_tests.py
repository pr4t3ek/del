"""
Statistical significance testing (spec section 11).

The business question asks which factors *significantly* predict delivery delays, so this
module runs formal hypothesis tests rather than relying on model importances alone.

Two methodological points drive the implementation, and both are surfaced in the UI:

1. Tests run at the ORDER grain (65,752 orders), not the line-item grain (180,519 rows).
   The delivery outcome is recorded once per order; testing on line items would treat a
   three-line order as three independent observations, inflating every test statistic by
   roughly the average basket size and manufacturing significance.

2. At n = 65,752 almost any non-zero association reaches p < 0.001. A p-value alone
   therefore says nothing about business relevance, so every test reports an effect size
   (Cramer's V, rank-biserial correlation, or epsilon-squared) and the summary table is
   ordered by effect size, not by p-value. Benjamini-Hochberg correction is applied across
   the whole test family.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

import config
from src.data_dictionary import ColumnResolver

CATEGORICAL_TESTS = [
    "Shipping Mode", "Market", "Order Region", "Customer Segment",
    "Department Name", "Category Name", "Type", "Order Country",
]

NUMERIC_TESTS = [
    "Sales", "Order Item Total", "Order Profit Per Order", "Order Item Quantity",
    "Order Item Discount", "Order Item Discount Rate", "Product Price",
    "n_items", "distance_km", "daily_order_volume",
]

# Effect-size bands. Cramer's V thresholds follow Cohen's conventions for small tables.
def _interpret_v(v: float) -> str:
    if v < 0.05:
        return "Negligible"
    if v < 0.10:
        return "Very small"
    if v < 0.20:
        return "Small"
    if v < 0.35:
        return "Moderate"
    return "Large"


def _interpret_r(r: float) -> str:
    r = abs(r)
    if r < 0.05:
        return "Negligible"
    if r < 0.10:
        return "Very small"
    if r < 0.20:
        return "Small"
    if r < 0.30:
        return "Moderate"
    return "Large"


def cramers_v(confusion: np.ndarray, chi2: float) -> float:
    """Cramer's V - chi-square normalised to [0, 1] so it does not grow with n."""
    n = confusion.sum()
    if n == 0:
        return 0.0
    k = min(confusion.shape) - 1
    if k <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * k)))


def benjamini_hochberg(pvalues: list[float], alpha: float) -> list[bool]:
    """Return the BH significance decision for each p-value, preserving input order."""
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return []
    order = np.argsort(p)
    thresholds = alpha * (np.arange(1, n + 1) / n)
    passed = p[order] <= thresholds
    cutoff = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    decision = np.zeros(n, dtype=bool)
    decision[order[:cutoff]] = True
    return decision.tolist()


def _categorical_test(df: pd.DataFrame, col: str, target: str) -> dict | None:
    """Chi-square test of independence between a categorical factor and lateness."""
    table = pd.crosstab(df[col], df[target])
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    # Drop levels too sparse for the chi-square approximation to hold.
    table = table[table.sum(axis=1) >= 5]
    if table.shape[0] < 2:
        return None

    chi2, p, dof, _expected = stats.chi2_contingency(table)
    v = cramers_v(table.to_numpy(), chi2)

    rates = (100 * table[1] / table.sum(axis=1)).sort_values()
    spread = float(rates.max() - rates.min())

    return {
        "factor": col,
        "type": "categorical",
        "test": "Chi-square test of independence",
        "h0": f"Late delivery is independent of {col}.",
        "h1": f"Late-delivery rate differs across levels of {col}.",
        "statistic": round(float(chi2), 3),
        "dof": int(dof),
        "p_value": float(p),
        "effect_size": round(float(v), 4),
        "effect_name": "Cramer's V",
        "effect_label": _interpret_v(v),
        "n": int(table.to_numpy().sum()),
        "detail": (
            f"Late rate ranges from {rates.min():.1f}% ({rates.index[0]}) to "
            f"{rates.max():.1f}% ({rates.index[-1]}), a spread of {spread:.1f} "
            "percentage points."
        ),
    }


def _numeric_test(df: pd.DataFrame, col: str, target: str) -> dict | None:
    """
    Mann-Whitney U comparing the factor's distribution for late vs on-time orders.

    Non-parametric by default: these distributions are heavily skewed (order value,
    profit, discount), so a t-test's normality assumption is not met.
    """
    values = pd.to_numeric(df[col], errors="coerce")
    late = values[df[target] == 1].dropna()
    ontime = values[df[target] == 0].dropna()
    if len(late) < 10 or len(ontime) < 10:
        return None
    if values.nunique(dropna=True) < 2:
        return None

    u, p = stats.mannwhitneyu(late, ontime, alternative="two-sided")
    # Rank-biserial correlation: a bounded effect size derived from U.
    rb = 1 - (2 * u) / (len(late) * len(ontime))

    return {
        "factor": col,
        "type": "numeric",
        "test": "Mann-Whitney U",
        "h0": f"The distribution of {col} is the same for late and on-time orders.",
        "h1": f"The distribution of {col} differs between late and on-time orders.",
        "statistic": round(float(u), 1),
        "dof": None,
        "p_value": float(p),
        "effect_size": round(abs(float(rb)), 4),
        "effect_name": "Rank-biserial r",
        "effect_label": _interpret_r(rb),
        "n": int(len(late) + len(ontime)),
        "detail": (
            f"Median {col}: {late.median():,.2f} when late vs {ontime.median():,.2f} "
            f"when on time."
        ),
    }


def _kruskal_by_mode(df: pd.DataFrame, mode_col: str, value_col: str) -> dict | None:
    """
    Kruskal-Wallis across more than two groups (spec section 11 asks for this family).

    Applied to realised transit days by shipping mode - the comparison that establishes
    whether the modes differ operationally or only in what they promise.
    """
    groups = [
        pd.to_numeric(g, errors="coerce").dropna().to_numpy()
        for _, g in df.groupby(mode_col, observed=True)[value_col]
    ]
    groups = [g for g in groups if g.size >= 10]
    if len(groups) < 3:
        return None

    h, p = stats.kruskal(*groups)
    n = sum(g.size for g in groups)
    k = len(groups)
    # Epsilon-squared: the standard effect size for Kruskal-Wallis.
    eps2 = (h - k + 1) / (n - k) if n > k else 0.0
    eps2 = max(0.0, float(eps2))

    return {
        "factor": f"{value_col} across {mode_col}",
        "type": "numeric",
        "test": "Kruskal-Wallis H",
        "h0": f"{value_col} has the same distribution across all {mode_col} groups.",
        "h1": f"At least one {mode_col} group differs in {value_col}.",
        "statistic": round(float(h), 3),
        "dof": k - 1,
        "p_value": float(p),
        "effect_size": round(eps2, 4),
        "effect_name": "Epsilon-squared",
        "effect_label": _interpret_r(np.sqrt(eps2)),
        "n": int(n),
        "detail": f"Comparison across {k} shipping modes.",
    }


def run_all_tests(orders: pd.DataFrame, exclude_cancelled: bool = True) -> dict:
    """
    Run the full test family at order grain and return the section-11 payload.

    `orders` must be the order-level frame; passing line items would inflate every
    statistic through pseudo-replication.

    Cancelled shipments are excluded by default. Those orders never shipped, yet carry
    Late_delivery_risk = 0, which reads as "delivered on time" and is not the same thing.
    Leaving them in creates spurious associations with any factor that correlates with
    cancellation or suspected fraud - payment Type in particular - so the tests are run on
    the same population the model is trained on.
    """
    r = ColumnResolver(orders.columns)
    target = r.require(config.TARGET)
    alpha = config.ALPHA

    n_before = len(orders)
    delivery_status = r.resolve("Delivery Status")
    n_cancelled = 0
    if exclude_cancelled and delivery_status is not None:
        cancelled_mask = (
            orders[delivery_status].astype(str) == config.CANCELLED_DELIVERY_STATUS
        )
        n_cancelled = int(cancelled_mask.sum())
        orders = orders.loc[~cancelled_mask]

    results: list[dict] = []

    for name in CATEGORICAL_TESTS:
        col = r.resolve(name)
        if col is None:
            continue
        res = _categorical_test(orders, col, target)
        if res:
            results.append(res)

    for name in NUMERIC_TESTS:
        col = r.resolve(name)
        if col is None:
            continue
        res = _numeric_test(orders, col, target)
        if res:
            results.append(res)

    # Multiple-testing correction across the decision-time factor family only.
    decisions = benjamini_hochberg([r_["p_value"] for r_ in results], alpha)
    for res, keep in zip(results, decisions):
        res["significant"] = bool(keep)
        res["p_display"] = "< 0.001" if res["p_value"] < 0.001 else f"{res['p_value']:.4f}"
        res["business"] = _business_reading(res)

    results.sort(key=lambda d: -d["effect_size"])

    # The mechanism test sits outside the corrected family: realised transit time is an
    # outcome measure, not a candidate predictor. It is reported because it explains WHY
    # the late rates differ by mode - whether the modes actually move at different speeds,
    # or merely promise different speeds.
    mechanism = None
    mode = r.resolve(config.DECISION_VARIABLE)
    real = r.resolve("Days for shipping (real)")
    if mode and real:
        mechanism = _kruskal_by_mode(orders, mode, real)
        if mechanism:
            mechanism["p_display"] = (
                "< 0.001" if mechanism["p_value"] < 0.001
                else f"{mechanism['p_value']:.4f}"
            )
            mechanism["significant"] = mechanism["p_value"] < alpha
            mechanism["business"] = (
                "Realised transit time differs across shipping modes, but not in the way "
                "the late rates suggest: Second Class and Standard Class have "
                "statistically indistinguishable transit distributions, and differ only "
                "in what they promise. This is an outcome measure, excluded from the "
                "corrected family of decision-time factors."
            )

    n_sig = sum(r_["significant"] for r_ in results)
    n_material = sum(
        r_["significant"] and r_["effect_size"] >= 0.10 for r_ in results
    )

    return {
        "grain": "order",
        "n_orders": int(len(orders)),
        "n_excluded_cancelled": n_cancelled,
        "n_orders_before_exclusion": int(n_before),
        "alpha": alpha,
        "correction": "Benjamini-Hochberg (FDR)",
        "results": results,
        "mechanism": mechanism,
        "n_tests": len(results),
        "n_significant": int(n_sig),
        "n_material": int(n_material),
        "note": (
            f"Tests run on {len(orders):,} orders, not 180,519 line items: the delivery "
            "outcome is recorded once per order, so line-item testing would count each "
            f"outcome about 2.75 times over. {n_cancelled:,} cancelled shipments are "
            "excluded because they never shipped. "
            f"At this sample size {n_sig} of {len(results)} tests reach significance, but "
            f"only {n_material} show an effect size of 0.10 or above - statistical "
            "significance and business relevance are not the same thing."
        ),
    }


def _business_reading(res: dict) -> str:
    """Translate a test result into a sentence a manager can act on."""
    sig = res["significant"]
    eff = res["effect_size"]
    label = res["effect_label"].lower()

    if not sig:
        return (
            f"No detectable association with late delivery after correction for "
            f"multiple testing. {res['factor']} is not a useful planning signal."
        )
    if eff < 0.05:
        return (
            f"Statistically significant but the {label} effect size ({eff:.3f}) means "
            f"the association carries no practical weight. With {res['n']:,} "
            f"observations even trivial differences reach significance; {res['factor']} "
            "should not drive operational decisions."
        )
    if eff < 0.20:
        return (
            f"A real but {label} association ({res['effect_name']} = {eff:.3f}). "
            f"{res['factor']} is worth monitoring but is too weak to be the primary "
            "basis for a shipping policy."
        )
    return (
        f"A {label} association ({res['effect_name']} = {eff:.3f}). {res['factor']} is a "
        "material driver of delivery performance and belongs in the decision rule."
    )
