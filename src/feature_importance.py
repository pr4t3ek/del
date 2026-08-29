"""
Feature importance and business interpretation (spec sections 17-18).

Three complementary views, because each alone is misleading:

  * Tree impurity importance  - fast, but biased toward high-cardinality features.
  * Permutation importance    - measures the actual drop in held-out AUC when a feature is
                                shuffled; the honest measure, and the one ranked on.
  * Logistic odds ratios      - direction and magnitude on an interpretable scale.

Permutation importance is computed on the held-out set, not on training data, so a feature
that the model merely memorised does not score highly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config


def _group_onehot(names: list[str], values: np.ndarray, original: list[str]) -> list[dict]:
    """
    Collapse one-hot columns back to their source feature.

    'Market_Europe' and 'Market_LATAM' are two columns of one business variable; reporting
    them separately makes a single factor look like several weak ones.
    """
    totals: dict[str, float] = {}
    for name, value in zip(names, values):
        source = next(
            (o for o in sorted(original, key=len, reverse=True) if name.startswith(o)),
            name,
        )
        totals[source] = totals.get(source, 0.0) + float(value)
    return [{"feature": k, "importance": v} for k, v in totals.items()]


def permutation_view(
    pipeline, X_test: pd.DataFrame, y_test: np.ndarray, n_repeats: int = 5
) -> list[dict]:
    """
    Permutation importance on the raw (pre-transform) feature columns.

    Shuffling the input column rather than a transformed one keeps the result readable and
    automatically aggregates a feature's one-hot expansion.
    """
    result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=n_repeats,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    rows = [
        {
            "feature": col,
            "importance": round(float(m), 6),
            "std": round(float(s), 6),
        }
        for col, m, s in zip(X_test.columns, result.importances_mean, result.importances_std)
    ]
    rows.sort(key=lambda d: -d["importance"])
    return rows


def tree_view(pipeline, original_features: list[str]) -> list[dict]:
    """Impurity-based importance from a tree model, grouped back to source features."""
    model = pipeline.named_steps.get("model")
    prep = pipeline.named_steps.get("prep")
    if model is None or not hasattr(model, "feature_importances_"):
        return []
    try:
        names = [str(n) for n in prep.get_feature_names_out()]
    except Exception:                                       # pragma: no cover - defensive
        return []

    grouped = _group_onehot(names, model.feature_importances_, original_features)
    grouped.sort(key=lambda d: -d["importance"])
    for g in grouped:
        g["importance"] = round(g["importance"], 6)
    return grouped


def coefficient_view(pipeline, original_features: list[str]) -> list[dict]:
    """Absolute standardised logistic coefficients, grouped back to source features."""
    model = pipeline.named_steps.get("model")
    prep = pipeline.named_steps.get("prep")
    if model is None or not hasattr(model, "coef_"):
        return []
    try:
        names = [str(n) for n in prep.get_feature_names_out()]
    except Exception:                                       # pragma: no cover - defensive
        return []

    grouped = _group_onehot(names, np.abs(model.coef_[0]), original_features)
    grouped.sort(key=lambda d: -d["importance"])
    for g in grouped:
        g["importance"] = round(g["importance"], 6)
    return grouped


# --------------------------------------------------------------------------------------
# Business interpretation (spec section 18)
# --------------------------------------------------------------------------------------
def build_interpretation(
    permutation: list[dict], stats_results: list[dict], mode_profile: list[dict]
) -> list[dict]:
    """
    Turn the top drivers into the four-part narrative section 18 asks for:
    driver, statistical interpretation, business interpretation, action.

    Language stays associational throughout. Shipping-mode assignment in this data is
    observational, so nothing here claims that switching a mode *causes* the difference.
    """
    effect_by_factor = {
        r["factor"]: r for r in stats_results if r.get("type") == "categorical"
    }
    blocks = []

    for row in permutation[:10]:
        feature = row["feature"]
        importance = row["importance"]
        test = effect_by_factor.get(feature)

        if importance < 0.001:
            statistical = (
                f"Shuffling {feature} changes held-out AUC by {importance:.5f} - "
                "indistinguishable from zero."
            )
            business = (
                f"{feature} carries no usable information about delivery delay in this "
                "dataset. Any apparent pattern in a univariate chart is sampling noise."
            )
            action = "Do not build policy on this factor."
        elif feature == config.DECISION_VARIABLE:
            promised = {m["mode"]: m for m in mode_profile}
            spread = ""
            if promised:
                worst = max(promised.values(), key=lambda m: m["late_rate"])
                best = min(promised.values(), key=lambda m: m["late_rate"])
                spread = (
                    f" Historically the late rate ranges from {best['late_rate']:.1f}% "
                    f"({best['mode']}) to {worst['late_rate']:.1f}% ({worst['mode']})."
                )
            statistical = (
                f"Permutation importance {importance:.4f} - by far the largest single "
                f"contribution to held-out AUC.{spread}"
            )
            business = (
                "Shipping mode is the dominant factor associated with late delivery. The "
                "association runs mainly through the delivery promise attached to each "
                "mode rather than through how fast the goods actually move: modes with "
                "tighter promises miss them more often."
            )
            action = (
                "Treat mode selection as the primary lever, and evaluate it on expected "
                "total cost in the decision engine rather than on late rate alone."
            )
        else:
            effect_note = ""
            if test:
                effect_note = (
                    f" {test['effect_name']} = {test['effect_size']:.4f} "
                    f"({test['effect_label'].lower()})."
                )
            statistical = (
                f"Permutation importance {importance:.4f} on held-out data.{effect_note}"
            )
            business = (
                f"{feature} shows a measurable but small association with delivery delay. "
                "It refines risk estimates at the margin rather than driving them."
            )
            action = "Use as a secondary segmentation, not as a primary decision rule."

        blocks.append(
            {
                "driver": feature,
                "importance": importance,
                "statistical": statistical,
                "business": business,
                "action": action,
            }
        )

    return blocks
