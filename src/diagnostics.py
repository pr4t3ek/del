"""
Statistical diagnostics for the logistic model (spec sections 19-21).

Provides what a statistics audience expects alongside the ML metrics: log-likelihood,
AIC/BIC, deviance, a coefficient table with odds ratios and confidence intervals,
multicollinearity diagnostics, and calibration.

On multicollinearity, note what this dataset does to `Days for shipment (scheduled)`: it
is a deterministic lookup on Shipping Mode (Same Day->0, First Class->1, Second Class->2,
Standard Class->4). Fitting both together produces a genuinely singular design, which is
why the VIF section deliberately includes the pair - it makes the diagnostic concrete
instead of hypothetical.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

import config
from src.data_dictionary import ColumnResolver

# Compact, interpretable feature set for the inference model. Deliberately smaller than
# the ML feature set: odds ratios across 250 one-hot columns are not readable, and
# section 20 warns against computing VIF blindly on a wide sparse matrix.
INFERENCE_CATEGORICAL = [
    "Shipping Mode", "Market", "Customer Segment", "Department Name", "Type",
]
INFERENCE_NUMERIC = [
    "Order Item Quantity", "Product Price", "Order Item Discount Rate",
    "distance_km", "daily_order_volume", "order_month", "order_dayofweek",
]


def _design_matrix(
    df: pd.DataFrame, categorical: list[str], numeric: list[str]
) -> tuple[pd.DataFrame, dict]:
    """Build a readable design matrix with explicit reference categories."""
    frames, references = [], {}

    for col in categorical:
        s = df[col].astype(str)
        # Reference = the most frequent level, so odds ratios read against the norm.
        ref = s.value_counts().index[0]
        references[col] = ref
        dummies = pd.get_dummies(s, prefix=col, drop_first=False, dtype=float)
        dummies = dummies.drop(columns=[f"{col}_{ref}"], errors="ignore")
        frames.append(dummies)

    if numeric:
        frames.append(df[numeric].astype(float).reset_index(drop=True))

    X = pd.concat([f.reset_index(drop=True) for f in frames], axis=1)
    X.columns = [str(c).replace(" ", "_") for c in X.columns]
    return X, references


def fit_logistic_inference(df: pd.DataFrame, include_scheduled: bool = False) -> dict:
    """
    Fit a statsmodels logistic regression for interpretation (spec section 19).

    include_scheduled adds `Days for shipment (scheduled)` alongside Shipping Mode. That
    pair is perfectly collinear here, so it is off by default and used only to demonstrate
    the multicollinearity diagnostic.
    """
    r = ColumnResolver(df.columns)
    target = r.require(config.TARGET)

    categorical = r.resolve_many(INFERENCE_CATEGORICAL)
    numeric = [c for c in r.resolve_many(INFERENCE_NUMERIC)
               if pd.api.types.is_numeric_dtype(df[c])]
    if include_scheduled:
        sched = r.resolve("Days for shipment (scheduled)")
        if sched:
            numeric = numeric + [sched]

    X, references = _design_matrix(df, categorical, numeric)
    y = df[target].to_numpy().astype(int)

    X_const = sm.add_constant(X, has_constant="add")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = sm.Logit(y, X_const)
            fit = model.fit(disp=False, maxiter=100, method="newton")
        except Exception:
            # Singular or non-converging design - fall back to a regularised fit so the
            # page still renders rather than 500-ing.
            fit = sm.Logit(y, X_const).fit_regularized(disp=False, alpha=1e-6)

    params = fit.params
    try:
        conf = fit.conf_int()
        conf.columns = ["lo", "hi"]
        pvalues = fit.pvalues
    except Exception:                                       # pragma: no cover - defensive
        conf = pd.DataFrame({"lo": params, "hi": params})
        pvalues = pd.Series(np.nan, index=params.index)

    rows = []
    for name in params.index:
        coef = float(params[name])
        lo, hi = float(conf.loc[name, "lo"]), float(conf.loc[name, "hi"])
        p = float(pvalues.get(name, np.nan))
        odds = float(np.exp(np.clip(coef, -30, 30)))
        rows.append(
            {
                "variable": name,
                "coefficient": round(coef, 4),
                "odds_ratio": round(odds, 4),
                "p_value": p,
                "p_display": "< 0.001" if p < 0.001 else f"{p:.4f}",
                "ci_low": round(float(np.exp(np.clip(lo, -30, 30))), 4),
                "ci_high": round(float(np.exp(np.clip(hi, -30, 30))), 4),
                "significant": bool(p < config.ALPHA) if np.isfinite(p) else False,
                "interpretation": _odds_reading(name, odds, p),
            }
        )

    rows.sort(key=lambda d: -abs(d["coefficient"]))

    n = len(y)
    k = int(X_const.shape[1])
    llf = float(getattr(fit, "llf", np.nan))
    llnull = float(getattr(fit, "llnull", np.nan))

    return {
        "n_observations": int(n),
        "n_parameters": k,
        "log_likelihood": round(llf, 2) if np.isfinite(llf) else None,
        "ll_null": round(llnull, 2) if np.isfinite(llnull) else None,
        "aic": round(float(fit.aic), 2) if np.isfinite(getattr(fit, "aic", np.nan)) else None,
        "bic": round(float(fit.bic), 2) if np.isfinite(getattr(fit, "bic", np.nan)) else None,
        "deviance": round(-2 * llf, 2) if np.isfinite(llf) else None,
        "pseudo_r2": (
            round(float(1 - llf / llnull), 4)
            if np.isfinite(llf) and np.isfinite(llnull) and llnull != 0
            else None
        ),
        "coefficients": rows,
        "references": references,
        "features": list(X.columns),
    }


def _odds_reading(name: str, odds: float, p: float) -> str:
    if name == "const":
        return "Baseline log-odds when every predictor is at its reference level or zero."
    if not np.isfinite(p) or p >= config.ALPHA:
        return "Not statistically distinguishable from no effect."
    if odds > 1.05:
        return f"Associated with {odds:.2f}x higher odds of late delivery than the reference."
    if odds < 0.95:
        return f"Associated with {odds:.2f}x the odds of late delivery - i.e. lower risk."
    return "Statistically significant but the odds ratio is close to 1; negligible effect."


# --------------------------------------------------------------------------------------
# Multicollinearity (spec section 20)
# --------------------------------------------------------------------------------------
def compute_vif(df: pd.DataFrame, include_scheduled: bool = True) -> dict:
    """
    Variance Inflation Factors on the interpretable feature set.

    Computed on numeric predictors plus the shipping-mode indicators, not on the full
    250-column one-hot matrix where VIF is uninformative. Including
    `Days for shipment (scheduled)` alongside Shipping Mode is intentional: the two are
    algebraically identical in this dataset, so the diagnostic should - and does - flag it.
    """
    r = ColumnResolver(df.columns)
    numeric = [c for c in r.resolve_many(INFERENCE_NUMERIC)
               if pd.api.types.is_numeric_dtype(df[c])]

    frames = [df[numeric].astype(float).reset_index(drop=True)]
    mode = r.resolve(config.DECISION_VARIABLE)
    if mode:
        s = df[mode].astype(str)
        ref = s.value_counts().index[0]
        dummies = pd.get_dummies(s, prefix="Mode", dtype=float).drop(
            columns=[f"Mode_{ref}"], errors="ignore"
        )
        frames.append(dummies.reset_index(drop=True))
    if include_scheduled:
        sched = r.resolve("Days for shipment (scheduled)")
        if sched:
            frames.append(df[[sched]].astype(float).reset_index(drop=True))

    X = pd.concat(frames, axis=1)
    X = X.loc[:, X.std(numeric_only=True) > 0]

    rows = []
    arr = X.to_numpy(dtype=float)
    for i, col in enumerate(X.columns):
        others = np.delete(arr, i, axis=1)
        target = arr[:, i]
        try:
            coef, *_ = np.linalg.lstsq(
                np.column_stack([np.ones(len(others)), others]), target, rcond=None
            )
            pred = np.column_stack([np.ones(len(others)), others]) @ coef
            ss_res = float(((target - pred) ** 2).sum())
            ss_tot = float(((target - target.mean()) ** 2).sum())
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = float("inf") if r2 >= 1 - 1e-10 else 1.0 / (1.0 - r2)
        except Exception:                                   # pragma: no cover - defensive
            vif = float("nan")

        rows.append(
            {
                "variable": str(col),
                "vif": None if not np.isfinite(vif) else round(vif, 3),
                "vif_display": "Infinite" if not np.isfinite(vif) else f"{vif:.2f}",
                "interpretation": _vif_reading(vif),
            }
        )

    rows.sort(key=lambda d: -(d["vif"] if d["vif"] is not None else 1e18))
    problematic = [r_ for r_ in rows if r_["vif"] is None or r_["vif"] > 10]

    return {
        "rows": rows,
        "n_problematic": len(problematic),
        "note": (
            "VIF is computed on the interpretable predictor set - numeric features plus "
            "shipping-mode indicators - not on the full one-hot design matrix, where the "
            "measure would be dominated by the mutual exclusivity of dummy columns and "
            "would not mean anything."
        ),
        "finding": (
            "Days for shipment (scheduled) and Shipping Mode return infinite or extreme "
            "VIF because they are algebraically the same variable in this dataset: the "
            "promised window is a fixed lookup on the mode (Same Day 0, First Class 1, "
            "Second Class 2, Standard Class 4). Only Shipping Mode is kept in the "
            "predictive models; retaining both would make the design singular and the "
            "coefficients uninterpretable."
        ),
    }


def _vif_reading(vif: float) -> str:
    if not np.isfinite(vif):
        return "Perfectly collinear with another predictor - not estimable."
    if vif > 10:
        return "Severe multicollinearity; coefficient is unstable."
    if vif > 5:
        return "Moderate multicollinearity; interpret the coefficient cautiously."
    return "Acceptable."
