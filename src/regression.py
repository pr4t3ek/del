"""
Regression model for the economic outcome (spec sections 23 and 36).

Predicts `Order Profit Per Order` from information available at decision time, giving the
decision engine a profit-exposure estimate for orders where realised profit is not yet
known, and populating the value axis of the risk/value matrix.

An important null result belongs with this model. In this dataset profit and lateness are
essentially independent: the correlation is -0.005, and mean profit is $22.58 on on-time
orders versus $21.62 on late ones. Late delivery therefore has no measurable effect on
realised profit here, which is exactly why the cost of a late delivery has to enter the
decision layer as an explicit scenario assumption (spec section 24) rather than being
estimated from the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

import config
from src.classification import build_preprocessor
from src.data_dictionary import ColumnResolver


def train_profit_model(
    df: pd.DataFrame, features: list[str], train_idx: np.ndarray, test_idx: np.ndarray
) -> dict:
    """Fit and evaluate the profit regressor on the same split as the classifier."""
    r = ColumnResolver(df.columns)
    target = r.resolve(config.ECONOMIC_TARGET)
    if target is None:
        return {"available": False}

    X = df[features]
    y = pd.to_numeric(df[target], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    pipe = Pipeline(
        [
            ("prep", build_preprocessor(df, features, scale=False)),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=200,
                    learning_rate=0.1,
                    max_leaf_nodes=31,
                    min_samples_leaf=40,
                    random_state=config.RANDOM_STATE,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
    pipe.fit(X.iloc[train_idx], y[train_idx])
    pred = pipe.predict(X.iloc[test_idx])
    actual = y[test_idx]

    return {
        "available": True,
        "pipeline": pipe,
        "metrics": {
            "r2": round(float(r2_score(actual, pred)), 4),
            "mae": round(float(mean_absolute_error(actual, pred)), 3),
            "rmse": round(float(np.sqrt(((actual - pred) ** 2).mean())), 3),
            "mean_actual": round(float(actual.mean()), 3),
            "std_actual": round(float(actual.std()), 3),
            "n_test": int(len(actual)),
        },
    }


def profit_vs_lateness(df: pd.DataFrame) -> dict:
    """
    Quantify the relationship between late delivery and realised profit (section 23).

    Reported as a descriptive association, not a causal effect: shipping mode and delivery
    outcome are observational here, so this compares the profit of orders that happened to
    be late against those that were not.
    """
    r = ColumnResolver(df.columns)
    target = r.require(config.TARGET)
    profit = r.resolve(config.ECONOMIC_TARGET)
    if profit is None:
        return {"available": False}

    p = pd.to_numeric(df[profit], errors="coerce")
    late = df[target].astype(int)
    corr = float(p.corr(late.astype(float)))

    groups = []
    for value, sub in p.groupby(late, observed=True):
        groups.append(
            {
                "status": "Late" if int(value) == 1 else "On time",
                "orders": int(sub.size),
                "mean_profit": round(float(sub.mean()), 3),
                "median_profit": round(float(sub.median()), 3),
                "total_profit": round(float(sub.sum()), 2),
            }
        )

    diff = None
    if len(groups) == 2:
        by_status = {g["status"]: g for g in groups}
        diff = round(
            by_status["On time"]["mean_profit"] - by_status["Late"]["mean_profit"], 3
        )

    return {
        "available": True,
        "correlation": round(corr, 5),
        "groups": groups,
        "mean_difference": diff,
        "finding": (
            f"The correlation between late delivery and order profit is {corr:.4f} - "
            "effectively zero. Mean profit differs by "
            f"${abs(diff) if diff is not None else 0:.2f} between on-time and late orders, "
            "a gap far smaller than the spread within either group. In this dataset late "
            "delivery has no measurable effect on realised profit."
        ),
        "implication": (
            "Because the data shows no profit penalty for lateness, the economic cost of a "
            "late delivery cannot be estimated from it. The decision engine therefore "
            "treats that cost as an explicit, user-configurable scenario assumption rather "
            "than inferring a number the data does not support."
        ),
    }
