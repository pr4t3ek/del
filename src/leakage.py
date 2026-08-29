"""
Predictor availability and data-leakage screening (spec section 3).

The question this module answers for every column is: *would a planner actually know
this at the moment the shipping-mode decision is made?* Anything that only becomes known
once the shipment has happened is excluded, no matter how predictive it looks.

Why this matters on this dataset specifically. Three excluded columns reconstruct the
target almost perfectly:

  * `Delivery Status`         - equals the target exactly on 100% of rows
                                ("Late delivery" <=> Late_delivery_risk == 1)
  * `Days for shipping (real)`- realised transit; > scheduled reproduces the target on 97.6%
  * `shipping date (DateOrders)` - subtracting the order date recovers realised transit

Including any of them yields the ~99% accuracy commonly reported for this dataset. That
number is an artefact of scoring the outcome against itself, not a forecast.

A fourth column needs different handling. `Days for shipment (scheduled)` IS known at
decision time, but in this dataset it is a deterministic lookup on Shipping Mode
(Same Day->0, First Class->1, Second Class->2, Standard Class->4), so the two are
perfectly collinear. It is kept for the statistical model, where it makes the
multicollinearity diagnostic in section 20 concrete, and dropped from the ML feature set
where it would merely duplicate the decision variable.
"""

from __future__ import annotations

import pandas as pd

import config
from src.data_dictionary import ColumnResolver

# Columns that must never enter a predictive feature set, with the reason shown to the user.
BLOCKED = {
    "Late_delivery_risk": "Target variable",
    "Delivery Status": "Reveals the delivery outcome directly",
    "Days for shipping (real)": "Realised shipping duration, known only after delivery",
    "Shipping date (DateOrders)": "Recorded after the shipping decision is made",
}

# Known at decision time but deliberately excluded, with the reason.
EXCLUDED_BY_DESIGN = {
    "Days for shipment (scheduled)": (
        "Deterministic function of Shipping Mode in this dataset; perfectly collinear. "
        "Retained for the statistical model to demonstrate multicollinearity."
    ),
    "Customer Email": "Constant placeholder value, no signal",
    "Customer Password": "Constant placeholder value, no signal; sensitive field",
    "Product Description": "100% missing",
    "Product Status": "Constant value, no signal",
    "Customer Fname": "Personal data, not a legitimate predictor",
    "Customer Lname": "Personal data, not a legitimate predictor",
    "Customer Street": "Personal data, not a legitimate predictor",
    "Customer Zipcode": "Near-identifier for a household; superseded by city/state",
    "Order Zipcode": "86% missing, and undocumented in the data dictionary",
    "Product Image": "URL, no predictive role",
    "Order Id": "Identifier",
    "Order Item Id": "Identifier",
    "Customer Id": "Identifier",
    "Order Customer Id": "Identifier (duplicate of Customer Id)",
    "Product Card Id": "Identifier",
    "Order Item Cardprod Id": "Identifier (duplicate of Product Card Id)",
    "Category Id": "Identifier; Category Name carries the same information readably",
    "Product Category Id": "Identifier (duplicate of Category Id)",
    "Department Id": "Identifier; Department Name carries the same information readably",
    "order date (DateOrders)": "Raw timestamp; used via engineered calendar features",
    "Order Status": (
        "Order lifecycle state at extraction time, not at shipping-decision time. "
        "Terminal values (COMPLETE, CLOSED) are post-fulfilment, and CANCELED / "
        "SUSPECTED_FRAUD reconstruct 'Shipping canceled' exactly. Carries no signal in "
        "any case: every status sits near the 57% base late rate."
    ),
    "Benefit per order": "Exact duplicate of Order Profit Per Order",
    "Sales per customer": "Exact duplicate of Order Item Total",
    "Order Item Product Price": "Exact duplicate of Product Price",
    "Order Profit Per Order": "Economic outcome, not known before fulfilment",
    "Order Item Profit Ratio": "Economic outcome, not known before fulfilment",
    "Product Name": "Near-duplicate of Product Card Id; category level is more stable",
}

# Engineered features and their provenance, so the availability table stays complete.
ENGINEERED_REASON = {
    "order_year": "Derived from order date - known at decision time",
    "order_month": "Derived from order date - known at decision time",
    "order_dayofweek": "Derived from order date - known at decision time",
    "order_hour": "Derived from order date - known at decision time",
    "order_quarter": "Derived from order date - known at decision time",
    "order_is_weekend": "Derived from order date - known at decision time",
    "month_sin": "Cyclical encoding of order month",
    "month_cos": "Cyclical encoding of order month",
    "dow_sin": "Cyclical encoding of order day-of-week",
    "dow_cos": "Cyclical encoding of order day-of-week",
    "daily_order_volume": "Orders placed on the same day - network congestion proxy",
    "discount_depth": "Discount as a share of sales - known at order time",
    "unit_value": "Sales per unit - known at order time",
    "log_product_price": "Log product price - known at order time",
    "log_order_value": "Log order value - known at order time",
    "distance_km": "Great-circle distance from network origin to customer geocode",
}


class LeakageError(ValueError):
    """Raised when a blocked column reaches a model feature set."""


def _is_blocked(resolver: ColumnResolver, column: str) -> str | None:
    """Return the block reason for a column, or None if it is permitted."""
    for name, reason in BLOCKED.items():
        if resolver.resolve(name) == column:
            return reason
    return None


def select_model_features(df: pd.DataFrame, include_decision: bool = True) -> list[str]:
    """
    Return the screened predictor list for the ML models.

    include_decision=False drops Shipping Mode as well. That variant exists to test the
    leakage screen: with the decision variable removed, the remaining operational and
    external factors should carry almost no signal in this dataset, so a materially
    above-chance AUC would indicate an unintended leak.
    """
    r = ColumnResolver(df.columns)
    blocked = set(r.resolve_many(list(BLOCKED)))
    excluded = set(r.resolve_many(list(EXCLUDED_BY_DESIGN)))
    decision = r.resolve(config.DECISION_VARIABLE)

    features = []
    for c in df.columns:
        if c in blocked or c in excluded:
            continue
        if c == decision and not include_decision:
            continue
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) <= 1:
            continue
        features.append(c)
    return features


def assert_no_leakage(df: pd.DataFrame, features: list[str]) -> None:
    """Raise LeakageError if any blocked column appears in the feature list."""
    r = ColumnResolver(df.columns)
    offenders = [
        (f, reason) for f in features if (reason := _is_blocked(r, f)) is not None
    ]
    if offenders:
        detail = "; ".join(f"{f} ({why})" for f, why in offenders)
        raise LeakageError(f"Outcome-derived column(s) in the feature set: {detail}")


def availability_table(df: pd.DataFrame, features: list[str]) -> list[dict]:
    """
    Build the section-3 'Predictor Availability & Data Leakage' table.

    Every column in the raw frame gets a row, so a reviewer can see not just what was used
    but what was considered and rejected, and why.
    """
    r = ColumnResolver(df.columns)
    decision = r.resolve(config.DECISION_VARIABLE)
    target = r.resolve(config.TARGET)
    used = set(features)

    rows = []
    for col in df.columns:
        blocked_reason = _is_blocked(r, col)
        excluded_reason = next(
            (reason for name, reason in EXCLUDED_BY_DESIGN.items()
             if r.resolve(name) == col),
            None,
        )

        if col == target:
            available, used_label, reason = "No", "No", "Target variable"
        elif col == decision:
            available, used_label = "Yes - decision variable", "Special handling"
            reason = ("Management decision; used as a predictor and as the lever in the "
                      "decision engine")
        elif blocked_reason:
            available, used_label, reason = "No", "No", blocked_reason
        elif excluded_reason:
            available, used_label, reason = "Yes", "No", excluded_reason
        elif col in used:
            available, used_label = "Yes", "Yes"
            reason = ENGINEERED_REASON.get(col, "Order, product or destination context")
        else:
            available, used_label, reason = "Yes", "No", "Not selected for the model"

        rows.append(
            {
                "variable": col,
                "available": available,
                "used": used_label,
                "reason": reason,
            }
        )

    order = {"No": 0, "Yes - decision variable": 1, "Yes": 2}
    rows.sort(key=lambda d: (order.get(d["available"], 3), d["variable"]))
    return rows
