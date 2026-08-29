"""
Cleaning, order-level aggregation and feature engineering.

Covers spec sections 8 (dataset health), 42 (robustness) and the feature-engineering
stage of section 38's pipeline.

Two structural facts about this dataset drive the design:

* One row is one ORDER LINE ITEM, but the delivery outcome is recorded per ORDER. All
  65,752 orders carry exactly one value of Late_delivery_risk, Shipping Mode, and
  shipping date across their line items - the 180,519 rows are repeated measures with an
  average of 2.75 lines per order. build_order_frame() collapses to the order grain for
  statistics and KPIs; the line-item frame is kept for modelling, where the split is made
  group-aware on Order Id so no order straddles train and test.

* Memory. The raw frame is ~349 MB with deep string storage. Dropping dead, duplicated
  and PII columns and converting to categoricals brings it to ~25 MB, which is what makes
  it practical for Flask to hold the data in memory across requests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src.data_dictionary import (
    DUPLICATE_PAIRS,
    ColumnResolver,
    classify_columns,
)

EARTH_RADIUS_KM = 6371.0

# Columns dropped outright: they carry no signal or are direct personal data.
DROP_ALWAYS = [
    "Product Description",   # 100% null
    "Customer Email",        # constant 'XXXXXXXXX'
    "Customer Password",     # constant 'XXXXXXXXX'
    "Product Status",        # constant 0
    "Customer Fname",        # PII
    "Customer Lname",        # PII
    "Customer Street",       # PII
    "Product Image",         # URL, no predictive role
    "Order Zipcode",         # 86.2% null
]

DATE_COLUMNS = ["order date (DateOrders)", "Shipping date (DateOrders)"]


# --------------------------------------------------------------------------------------
# Dataset health (spec section 8)
# --------------------------------------------------------------------------------------
def dataset_health(df: pd.DataFrame) -> dict:
    """Profile the RAW frame before cleaning, for the Data Overview page."""
    r = ColumnResolver(df.columns)
    n_rows = len(df)

    per_column = []
    for c in df.columns:
        s = df[c]
        n_null = int(s.isna().sum())
        nunique = int(s.nunique(dropna=True))
        example = ""
        non_null = s.dropna()
        if len(non_null):
            example = str(non_null.iloc[0])[:60]
        per_column.append(
            {
                "column": c,
                "dtype": str(s.dtype),
                "missing": n_null,
                "missing_pct": round(100 * n_null / n_rows, 2) if n_rows else 0.0,
                "unique": nunique,
                "example": example,
            }
        )

    # Outliers by the 1.5*IQR rule on genuine measures (not identifiers).
    outliers = []
    skip = {"Latitude", "Longitude"}
    for c in df.select_dtypes(include="number").columns:
        if c in skip or "Id" in c or c == config.TARGET:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        if n_out:
            outliers.append(
                {"column": c, "count": n_out, "pct": round(100 * n_out / n_rows, 2)}
            )
    outliers.sort(key=lambda d: -d["count"])

    duplicated_pairs = []
    for keep, drop in DUPLICATE_PAIRS:
        a, b = r.resolve(keep), r.resolve(drop)
        if a and b:
            identical = bool((df[a].fillna(-9e12) == df[b].fillna(-9e12)).all())
            if identical:
                duplicated_pairs.append({"kept": a, "dropped": b})

    n_dup_rows = int(df.duplicated().sum())
    cls = classify_columns(df)

    return {
        "n_rows": n_rows,
        "n_columns": int(df.shape[1]),
        "n_numeric": int(df.select_dtypes(include="number").shape[1]),
        "n_categorical": int(df.shape[1] - df.select_dtypes(include="number").shape[1]),
        "n_missing_cells": int(df.isna().sum().sum()),
        "missing_pct": round(100 * df.isna().sum().sum() / (n_rows * df.shape[1]), 3),
        "n_duplicate_rows": n_dup_rows,
        "duplicate_pct": round(100 * n_dup_rows / n_rows, 3) if n_rows else 0.0,
        "columns": per_column,
        "outliers": outliers[:15],
        "duplicated_pairs": duplicated_pairs,
        "dead_columns": cls["dead"],
        "pii_columns": cls["pii"],
        "leakage_columns": cls["leakage"],
        "identifier_columns": cls["identifiers"],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1),
    }


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------
def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the analysis frame: drop dead/PII/duplicate columns, parse dates, impute the
    few stray nulls, and compact dtypes.
    """
    r = ColumnResolver(df.columns)
    out = df.copy()

    to_drop = r.resolve_many(DROP_ALWAYS)
    for _keep, drop in DUPLICATE_PAIRS:
        actual = r.resolve(drop)
        if actual and actual not in to_drop:
            to_drop.append(actual)
    out = out.drop(columns=[c for c in to_drop if c in out.columns])

    # Dates. Format is m/d/Y H:M; fall back to inference if a variant appears.
    r2 = ColumnResolver(out.columns)
    for name in DATE_COLUMNS:
        col = r2.resolve(name)
        if col is None:
            continue
        parsed = pd.to_datetime(out[col], format="%m/%d/%Y %H:%M", errors="coerce")
        if parsed.isna().all():
            parsed = pd.to_datetime(out[col], errors="coerce")
        out[col] = parsed

    # Residual missing values: median for numerics, explicit "Unknown" for categoricals.
    for c in out.columns:
        if out[c].isna().any():
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c].fillna(out[c].median())
            elif pd.api.types.is_datetime64_any_dtype(out[c]):
                continue
            else:
                out[c] = out[c].astype("object").fillna("Unknown")

    # Compact dtypes: 349 MB -> ~25 MB.
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            continue
        if pd.api.types.is_numeric_dtype(out[c]):
            kind = "integer" if pd.api.types.is_integer_dtype(out[c]) else "float"
            out[c] = pd.to_numeric(out[c], downcast=kind)
        else:
            out[c] = out[c].astype("category")

    return out


# --------------------------------------------------------------------------------------
# Order-grain aggregation
# --------------------------------------------------------------------------------------
def build_order_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse line items to one row per Order Id.

    Fields that are invariant within an order (mode, market, dates, outcome) are carried
    through unchanged; money and quantity are summed; basket descriptors are derived.
    Used for statistical testing and KPIs so that a 3-line order does not count three
    times toward a single delivery outcome.
    """
    r = ColumnResolver(df.columns)
    order_id = r.resolve("Order Id")
    if order_id is None:
        return df.copy()

    sum_cols = r.resolve_many(
        ["Sales", "Order Item Total", "Order Profit Per Order",
         "Order Item Discount", "Order Item Quantity"]
    )
    mean_cols = r.resolve_many(
        ["Order Item Discount Rate", "Order Item Profit Ratio", "Product Price"]
    )
    invariant = r.resolve_many(
        ["Shipping Mode", "Days for shipment (scheduled)", "Days for shipping (real)",
         "Delivery Status", "Late_delivery_risk", "Type", "Order Status",
         "Market", "Order Region", "Order Country", "Order City", "Order State",
         "Customer Segment", "Customer City", "Customer State", "Customer Country",
         "Customer Id", "Latitude", "Longitude",
         "order date (DateOrders)", "Shipping date (DateOrders)"]
    )

    spec = {c: (c, "sum") for c in sum_cols}
    spec.update({c: (c, "mean") for c in mean_cols})
    spec.update({c: (c, "first") for c in invariant})

    grouped = df.groupby(order_id, observed=True, sort=False)
    orders = grouped.agg(**spec)

    orders["n_items"] = grouped.size()
    product = r.resolve("Product Card Id")
    if product:
        orders["n_distinct_products"] = grouped[product].nunique()

    # Dominant category/department = the one on the highest-value line of the order.
    sales = r.resolve("Sales")
    for name, out_name in [("Category Name", "Category Name"),
                           ("Department Name", "Department Name")]:
        col = r.resolve(name)
        if col is None or sales is None:
            continue
        idx = df.groupby(order_id, observed=True, sort=False)[sales].idxmax()
        orders[out_name] = df.loc[idx, col].to_numpy()

    orders = orders.reset_index()

    for c in orders.columns:
        if orders[c].dtype == object:
            orders[c] = orders[c].astype("category")

    return orders


# --------------------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------------------
def _haversine(lat, lon, lat0, lon0):
    lat, lon = np.radians(lat), np.radians(lon)
    lat0, lon0 = np.radians(lat0), np.radians(lon0)
    dlat, dlon = lat - lat0, lon - lon0
    a = np.sin(dlat / 2) ** 2 + np.cos(lat0) * np.cos(lat) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add decision-time features.

    Everything here is computable at the moment the shipping-mode decision is made: order
    contents, calendar position of the order date, geography, and how busy the network was
    on the order date. Nothing uses the shipping date or the realised transit time.
    """
    r = ColumnResolver(df.columns)
    out = df.copy()

    order_date = r.resolve("order date (DateOrders)")
    if order_date and pd.api.types.is_datetime64_any_dtype(out[order_date]):
        od = out[order_date]
        out["order_year"] = od.dt.year.astype("int16")
        out["order_month"] = od.dt.month.astype("int8")
        out["order_dayofweek"] = od.dt.dayofweek.astype("int8")
        out["order_hour"] = od.dt.hour.astype("int8")
        out["order_quarter"] = od.dt.quarter.astype("int8")
        out["order_is_weekend"] = (od.dt.dayofweek >= 5).astype("int8")
        # Cyclical encodings so December is adjacent to January.
        out["month_sin"] = np.sin(2 * np.pi * od.dt.month / 12).astype("float32")
        out["month_cos"] = np.cos(2 * np.pi * od.dt.month / 12).astype("float32")
        out["dow_sin"] = np.sin(2 * np.pi * od.dt.dayofweek / 7).astype("float32")
        out["dow_cos"] = np.cos(2 * np.pi * od.dt.dayofweek / 7).astype("float32")
        # Network load: how many orders were placed on the same calendar day. A congestion
        # proxy that stands in for the external demand pressure the dataset does not carry.
        day = od.dt.normalize()
        out["daily_order_volume"] = day.map(day.value_counts()).astype("float32")

    # Order economics available before shipping.
    qty = r.resolve("Order Item Quantity")
    sales = r.resolve("Sales")
    total = r.resolve("Order Item Total")
    discount = r.resolve("Order Item Discount")
    price = r.resolve("Product Price")

    if sales and discount:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["discount_depth"] = (
                out[discount] / out[sales].replace(0, np.nan)
            ).fillna(0).astype("float32")
    if sales and qty:
        with np.errstate(divide="ignore", invalid="ignore"):
            out["unit_value"] = (
                out[sales] / out[qty].replace(0, np.nan)
            ).fillna(0).astype("float32")
    if price:
        out["log_product_price"] = np.log1p(out[price].clip(lower=0)).astype("float32")
    if total:
        out["log_order_value"] = np.log1p(out[total].clip(lower=0)).astype("float32")

    # Distance from the network's origin centroid to the customer geocode.
    lat, lon = r.resolve("Latitude"), r.resolve("Longitude")
    if lat and lon:
        lat0 = float(out[lat].median())
        lon0 = float(out[lon].median())
        out["distance_km"] = _haversine(
            out[lat].to_numpy(dtype="float64"),
            out[lon].to_numpy(dtype="float64"),
            lat0, lon0,
        ).astype("float32")

    return out


def bucket_high_cardinality(
    df: pd.DataFrame, mapping: dict | None = None, fit: bool = True
) -> tuple[pd.DataFrame, dict]:
    """
    Collapse high-cardinality categoricals to their top-N levels plus 'Other'.

    Order City alone has 3,597 levels; one-hot encoding it raw would dominate the design
    matrix. The retained level sets are learned on the training data only and reused at
    prediction time so unseen levels fall into 'Other' rather than creating new columns.
    """
    r = ColumnResolver(df.columns)
    out = df.copy()
    mapping = {} if fit else dict(mapping or {})

    for name, top_n in config.HIGH_CARDINALITY_TOP_N.items():
        col = r.resolve(name)
        if col is None:
            continue
        if fit:
            keep = out[col].value_counts().head(top_n).index.tolist()
            mapping[col] = [str(k) for k in keep]
        keep = set(mapping.get(col, []))
        values = out[col].astype(str)
        out[col] = np.where(values.isin(keep), values, "Other")
        out[col] = out[col].astype("category")

    return out, mapping
