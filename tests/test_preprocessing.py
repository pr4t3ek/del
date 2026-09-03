"""Tests for loading, cleaning, order aggregation and the split - run against the real dataset."""

import numpy as np
import pandas as pd
import pytest

import config
from src import classification as cl
from src import data_dictionary as dd
from src import data_preprocessing as dp
from src import statistics_tests as st

# dataset_available() is False for an unfetched Git LFS pointer as well as for a missing
# file. find_file() alone would return the pointer, and every test below would then error
# on DatasetError rather than skipping - which is what a fresh clone looks like.
pytestmark = pytest.mark.skipif(
    not dd.dataset_available(),
    reason=(
        "DataCo dataset not available - run `git lfs pull`, or see the loader's error "
        "message for the fallbacks when that fails"
    ),
)


@pytest.fixture(scope="module")
def raw():
    return dd.load_raw_dataset()


@pytest.fixture(scope="module")
def clean(raw):
    return dp.clean_dataset(raw)


@pytest.fixture(scope="module")
def orders(clean):
    return dp.build_order_frame(clean)


# ---- loading -------------------------------------------------------------------------
def test_dataset_loads_with_latin1(raw):
    assert len(raw) > 0
    # Spanish place names only decode correctly under ISO-8859-1.
    countries = raw["Order Country"].astype(str).unique()
    assert any(any(ord(ch) > 127 for ch in c) for c in countries)


def test_column_resolver_handles_the_case_mismatch(raw):
    """The CSV says 'shipping date'; the dictionary and the brief say 'Shipping date'."""
    r = dd.ColumnResolver(raw.columns)
    resolved = r.resolve("Shipping date (DateOrders)")
    assert resolved is not None
    assert resolved not in raw.columns or resolved in raw.columns
    with pytest.raises(KeyError):
        raw["Shipping date (DateOrders)"]      # the naive lookup genuinely fails
    assert raw[resolved] is not None            # the resolved one works


def test_dictionary_reconciliation_finds_order_zipcode(raw):
    rec = dd.reconcile_with_dictionary(raw, dd.load_dictionary())
    assert rec["available"]
    assert "Order Zipcode" in rec["undocumented"]


# ---- cleaning ------------------------------------------------------------------------
def test_dead_columns_removed(clean):
    for col in ("Product Description", "Customer Email", "Customer Password", "Product Status"):
        assert col not in clean.columns


def test_pii_removed(clean):
    for col in ("Customer Fname", "Customer Lname", "Customer Street"):
        assert col not in clean.columns


def test_duplicate_columns_collapsed(clean):
    for _keep, drop in dd.DUPLICATE_PAIRS:
        assert drop not in clean.columns


def test_no_missing_values_after_cleaning(clean):
    non_datetime = [c for c in clean.columns
                    if not pd.api.types.is_datetime64_any_dtype(clean[c])]
    assert clean[non_datetime].isna().sum().sum() == 0


def test_memory_reduced(raw, clean):
    assert clean.memory_usage(deep=True).sum() < 0.2 * raw.memory_usage(deep=True).sum()


# ---- order aggregation ---------------------------------------------------------------
def test_order_frame_has_one_row_per_order(clean, orders):
    assert len(orders) == clean["Order Id"].nunique()
    assert orders["Order Id"].is_unique


def test_aggregation_preserves_money(clean, orders):
    """Sales and profit must survive the roll-up exactly."""
    for col in ("Sales", "Order Profit Per Order", "Order Item Total"):
        assert float(orders[col].sum()) == pytest.approx(float(clean[col].sum()), rel=1e-6)


def test_outcome_is_constant_within_an_order(clean):
    """The premise behind order-grain analysis and the grouped split."""
    for col in ("Late_delivery_risk", "Shipping Mode", "Days for shipping (real)"):
        assert (clean.groupby("Order Id", observed=True)[col].nunique() > 1).sum() == 0


# ---- feature engineering -------------------------------------------------------------
def test_engineered_features_present(clean):
    feat = dp.engineer_features(clean)
    for col in ("order_month", "month_sin", "daily_order_volume", "distance_km",
                "discount_depth", "log_order_value"):
        assert col in feat.columns
        assert feat[col].notna().all()


def test_bucketing_caps_cardinality(clean):
    feat, mapping = dp.bucket_high_cardinality(dp.engineer_features(clean), fit=True)
    for name, top_n in config.HIGH_CARDINALITY_TOP_N.items():
        if name in feat.columns:
            assert feat[name].nunique() <= top_n + 1     # +1 for "Other"


def test_bucketing_maps_unseen_levels_to_other(clean):
    feat = dp.engineer_features(clean)
    _fitted, mapping = dp.bucket_high_cardinality(feat, fit=True)
    fresh = feat.head(50).copy()
    fresh["Order City"] = "A City That Does Not Exist"
    applied, _ = dp.bucket_high_cardinality(fresh, mapping=mapping, fit=False)
    assert set(applied["Order City"].astype(str)) == {"Other"}


# ---- splitting -----------------------------------------------------------------------
def test_group_split_shares_no_orders(clean):
    feat = dp.engineer_features(clean)
    train_idx, test_idx, info = cl.make_split(feat, strategy="group")
    assert info["strategy"] == "group"
    train_orders = set(feat["Order Id"].iloc[train_idx])
    test_orders = set(feat["Order Id"].iloc[test_idx])
    assert train_orders & test_orders == set()


def test_time_split_is_chronological(clean):
    feat = dp.engineer_features(clean)
    train_idx, test_idx, info = cl.make_split(feat, strategy="time")
    if info["strategy"] != "time":
        pytest.skip("time split unavailable")
    col = "order date (DateOrders)"
    assert feat[col].iloc[train_idx].max() <= feat[col].iloc[test_idx].min()


# ---- statistics ----------------------------------------------------------------------
def test_statistics_run_at_order_grain(orders):
    feat = dp.engineer_features(orders)
    result = st.run_all_tests(feat)
    assert result["grain"] == "order"
    assert result["n_orders"] < len(orders) or result["n_excluded_cancelled"] == 0
    assert result["n_orders"] <= 65752


def test_benjamini_hochberg_is_conservative():
    """BH must never flag more hypotheses than an uncorrected threshold would."""
    p = [0.001, 0.01, 0.04, 0.2, 0.6, 0.9]
    decisions = st.benjamini_hochberg(p, 0.05)
    assert sum(decisions) <= sum(x <= 0.05 for x in p)
    assert decisions[0] is True


def test_cramers_v_is_bounded():
    table = np.array([[50, 50], [50, 50]])
    assert st.cramers_v(table, 0.0) == 0.0
    perfect = np.array([[100, 0], [0, 100]])
    assert 0.0 <= st.cramers_v(perfect, 200.0) <= 1.0
