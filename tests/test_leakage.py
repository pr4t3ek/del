"""Tests for the leakage screen - the guard that separates a forecast from self-scoring."""

import numpy as np
import pandas as pd
import pytest

import config
from src import leakage
from src.data_dictionary import ColumnResolver


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame({
        "Late_delivery_risk": rng.integers(0, 2, n),
        "Delivery Status": rng.choice(["Late delivery", "Shipping on time"], n),
        "Days for shipping (real)": rng.integers(0, 7, n),
        "shipping date (DateOrders)": pd.date_range("2017-01-01", periods=n, freq="h"),
        "Days for shipment (scheduled)": rng.integers(0, 5, n),
        "Shipping Mode": rng.choice(["Standard Class", "First Class"], n),
        "Market": rng.choice(["Europe", "LATAM"], n),
        "Order Item Quantity": rng.integers(1, 5, n),
        "Customer Fname": ["name"] * n,
        "Order Id": rng.integers(1, 50, n),
        "Order Status": rng.choice(["COMPLETE", "PENDING"], n),
    })


def test_blocked_columns_are_never_selected(frame):
    features = leakage.select_model_features(frame)
    for name in leakage.BLOCKED:
        actual = ColumnResolver(frame.columns).resolve(name)
        if actual is not None:
            assert actual not in features, f"{actual} leaked into the feature set"


def test_guard_raises_on_each_blocked_column(frame):
    features = leakage.select_model_features(frame)
    resolver = ColumnResolver(frame.columns)
    for name in leakage.BLOCKED:
        actual = resolver.resolve(name)
        if actual is None:
            continue
        with pytest.raises(leakage.LeakageError):
            leakage.assert_no_leakage(frame, features + [actual])


def test_clean_feature_set_passes_the_guard(frame):
    features = leakage.select_model_features(frame)
    leakage.assert_no_leakage(frame, features)


def test_scheduled_days_excluded_as_alias_of_shipping_mode(frame):
    """Perfectly collinear with the decision variable, so it must not enter the ML model."""
    features = leakage.select_model_features(frame)
    assert "Days for shipment (scheduled)" not in features
    assert config.DECISION_VARIABLE in features


def test_order_status_excluded_as_post_decision(frame):
    assert "Order Status" not in leakage.select_model_features(frame)


def test_pii_excluded(frame):
    assert "Customer Fname" not in leakage.select_model_features(frame)


def test_include_decision_false_drops_the_lever(frame):
    features = leakage.select_model_features(frame, include_decision=False)
    assert config.DECISION_VARIABLE not in features


def test_availability_table_covers_every_column(frame):
    features = leakage.select_model_features(frame)
    table = leakage.availability_table(frame, features)
    assert {r["variable"] for r in table} == set(frame.columns)


def test_target_row_marked_unavailable(frame):
    features = leakage.select_model_features(frame)
    table = leakage.availability_table(frame, features)
    target_row = next(r for r in table if r["variable"] == config.TARGET)
    assert target_row["available"] == "No"
    assert target_row["used"] == "No"
