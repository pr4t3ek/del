"""Tests for the expected-cost model and the decision engine."""

import numpy as np
import pandas as pd
import pytest

import config
from src import decision_analysis as da


@pytest.fixture
def mode_profile():
    return {
        "Standard Class": {"mode": "Standard Class", "orders": 100, "late_rate": 0.40,
                           "promised_days": 4, "mean_transit": 4.0, "median_transit": 4.0,
                           "std_transit": 1.4},
        "Second Class":   {"mode": "Second Class", "orders": 100, "late_rate": 0.80,
                           "promised_days": 2, "mean_transit": 4.0, "median_transit": 4.0,
                           "std_transit": 1.4},
        "First Class":    {"mode": "First Class", "orders": 100, "late_rate": 1.00,
                           "promised_days": 1, "mean_transit": 2.0, "median_transit": 2.0,
                           "std_transit": 0.0},
        "Same Day":       {"mode": "Same Day", "orders": 100, "late_rate": 0.48,
                           "promised_days": 0, "mean_transit": 0.5, "median_transit": 0.0,
                           "std_transit": 0.5},
    }


@pytest.fixture
def assumptions():
    return da.resolve_assumptions()


def test_expected_cost_matches_hand_calculation(mode_profile):
    """freight + P(late)*penalty + holding*value*transit + speed*transit."""
    a = da.resolve_assumptions({
        "late_penalty_fixed": 10.0, "late_penalty_rate": 0.10,
        "holding_rate_per_day": 0.001, "speed_value_per_day": 5.0,
    })
    a["freight_cost"]["Standard Class"] = 6.0

    result = da.expected_cost(0.5, 100.0, "Standard Class", a, mode_profile)

    expected = 6.0 + 0.5 * (10.0 + 0.10 * 100.0) + 0.001 * 100.0 * 4.0 + 5.0 * 4.0
    assert float(result["expected_total_cost"]) == pytest.approx(expected)
    assert float(result["expected_delay_cost"]) == pytest.approx(10.0)
    assert float(result["holding_cost"]) == pytest.approx(0.4)
    assert result["speed_cost"] == pytest.approx(20.0)


def test_zero_probability_removes_the_delay_term(mode_profile, assumptions):
    r = da.expected_cost(0.0, 200.0, "Standard Class", assumptions, mode_profile)
    assert float(r["expected_delay_cost"]) == 0.0


def test_cost_increases_with_late_probability(mode_profile, assumptions):
    low = da.expected_cost(0.1, 100.0, "Standard Class", assumptions, mode_profile)
    high = da.expected_cost(0.9, 100.0, "Standard Class", assumptions, mode_profile)
    assert float(high["expected_total_cost"]) > float(low["expected_total_cost"])


def test_penalty_scales_with_order_value(mode_profile, assumptions):
    cheap = da.expected_cost(0.5, 50.0, "Standard Class", assumptions, mode_profile)
    rich = da.expected_cost(0.5, 5000.0, "Standard Class", assumptions, mode_profile)
    assert float(rich["expected_delay_cost"]) > float(cheap["expected_delay_cost"])


def test_compare_modes_picks_the_cheapest(mode_profile, assumptions):
    p = {m: prof["late_rate"] for m, prof in mode_profile.items()}
    result = da.compare_modes(p, 150.0, assumptions, mode_profile)
    costs = [r["expected_total_cost"] for r in result["rows"]]
    assert costs == sorted(costs)
    assert result["recommended"] == result["rows"][0]["mode"]


def test_constraint_excludes_high_risk_modes(mode_profile, assumptions):
    p = {m: prof["late_rate"] for m, prof in mode_profile.items()}
    result = da.compare_modes(
        p, 150.0, assumptions, mode_profile, constraints={"max_late_probability": 0.5}
    )
    chosen = next(r for r in result["rows"] if r["mode"] == result["recommended"])
    assert chosen["p_late"] <= 0.5
    assert any(not r["feasible"] for r in result["rows"])


def test_degeneracy_without_time_terms(mode_profile):
    """
    The documented reason the cost model carries time terms.

    With holding and speed set to zero, Standard Class has both the lowest freight cost and
    the lowest late rate, so it should dominate nearly every order.
    """
    rng = np.random.default_rng(0)
    n = 400
    orders = pd.DataFrame({"Order Item Total": rng.uniform(20, 400, n)})
    p_by_mode = {m: np.full(n, prof["late_rate"]) for m, prof in mode_profile.items()}

    result = da.degeneracy_check(orders, p_by_mode, "Order Item Total", mode_profile)
    assert result["is_degenerate"]
    assert result["dominant_mode"] == "Standard Class"
    assert result["dominant_share"] >= 0.99


def test_time_terms_restore_a_real_tradeoff(mode_profile):
    """With a high enough value of speed, a faster mode must be able to win."""
    a = da.resolve_assumptions({"speed_value_per_day": 40.0})
    p = {m: prof["late_rate"] for m, prof in mode_profile.items()}
    result = da.compare_modes(p, 60.0, a, mode_profile)
    assert result["recommended"] == "Same Day"


def test_promise_redesign_detects_identical_transit(mode_profile):
    result = da.promise_redesign_analysis(mode_profile)
    assert result["available"]
    assert result["transit_identical"]
    assert result["late_rate_reduction"] == pytest.approx(0.40)


def test_presets_change_the_assumptions():
    low, high = da.apply_preset("low"), da.apply_preset("high")
    assert high["late_penalty_fixed"] > low["late_penalty_fixed"]
    assert high["speed_value_per_day"] > low["speed_value_per_day"]


def test_resolve_assumptions_ignores_garbage():
    a = da.resolve_assumptions({"late_penalty_fixed": "not a number"})
    assert a["late_penalty_fixed"] == config.SCENARIO_DEFAULTS["late_penalty_fixed"]


def test_break_even_returns_the_full_ladder(mode_profile, assumptions):
    ladder = da.break_even_analysis(assumptions, mode_profile, 150.0)
    assert [(b["from_mode"], b["to_mode"]) for b in ladder] == [
        ("Standard Class", "Second Class"),
        ("Second Class", "First Class"),
        ("First Class", "Same Day"),
    ]
