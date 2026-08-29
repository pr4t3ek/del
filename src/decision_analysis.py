"""
Decision optimization under uncertainty (spec sections 22-30).

The question: given an order's predicted probability of late delivery, which shipping mode
minimises expected total supply-chain cost while protecting profitability?

--------------------------------------------------------------------------------------
THE COST MODEL IS BUILT ON ASSUMPTIONS, NOT OBSERVED DATA
--------------------------------------------------------------------------------------
The DataCo dataset contains no freight cost, carrier, or logistics-cost field. Total
logistics cost therefore cannot be measured from it. Every monetary parameter below is a
scenario assumption supplied by the analyst and changeable from the UI. The application
labels them as such everywhere they appear.

--------------------------------------------------------------------------------------
WHY THERE IS A TIME TERM IN THE COST FUNCTION
--------------------------------------------------------------------------------------
Section 25 frames expected total cost as:

    shipping mode cost  +  P(late) x cost of late delivery

Applied literally to this dataset that formula is degenerate. Standard Class has both the
lowest assumed freight cost AND the lowest observed late rate (38.1%, against 76.7% for
Second Class and 95.3% for First Class), so it dominates every alternative for every
order under every assumption, and the "optimisation" has a single answer before it starts.

The reason is structural rather than operational. Realised transit time in this dataset is
a function of shipping mode alone, and the modes differ mainly in what they *promise*:
Second Class and Standard Class have statistically identical transit distributions
(mean 3.99 days each) but promise 2 days and 4 days respectively. A mode is therefore
"late" largely because of the commitment attached to it.

What the binary late/on-time framing misses is that faster modes deliver sooner even when
they miss their promise - Same Day averages 0.5 days in transit against Standard Class's
4.0. The model adds two time-related terms to capture that value:

    holding cost      - inventory/pipeline capital tied up while goods are in transit
    value of speed    - the business value of receiving goods sooner (customer experience,
                        competitive service)

With these included the trade-off becomes real and order-dependent: low-value orders
favour speed, high-value orders favour the lower late risk of Standard Class, and the
crossover point moves with the assumptions. Setting both time parameters to zero restores
the degenerate case, which the decision page shows explicitly rather than hiding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

MODES = ["Standard Class", "Second Class", "First Class", "Same Day"]


# --------------------------------------------------------------------------------------
# Assumption handling
# --------------------------------------------------------------------------------------
def resolve_assumptions(overrides: dict | None = None) -> dict:
    """Merge user overrides onto the configured defaults, coercing types safely."""
    base = {
        "freight_cost": dict(config.SCENARIO_DEFAULTS["freight_cost"]),
        "late_penalty_fixed": config.SCENARIO_DEFAULTS["late_penalty_fixed"],
        "late_penalty_rate": config.SCENARIO_DEFAULTS["late_penalty_rate"],
        "holding_rate_per_day": config.SCENARIO_DEFAULTS["holding_rate_per_day"],
        "speed_value_per_day": config.SCENARIO_DEFAULTS["speed_value_per_day"],
    }
    if not overrides:
        return base

    for key in ("late_penalty_fixed", "late_penalty_rate",
                "holding_rate_per_day", "speed_value_per_day"):
        if overrides.get(key) is not None:
            try:
                base[key] = float(overrides[key])
            except (TypeError, ValueError):
                pass

    freight = overrides.get("freight_cost")
    if isinstance(freight, dict):
        for mode, value in freight.items():
            if mode in base["freight_cost"]:
                try:
                    base["freight_cost"][mode] = float(value)
                except (TypeError, ValueError):
                    pass
    return base


def apply_preset(name: str) -> dict:
    """Return the assumption set for a named scenario preset (spec section 28)."""
    preset = config.SCENARIO_PRESETS.get(name)
    if not preset:
        return resolve_assumptions()
    return resolve_assumptions(
        {k: v for k, v in preset.items() if k not in ("label", "description")}
    )


# --------------------------------------------------------------------------------------
# Mode profile: what each mode actually does, measured from the data
# --------------------------------------------------------------------------------------
def build_mode_profile(orders: pd.DataFrame, target: str, mode_col: str,
                       real_col: str, sched_col: str) -> dict:
    """
    Observed behaviour of each shipping mode: late rate and transit-time distribution.

    These are measured values, unlike the cost parameters. The decision engine uses the
    per-mode transit distribution for the time-related cost terms and the model's predicted
    probability for the delay term.
    """
    profile = {}
    for mode, sub in orders.groupby(mode_col, observed=True):
        transit = pd.to_numeric(sub[real_col], errors="coerce").dropna()
        profile[str(mode)] = {
            "mode": str(mode),
            "orders": int(len(sub)),
            "late_rate": round(float(sub[target].mean()), 4),
            "promised_days": int(pd.to_numeric(sub[sched_col], errors="coerce").median()),
            "mean_transit": round(float(transit.mean()), 4),
            "median_transit": round(float(transit.median()), 4),
            "std_transit": round(float(transit.std()), 4),
        }
    return profile


# --------------------------------------------------------------------------------------
# Expected-cost framework (spec sections 25-26)
# --------------------------------------------------------------------------------------
def expected_cost(
    p_late: float | np.ndarray,
    order_value: float | np.ndarray,
    mode: str,
    assumptions: dict,
    mode_profile: dict,
) -> dict:
    """
    Expected total cost of fulfilling an order under one shipping mode.

        freight               assumed cost of the mode
      + P(late) x penalty     assumed economic loss from a late delivery
      + holding               assumed capital cost of goods in transit
      + speed cost            assumed value forgone per day of transit

    p_late comes from the trained model; mean transit comes from the data; every monetary
    coefficient is an assumption.
    """
    p_late = np.asarray(p_late, dtype=float)
    order_value = np.asarray(order_value, dtype=float)

    freight = float(assumptions["freight_cost"].get(mode, 0.0))
    penalty = (
        float(assumptions["late_penalty_fixed"])
        + float(assumptions["late_penalty_rate"]) * order_value
    )
    transit = float(mode_profile.get(mode, {}).get("mean_transit", 0.0))

    delay_cost = p_late * penalty
    holding = float(assumptions["holding_rate_per_day"]) * order_value * transit
    speed_cost = float(assumptions["speed_value_per_day"]) * transit

    total = freight + delay_cost + holding + speed_cost

    return {
        "mode": mode,
        "freight": freight,
        "p_late": p_late,
        "penalty_if_late": penalty,
        "expected_delay_cost": delay_cost,
        "holding_cost": holding,
        "speed_cost": speed_cost,
        "expected_total_cost": total,
        "mean_transit_days": transit,
    }


def compare_modes(
    p_late_by_mode: dict,
    order_value: float,
    assumptions: dict,
    mode_profile: dict,
    constraints: dict | None = None,
    profit: float | None = None,
) -> dict:
    """
    Evaluate every shipping mode for a single order and recommend one (spec section 26).

    p_late_by_mode maps mode -> model-predicted probability with that mode substituted in.
    Constraints are optional and always labelled as user-set, never hardcoded.
    """
    constraints = constraints or {}
    max_p = constraints.get("max_late_probability")

    rows = []
    for mode in MODES:
        if mode not in mode_profile:
            continue
        p = float(p_late_by_mode.get(mode, np.nan))
        if not np.isfinite(p):
            continue
        cost = expected_cost(p, order_value, mode, assumptions, mode_profile)

        feasible, reason = True, ""
        if max_p is not None and p > float(max_p):
            feasible = False
            reason = f"Predicted late risk {p:.1%} exceeds the {float(max_p):.0%} limit."

        row = {
            "mode": mode,
            "p_late": round(p, 4),
            "freight": round(cost["freight"], 2),
            "expected_delay_cost": round(float(cost["expected_delay_cost"]), 2),
            "holding_cost": round(float(cost["holding_cost"]), 2),
            "speed_cost": round(float(cost["speed_cost"]), 2),
            "expected_total_cost": round(float(cost["expected_total_cost"]), 2),
            "mean_transit_days": cost["mean_transit_days"],
            "penalty_if_late": round(float(cost["penalty_if_late"]), 2),
            "feasible": feasible,
            "infeasible_reason": reason,
        }
        if profit is not None:
            row["expected_net"] = round(
                float(profit) - float(cost["expected_total_cost"]), 2
            )
        rows.append(row)

    rows.sort(key=lambda d: d["expected_total_cost"])
    feasible_rows = [r for r in rows if r["feasible"]]
    chosen = feasible_rows[0] if feasible_rows else (rows[0] if rows else None)

    baseline = next((r for r in rows if r["mode"] == "Standard Class"), None)
    saving = None
    if chosen and baseline:
        saving = round(baseline["expected_total_cost"] - chosen["expected_total_cost"], 2)

    return {
        "rows": rows,
        "recommended": chosen["mode"] if chosen else None,
        "recommended_cost": chosen["expected_total_cost"] if chosen else None,
        "saving_vs_standard": saving,
        "constrained": bool(feasible_rows) and len(feasible_rows) < len(rows),
    }


# --------------------------------------------------------------------------------------
# Break-even analysis (spec section 27)
# --------------------------------------------------------------------------------------
def break_even_analysis(assumptions: dict, mode_profile: dict, order_value: float) -> list[dict]:
    """
    At what late-delivery probability does upgrading become worthwhile?

    Solves for the probability at which two modes have equal expected total cost, holding
    order value fixed. An upgrade pays when the expected saving in delay-related cost plus
    the time-related benefit exceeds the incremental freight cost.
    """
    ladder = [
        ("Standard Class", "Second Class"),
        ("Second Class", "First Class"),
        ("First Class", "Same Day"),
    ]
    penalty = (
        float(assumptions["late_penalty_fixed"])
        + float(assumptions["late_penalty_rate"]) * float(order_value)
    )

    out = []
    for base_mode, upgrade in ladder:
        if base_mode not in mode_profile or upgrade not in mode_profile:
            continue

        base_p = mode_profile[base_mode]["late_rate"]
        up_p = mode_profile[upgrade]["late_rate"]

        base_cost = expected_cost(base_p, order_value, base_mode, assumptions, mode_profile)
        up_cost = expected_cost(up_p, order_value, upgrade, assumptions, mode_profile)

        # Time-independent difference the upgrade must overcome.
        fixed_gap = (
            (up_cost["freight"] + float(up_cost["holding_cost"]) + up_cost["speed_cost"])
            - (base_cost["freight"] + float(base_cost["holding_cost"]) + base_cost["speed_cost"])
        )
        # Cost equalises when: base_p * penalty - up_p * penalty = fixed_gap
        delta_p = fixed_gap / penalty if penalty else np.nan
        # Expressed as the base-mode late probability at which the upgrade breaks even,
        # holding the upgrade's own late probability at its observed level.
        break_even_p = up_p + delta_p if np.isfinite(delta_p) else np.nan

        worthwhile = float(base_cost["expected_total_cost"]) > float(up_cost["expected_total_cost"])

        if not np.isfinite(break_even_p):
            verdict = "Not computable under the current assumptions."
        elif break_even_p > 1:
            verdict = (
                f"Never worthwhile at this order value: even a 100% late rate on "
                f"{base_mode} would not justify the incremental cost of {upgrade}."
            )
        elif break_even_p < 0:
            verdict = (
                f"Always worthwhile at this order value: {upgrade} is cheaper in expected "
                f"total cost regardless of {base_mode}'s late rate."
            )
        else:
            verdict = (
                f"{upgrade} becomes worthwhile once {base_mode}'s late probability exceeds "
                f"{break_even_p:.1%}. Its observed rate is {base_p:.1%}, so the upgrade is "
                f"{'justified' if base_p > break_even_p else 'not justified'} today."
            )

        out.append(
            {
                "from_mode": base_mode,
                "to_mode": upgrade,
                "observed_p_from": round(base_p, 4),
                "observed_p_to": round(up_p, 4),
                "incremental_freight": round(
                    up_cost["freight"] - base_cost["freight"], 2
                ),
                "break_even_p": (
                    round(float(break_even_p), 4) if np.isfinite(break_even_p) else None
                ),
                "cost_from": round(float(base_cost["expected_total_cost"]), 2),
                "cost_to": round(float(up_cost["expected_total_cost"]), 2),
                "worthwhile": bool(worthwhile),
                "verdict": verdict,
            }
        )
    return out


# --------------------------------------------------------------------------------------
# Portfolio-level policy evaluation (spec sections 28-30)
# --------------------------------------------------------------------------------------
def _portfolio_costs(
    values: np.ndarray,
    p_by_mode: dict,
    assumptions: dict,
    mode_profile: dict,
) -> dict:
    """Expected total cost of assigning every order to each single mode."""
    out = {}
    for mode in MODES:
        if mode not in mode_profile:
            continue
        cost = expected_cost(p_by_mode[mode], values, mode, assumptions, mode_profile)
        out[mode] = np.asarray(cost["expected_total_cost"], dtype=float)
    return out


def evaluate_policies(
    orders: pd.DataFrame,
    p_by_mode: dict,
    value_col: str,
    assumptions: dict,
    mode_profile: dict,
    segment_cols: dict | None = None,
    constraints: dict | None = None,
) -> dict:
    """
    Compare the logistics service-network policies in spec section 29.

        A  Single mode for every order
        B  Market-specific policy
        C  Region-specific policy
        D  Risk-based assignment (per-order expected-cost minimisation)
        E  Risk plus market/region segmentation

    Section 29 is explicit that this dataset does not support physical network design -
    there are no warehouse, facility, or carrier fields, and origin geography has only two
    values. What it does support is service-network policy: which mode to use, for whom.
    """
    constraints = constraints or {}
    values = pd.to_numeric(orders[value_col], errors="coerce").fillna(0.0).to_numpy(float)
    n = len(values)
    costs = _portfolio_costs(values, p_by_mode, assumptions, mode_profile)
    available = list(costs.keys())
    cost_matrix = np.column_stack([costs[m] for m in available])
    late_matrix = np.column_stack(
        [np.asarray(p_by_mode[m], dtype=float) for m in available]
    )

    max_p = constraints.get("max_late_probability")
    feasible = np.ones_like(cost_matrix, dtype=bool)
    if max_p is not None:
        feasible = late_matrix <= float(max_p)
        # Never leave an order with no option: fall back to its lowest-risk mode.
        empty = ~feasible.any(axis=1)
        if empty.any():
            feasible[empty, np.argmin(late_matrix[empty], axis=1)] = True

    masked = np.where(feasible, cost_matrix, np.inf)

    def summarise(choice_idx: np.ndarray, name: str, description: str) -> dict:
        chosen_cost = masked[np.arange(n), choice_idx]
        chosen_cost = np.where(np.isfinite(chosen_cost), chosen_cost,
                               cost_matrix[np.arange(n), choice_idx])
        chosen_late = late_matrix[np.arange(n), choice_idx]
        mix = pd.Series([available[i] for i in choice_idx]).value_counts(normalize=True)
        return {
            "policy": name,
            "description": description,
            "total_cost": round(float(chosen_cost.sum()), 2),
            "avg_cost": round(float(chosen_cost.mean()), 3),
            "expected_late_rate": round(float(chosen_late.mean()), 4),
            "mode_mix": {k: round(float(v), 4) for k, v in mix.items()},
            "n_modes_used": int(len(mix)),
        }

    policies = []

    # Policy A - one mode for everything. Report the best single mode.
    single = [
        summarise(
            np.full(n, i),
            f"A - Single mode ({m})",
            f"Every order ships {m}, regardless of risk, value or destination.",
        )
        for i, m in enumerate(available)
    ]
    best_single = min(single, key=lambda d: d["total_cost"])
    policies.append(best_single)
    policy_a_all = single

    # Policies B and C - one mode per market / per region, chosen on that segment's cost.
    segment_cols = segment_cols or {}
    for label, (letter, col) in {
        "market": ("B", segment_cols.get("market")),
        "region": ("C", segment_cols.get("region")),
    }.items():
        if not col or col not in orders.columns:
            continue
        choice = np.zeros(n, dtype=int)
        groups = orders[col].astype(str).to_numpy()
        for level in pd.unique(groups):
            mask = groups == level
            totals = np.where(feasible[mask], cost_matrix[mask], np.inf).sum(axis=0)
            choice[mask] = int(np.argmin(totals))
        policies.append(
            summarise(
                choice,
                f"{letter} - {label.capitalize()}-specific policy",
                f"One shipping mode chosen per {label}, minimising that "
                f"{label}'s total expected cost.",
            )
        )

    # Policy D - per-order risk-based assignment.
    choice_d = np.argmin(masked, axis=1)
    policies.append(
        summarise(
            choice_d,
            "D - Risk-based assignment",
            "Each order is assigned the mode with the lowest expected total cost, given "
            "its predicted late risk and order value.",
        )
    )

    # Policy E - risk-based within market/region segments (same optimum per order, but
    # constrained to one rule per segment x value band, so it is operable as a policy).
    market_col = segment_cols.get("market")
    if market_col and market_col in orders.columns:
        bands = pd.qcut(values, q=4, labels=False, duplicates="drop")
        seg = pd.Series(orders[market_col].astype(str).to_numpy()).astype(str) + "|" + pd.Series(bands).astype(str)
        choice_e = np.zeros(n, dtype=int)
        seg_arr = seg.to_numpy()
        for level in pd.unique(seg_arr):
            mask = seg_arr == level
            totals = np.where(feasible[mask], cost_matrix[mask], np.inf).sum(axis=0)
            choice_e[mask] = int(np.argmin(totals))
        policies.append(
            summarise(
                choice_e,
                "E - Risk + segment policy",
                "One rule per market and order-value quartile: operable as a written "
                "policy, while still adapting to risk and value.",
            )
        )

    baseline = policies[0]["total_cost"]
    for p in policies:
        p["saving_vs_best_single"] = round(baseline - p["total_cost"], 2)
        p["saving_pct"] = (
            round(100 * (baseline - p["total_cost"]) / baseline, 2) if baseline else 0.0
        )
        p["complexity"] = _policy_complexity(p["policy"])
    policies.sort(key=lambda d: d["total_cost"])

    return {
        "policies": policies,
        "single_mode_detail": policy_a_all,
        "n_orders": int(n),
        "scope_note": (
            "These are logistics SERVICE-network policies - which shipping mode is used "
            "for which orders. The dataset carries no warehouse, distribution-centre, "
            "facility or carrier fields, and origin geography has only two values, so it "
            "cannot support physical network design such as facility location."
        ),
    }


def _policy_complexity(name: str) -> str:
    if name.startswith("A"):
        return "Lowest - a single rule."
    if name.startswith(("B", "C")):
        return "Low - one rule per market or region."
    if name.startswith("E"):
        return "Moderate - one rule per segment and value band."
    return "Highest - a per-order scoring decision, needs the model in the fulfilment path."


def sensitivity_analysis(
    orders: pd.DataFrame,
    p_by_mode: dict,
    value_col: str,
    mode_profile: dict,
    parameter: str,
    sweep: list[float],
    base_assumptions: dict | None = None,
) -> dict:
    """
    Sweep one assumption and record how the optimal policy responds (spec section 28).

    The output is a stability region rather than a point estimate: the point of the sweep
    is to show where the recommendation changes, and how far the current assumption sits
    from that boundary.
    """
    base = base_assumptions or resolve_assumptions()
    values = pd.to_numeric(orders[value_col], errors="coerce").fillna(0.0).to_numpy(float)
    n = len(values)

    rows = []
    for level in sweep:
        assumptions = {**base, "freight_cost": dict(base["freight_cost"])}
        if parameter in assumptions and parameter != "freight_cost":
            assumptions[parameter] = float(level)
        elif parameter.startswith("freight:"):
            assumptions["freight_cost"][parameter.split(":", 1)[1]] = float(level)

        costs = _portfolio_costs(values, p_by_mode, assumptions, mode_profile)
        available = list(costs.keys())
        matrix = np.column_stack([costs[m] for m in available])
        choice = np.argmin(matrix, axis=1)
        chosen_cost = matrix[np.arange(n), choice]
        late = np.column_stack([np.asarray(p_by_mode[m], float) for m in available])
        mix = pd.Series([available[i] for i in choice]).value_counts(normalize=True)

        rows.append(
            {
                "value": round(float(level), 5),
                "total_cost": round(float(chosen_cost.sum()), 2),
                "avg_cost": round(float(chosen_cost.mean()), 3),
                "expected_late_rate": round(
                    float(late[np.arange(n), choice].mean()), 4
                ),
                "dominant_mode": str(mix.index[0]),
                "dominant_share": round(float(mix.iloc[0]), 4),
                "mode_mix": {k: round(float(v), 4) for k, v in mix.items()},
            }
        )

    switches = [
        {"from": rows[i - 1]["dominant_mode"], "to": rows[i]["dominant_mode"],
         "at": rows[i]["value"]}
        for i in range(1, len(rows))
        if rows[i]["dominant_mode"] != rows[i - 1]["dominant_mode"]
    ]

    return {
        "parameter": parameter,
        "rows": rows,
        "switch_points": switches,
        "stable": not switches,
        "summary": (
            f"The dominant mode does not change across the swept range of {parameter}; "
            "the recommendation is robust to this assumption."
            if not switches
            else
            f"The recommendation changes {len(switches)} time(s) across the range: "
            + "; ".join(f"{s['from']} to {s['to']} at {s['at']:g}" for s in switches)
            + "."
        ),
    }


def degeneracy_check(
    orders: pd.DataFrame,
    p_by_mode: dict,
    value_col: str,
    mode_profile: dict,
) -> dict:
    """
    Demonstrate why the cost model needs its time terms (documented at module level).

    Re-runs the portfolio optimisation with holding cost and value of speed set to zero -
    section 25's formula taken literally - and reports the resulting mode mix. Standard
    Class should absorb every order, which is the degenerate outcome the time terms exist
    to avoid.
    """
    stripped = resolve_assumptions(
        {"holding_rate_per_day": 0.0, "speed_value_per_day": 0.0}
    )
    values = pd.to_numeric(orders[value_col], errors="coerce").fillna(0.0).to_numpy(float)
    costs = _portfolio_costs(values, p_by_mode, stripped, mode_profile)
    available = list(costs.keys())
    matrix = np.column_stack([costs[m] for m in available])
    choice = np.argmin(matrix, axis=1)
    mix = pd.Series([available[i] for i in choice]).value_counts(normalize=True)

    full = resolve_assumptions()
    full_costs = _portfolio_costs(values, p_by_mode, full, mode_profile)
    full_matrix = np.column_stack([full_costs[m] for m in available])
    full_mix = pd.Series(
        [available[i] for i in np.argmin(full_matrix, axis=1)]
    ).value_counts(normalize=True)

    dominant_mode = str(mix.index[0])
    dominant_share = float(mix.iloc[0])

    return {
        "degenerate_mix": {k: round(float(v), 4) for k, v in mix.items()},
        "full_mix": {k: round(float(v), 4) for k, v in full_mix.items()},
        "dominant_mode": dominant_mode,
        "dominant_share": round(dominant_share, 4),
        "is_degenerate": bool(dominant_share >= 0.99),
        "explanation": (
            "With the time-related terms removed, the cost function reduces to "
            "freight + P(late) x penalty, exactly as written in the brief. Standard Class "
            "then has both the lowest assumed freight cost and the lowest observed late "
            f"rate, so it absorbs {dominant_share:.1%} of orders and the optimisation "
            "returns effectively one answer whatever the other assumptions are. The only "
            "orders that escape are those valuable enough for the percentage-of-value "
            "penalty to outweigh a large freight gap - a handful at the top of the value "
            "distribution. The holding and speed terms are what make the trade-off real: "
            "they price the fact that faster modes deliver sooner even when they miss "
            "their promise."
        ),
    }


def promise_redesign_analysis(mode_profile: dict) -> dict:
    """
    Quantify the zero-cost lever this dataset exposes (spec sections 29-30).

    Second Class and Standard Class have statistically identical realised transit
    distributions, but Second Class promises 2 days against Standard's 4. Its far higher
    late rate is therefore produced by the commitment, not by the operation. Re-promising
    Second Class at the same window as Standard would remove that gap without changing
    anything physical.

    Reported as a scenario-implied estimate: it assumes the transit distribution is
    unchanged by the promise, which holds in this data because transit does not depend on
    anything the promise touches.
    """
    second = mode_profile.get("Second Class")
    standard = mode_profile.get("Standard Class")
    if not second or not standard:
        return {"available": False}

    transit_gap = abs(second["mean_transit"] - standard["mean_transit"])
    identical = transit_gap < 0.05

    return {
        "available": True,
        "second_class": second,
        "standard_class": standard,
        "transit_gap_days": round(transit_gap, 4),
        "transit_identical": bool(identical),
        "current_late_rate": second["late_rate"],
        "implied_late_rate": standard["late_rate"],
        "late_rate_reduction": round(second["late_rate"] - standard["late_rate"], 4),
        "orders_affected": second["orders"],
        "finding": (
            f"Second Class and Standard Class average {second['mean_transit']:.2f} and "
            f"{standard['mean_transit']:.2f} days in transit respectively - a difference "
            f"of {transit_gap:.3f} days, which is not operationally meaningful. Their late "
            f"rates differ enormously ({second['late_rate']:.1%} against "
            f"{standard['late_rate']:.1%}) only because Second Class promises "
            f"{second['promised_days']} days where Standard Class promises "
            f"{standard['promised_days']}."
        ),
        "recommendation": (
            "Re-setting the Second Class delivery promise to match Standard Class would be "
            "expected to bring its late rate down to roughly the Standard Class level, "
            "without changing carriers, routes or cost. This is a commitment-setting "
            "decision, not a logistics investment. It should be weighed against the "
            "commercial value customers place on the shorter promise, which this dataset "
            "cannot measure."
        ),
    }
