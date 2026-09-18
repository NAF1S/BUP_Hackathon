"""Solver behaviour: optimality, feasibility recovery, and fallback tiers."""

from __future__ import annotations

from typing import Any

import pytest

from app.optimization import solver as solver_module
from app.optimization.constraints import DirectiveConstraints, compile_constraints
from app.optimization.replay import verify_plan
from app.pipeline import _plan_entries
from app.schemas import DirectiveInterpretation, DirectiveType, ScenarioRequest

TOLERANCE = 0.01


def make_scenario(**overrides: Any) -> ScenarioRequest:
    """A flat baseline scenario that tests then perturb."""
    hours = [
        {"hour": hour, "demand_kwh": 100.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": 10.0}
        for hour in range(24)
    ]
    battery = {
        "capacity_kwh": 300.0,
        "initial_energy_kwh": 100.0,
        "minimum_energy_kwh": 0.0,
        "max_charge_kwh_per_hour": 100.0,
        "max_discharge_kwh_per_hour": 100.0,
    }
    payload: dict[str, Any] = {
        "scenario_id": "SYNTH",
        "operator_notes": ["synthetic note"],
        "hours": hours,
        "battery": battery,
    }
    if "battery" in overrides:
        battery.update(overrides.pop("battery"))
        payload["battery"] = battery
    payload.update(overrides)
    return ScenarioRequest(**payload)


def no_op_entry() -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=0,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        structured_adjustment=None,
        explanation="no-op",
    )


def test_energy_balance_holds_exactly() -> None:
    request = make_scenario()
    result = solver_module.optimize(request, compile_constraints([no_op_entry()]))
    for hour in range(24):
        charge = result.charge[hour]
        discharge = result.discharge[hour]
        lhs = result.grid[hour] + result.solar_used[hour] + discharge
        rhs = request.demand()[hour] + charge
        assert lhs == pytest.approx(rhs, abs=1e-6), f"hour {hour}"


def test_end_of_day_neutrality_is_enforced() -> None:
    request = make_scenario()
    result = solver_module.optimize(request, compile_constraints([no_op_entry()]))
    assert result.soc[-1] == pytest.approx(request.battery.initial_energy_kwh, abs=1e-6)


def test_charges_cheap_and_discharges_expensive() -> None:
    """Classic arbitrage: a price spike should be served from the battery."""
    hours = []
    for hour in range(24):
        tariff = 30.0 if hour == 19 else 5.0
        hours.append({"hour": hour, "demand_kwh": 100.0, "solar_kwh": 0.0, "tariff_bdt_per_kwh": tariff})
    request = make_scenario(hours=hours)
    result = solver_module.optimize(request, compile_constraints([no_op_entry()]))

    assert result.discharge[19] > 0.0
    assert result.grid[19] < 100.0
    # Displacing 30 BDT energy must beat a do-nothing plan of 24 * 100 * 5 + spike.
    baseline = sum(100.0 * entry["tariff_bdt_per_kwh"] for entry in hours)
    assert result.objective < baseline


def test_grid_cap_forces_precharging() -> None:
    constraints = DirectiveConstraints()
    for hour in (18, 19, 20):
        constraints.grid_cap_kwh[hour] = 150.0

    request = make_scenario()
    result = solver_module.optimize(request, constraints)
    report = verify_plan(request, [no_op_entry()], _plan_entries(result), tolerance=TOLERANCE)

    for hour in (18, 19, 20):
        assert result.grid[hour] <= 150.0 + TOLERANCE
    # Battery energy must have been accumulated before the capped window.
    assert max(result.soc[:18]) > request.battery.initial_energy_kwh
    assert report.ok, [str(v) for v in report.violations]


def test_solar_reduction_caps_usable_solar() -> None:
    hours = [
        {"hour": hour, "demand_kwh": 50.0, "solar_kwh": 100.0, "tariff_bdt_per_kwh": 10.0}
        for hour in range(24)
    ]
    request = make_scenario(hours=hours)
    interpretation = [
        DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type=DirectiveType.SOLAR_REDUCTION,
            structured_adjustment={"hours": [12], "factor": 0.5},
            explanation="half solar",
        )
    ]
    result = solver_module.optimize(request, compile_constraints(interpretation))
    assert result.solar_used[12] <= 50.0 + TOLERANCE
    assert result.solar_used[11] == pytest.approx(50.0, abs=TOLERANCE)


def test_no_charge_window_forces_zero_charge() -> None:
    interpretation = [
        DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type=DirectiveType.NO_CHARGE_WINDOW,
            structured_adjustment={"hours": [2, 3, 4]},
            explanation="charger offline",
        )
    ]
    request = make_scenario()
    result = solver_module.optimize(request, compile_constraints(interpretation))
    for hour in (2, 3, 4):
        assert result.charge[hour] == 0.0


def test_minimum_reserve_is_respected() -> None:
    interpretation = [
        DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type=DirectiveType.MINIMUM_BATTERY_RESERVE,
            structured_adjustment={"hours": [18, 19, 20, 21], "minimum_energy_kwh": 250.0},
            explanation="reserve",
        )
    ]
    request = make_scenario()
    result = solver_module.optimize(request, compile_constraints(interpretation))
    for hour in (18, 19, 20, 21):
        assert result.soc[hour] >= 250.0 - TOLERANCE


def test_infeasible_scenario_degrades_instead_of_raising() -> None:
    """A reserve that cannot be reached (no charge rate) must not crash."""
    request = make_scenario(
        battery={"capacity_kwh": 100.0, "initial_energy_kwh": 10.0, "max_charge_kwh_per_hour": 0.0}
    )
    constraints = DirectiveConstraints()
    constraints.reserve_kwh[5] = 50.0

    result = solver_module.optimize(request, constraints)
    assert len(result.grid) == 24
    assert result.engine in {"highs-elastic", "heuristic"}
    assert result.relaxations


def test_solver_unavailable_falls_back_to_heuristic(monkeypatch) -> None:
    monkeypatch.setattr(
        solver_module, "_try_linprog", lambda *args, **kwargs: ("solver_unavailable", None)
    )
    request = make_scenario()
    constraints = compile_constraints([no_op_entry()])
    result = solver_module.optimize(request, constraints)

    assert result.engine == "heuristic"
    report = verify_plan(request, [no_op_entry()], _plan_entries(result), tolerance=TOLERANCE)
    assert report.ok, [str(v) for v in report.violations]


def test_heuristic_respects_no_charge_and_cap_together() -> None:
    constraints = DirectiveConstraints()
    constraints.charge_allowed[2] = False
    constraints.charge_allowed[3] = False
    for hour in (18, 19, 20):
        constraints.grid_cap_kwh[hour] = 150.0

    request = make_scenario()
    result = solver_module.heuristic_only(request, constraints)
    report = verify_plan(request, [no_op_entry()], _plan_entries(result), tolerance=TOLERANCE)

    assert result.charge[2] == 0.0 and result.charge[3] == 0.0
    assert report.ok, [str(v) for v in report.violations]


def test_replay_catches_a_tampered_plan() -> None:
    """The verifier must actually reject a plan that breaks a directive."""
    request = make_scenario()
    noise = no_op_entry()

    interpretation = [
        DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type=DirectiveType.MAX_GRID_WINDOW,
            structured_adjustment={"hours": [10], "max_grid_kwh": 1.0},
            explanation="cap",
        )
    ]
    good = solver_module.optimize(request, compile_constraints(interpretation))
    plan = _plan_entries(good)
    assert verify_plan(request, interpretation, plan, tolerance=TOLERANCE).ok

    # Tamper: raise grid in the capped hour and rebalance solar to keep balance.
    tampered = list(plan)
    original = tampered[10]
    tampered[10] = original.model_copy(update={"grid_kwh": original.grid_kwh + 50.0})
    report = verify_plan(request, interpretation, tampered, tolerance=TOLERANCE)
    assert not report.ok
    assert "grid_cap" in report.codes()
    assert noise is not None
