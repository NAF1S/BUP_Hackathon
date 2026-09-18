"""Correctness against the 10 public sample cases.

Three independent properties are checked:

1. the LP reaches the organizer's reference optimum (never worse),
2. the returned plan passes the independent replay verifier,
3. both the LLM path and the offline fallback path produce valid plans.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.heuristics import interpret_notes
from app.optimization.constraints import compile_constraints
from app.optimization.replay import verify_plan
from app.optimization.solver import heuristic_only, optimize
from app.pipeline import GridWiseService, _plan_entries
from app.schemas import DirectiveInterpretation, ScenarioRequest
from tests.support import FakeLLMClient, interpretation_payload, signature

TOLERANCE = 0.01


def _request(case: dict[str, Any]) -> ScenarioRequest:
    return ScenarioRequest(**case["input"])


def _truth(case: dict[str, Any]) -> list[DirectiveInterpretation]:
    return [
        DirectiveInterpretation(**entry)
        for entry in case["expected_output"]["directive_interpretation"]
    ]


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------


def test_solver_reaches_reference_optimum(cases: list[dict[str, Any]]) -> None:
    """Our LP must never be more expensive than the organizer's optimum."""
    failures: list[str] = []
    for case in cases:
        request = _request(case)
        truth = _truth(case)
        result = optimize(request, compile_constraints(truth), tolerance=TOLERANCE)
        reference = float(case["expected_output"]["total_cost_bdt"])
        if result.status != "optimal":
            failures.append(f"{case['id']}: status={result.status}")
        elif result.objective > reference + TOLERANCE:
            failures.append(
                f"{case['id']}: {result.objective:.4f} > reference {reference:.4f}"
            )
    assert not failures, "; ".join(failures)


def test_solver_matches_reference_cost_exactly(cases: list[dict[str, Any]]) -> None:
    """Every public reference schedule is provably optimal, so costs must tie."""
    for case in cases:
        request = _request(case)
        result = optimize(request, compile_constraints(_truth(case)), tolerance=TOLERANCE)
        reference = float(case["expected_output"]["total_cost_bdt"])
        assert result.objective == pytest.approx(reference, abs=TOLERANCE), case["id"]


def test_replay_accepts_reference_directive_plans(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        request = _request(case)
        truth = _truth(case)
        result = optimize(request, compile_constraints(truth), tolerance=TOLERANCE)
        report = verify_plan(request, truth, _plan_entries(result), tolerance=TOLERANCE)
        assert report.ok, f"{case['id']}: {[str(v) for v in report.violations]}"


def test_reported_totals_match_plan(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        request = _request(case)
        result = optimize(request, compile_constraints(_truth(case)), tolerance=TOLERANCE)
        plan = _plan_entries(result)
        tariff = request.tariff()
        expected_cost = sum(entry.grid_kwh * tariff[entry.hour] for entry in plan)
        assert sum(entry.grid_kwh for entry in plan) == pytest.approx(
            sum(result.grid), abs=TOLERANCE
        )
        assert expected_cost == pytest.approx(result.objective, abs=TOLERANCE)


def test_no_simultaneous_charge_and_discharge(cases: list[dict[str, Any]]) -> None:
    """The netting post-process must never leave dual action in one hour."""
    for case in cases:
        request = _request(case)
        result = optimize(request, compile_constraints(_truth(case)), tolerance=TOLERANCE)
        for hour in range(24):
            assert not (result.charge[hour] > 1e-9 and result.discharge[hour] > 1e-9), (
                f"{case['id']} hour {hour}"
            )


def test_heuristic_rescue_produces_valid_plans(cases: list[dict[str, Any]]) -> None:
    """The dependency-free path must always yield a replay-clean schedule."""
    for case in cases:
        request = _request(case)
        truth = _truth(case)
        result = heuristic_only(request, compile_constraints(truth), tolerance=TOLERANCE)
        report = verify_plan(request, truth, _plan_entries(result), tolerance=TOLERANCE)
        assert report.ok, f"{case['id']}: {[str(v) for v in report.violations]}"


# ---------------------------------------------------------------------------
# deterministic fallback interpreter
# ---------------------------------------------------------------------------


def test_rule_interpreter_matches_public_ground_truth(cases: list[dict[str, Any]]) -> None:
    """The offline interpreter is the last recovery stage, so it must be exact here."""
    failures: list[str] = []
    for case in cases:
        request = _request(case)
        predicted = interpret_notes(
            request.operator_notes, request.battery.capacity_kwh
        )
        expected = _truth(case)
        assert len(predicted) == len(expected), case["id"]
        for got, want in zip(predicted, expected):
            if signature(got) != signature(want):
                failures.append(
                    f"{case['id']} note {want.note_index}: {signature(got)} != {signature(want)}"
                )
    assert not failures, "; ".join(failures)


# ---------------------------------------------------------------------------
# full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("use_llm", [True, False])
def test_pipeline_produces_valid_response(
    cases: list[dict[str, Any]], settings, use_llm: bool
) -> None:
    for case in cases:
        request = _request(case)
        truth = _truth(case)
        client = (
            FakeLLMClient(interpretation_payload([entry.model_dump(mode="json") for entry in truth]))
            if use_llm
            else None
        )
        service = GridWiseService(settings, client)
        try:
            response = asyncio.run(service.optimize(request))
        finally:
            asyncio.run(service.aclose())

        assert response.scenario_id == request.scenario_id
        assert len(response.hourly_plan) == 24
        assert len(response.directive_interpretation) == len(request.operator_notes)

        reference = float(case["expected_output"]["total_cost_bdt"])
        assert response.total_cost_bdt <= reference + TOLERANCE, case["id"]

        report = verify_plan(
            request,
            response.directive_interpretation,
            response.hourly_plan,
            tolerance=TOLERANCE,
        )
        assert report.ok, f"{case['id']}: {[str(v) for v in report.violations]}"

        if use_llm:
            assert service.counters["fallbacks"] == 0, case["id"]
            assert service.counters["llm_calls"] == 1, case["id"]


def test_pipeline_is_deterministic(cases: list[dict[str, Any]], settings) -> None:
    case = cases[0]
    request = _request(case)
    service = GridWiseService(settings, None)
    try:
        first = asyncio.run(service.optimize(request))
        second = asyncio.run(service.optimize(request))
    finally:
        asyncio.run(service.aclose())
    assert first.model_dump() == second.model_dump()
