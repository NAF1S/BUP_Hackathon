"""End-to-end request orchestration.

    notes ──LLM──▶ raw text ──guardrails──▶ directives ──compile──▶ constraints
                                                            │
                                            ┌───────────────┘
                                            ▼
                                     LP solve ──▶ hourly_plan ──replay──▶ verified
                                                                  │
                                                          (repair if needed)

The optimizer never sees natural language, and the replay verifier never trusts
the optimizer. Recovery ladder when the model misbehaves:

    1. LLM call with guardrail validation
    2. one retry carrying the validation errors back to the model
    3. deterministic rule-based interpreter for any note still missing
    4. guaranteed-correctness heuristic solver if replay rejects the LP result
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Sequence

from app.cache import TTLCache, cache_key
from app.config import Settings
from app.errors import LLMError
from app.guardrails.normalize import extract_json_payload, normalize_interpretation
from app.guardrails.validate import check_interpretation
from app.heuristics import interpret_notes
from app.llm.base import LLMClient
from app.llm.openai_compat import build_llm_client
from app.llm.prompts import build_messages, describe_feedback
from app.optimization import solver as solver_module
from app.optimization.constraints import compile_constraints
from app.optimization.replay import verify_plan
from app.schemas import (
    BatteryAction,
    DirectiveInterpretation,
    DirectiveType,
    HourlyPlanEntry,
    OptimizeResponse,
    ScenarioRequest,
)

logger = logging.getLogger(__name__)

HORIZON = 24
_ACTION_TOLERANCE = 1e-9
_SUMMARY_LIMIT = 900


class GridWiseService:
    """Stateless request handler; safe to share across concurrent requests."""

    def __init__(self, settings: Settings, llm_client: LLMClient | None = None) -> None:
        self._settings = settings
        self._llm = llm_client
        self._cache = TTLCache(settings.cache_size, settings.cache_ttl_seconds)
        self._semaphore = asyncio.Semaphore(settings.llm_max_concurrency)
        self.counters = {"requests": 0, "cache_hits": 0, "llm_calls": 0, "fallbacks": 0}

    @classmethod
    def from_settings(cls, settings: Settings) -> "GridWiseService":
        """Build the production service, wiring the configured provider."""
        return cls(settings, build_llm_client(settings))

    async def aclose(self) -> None:
        if self._llm is not None:
            await self._llm.aclose()

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    async def optimize(self, request: ScenarioRequest) -> OptimizeResponse:
        """Produce the interpretation and the final 24-hour plan."""
        key = cache_key(request)
        cached = self._cache.get(key)
        if cached is not None:
            self.counters["cache_hits"] += 1
            return cached

        self.counters["requests"] += 1
        response = await self._optimize_uncached(request)
        self._cache.put(key, response)
        return response

    # ------------------------------------------------------------------
    # pipeline stages
    # ------------------------------------------------------------------

    async def _optimize_uncached(self, request: ScenarioRequest) -> OptimizeResponse:
        started = time.monotonic()
        deadline = started + self._settings.request_timeout_seconds

        interpretation, warnings, source = await self._interpret(request, deadline=deadline)

        policy = check_interpretation(interpretation, len(request.operator_notes), request.battery)
        for problem in policy:
            logger.warning("interpretation policy note: %s", problem)

        constraints = compile_constraints(interpretation)
        result = solver_module.optimize(
            request,
            constraints,
            epsilon=self._settings.battery_cycling_epsilon,
            tolerance=self._settings.numeric_tolerance,
        )
        plan = _plan_entries(result)
        report = verify_plan(
            request, interpretation, plan, tolerance=self._settings.numeric_tolerance
        )

        if not report.ok:
            logger.warning(
                "replay rejected the %s plan (%s); trying the heuristic path",
                result.engine,
                ", ".join(sorted(set(report.codes()))),
            )
            alternative = solver_module.heuristic_only(
                request, constraints, tolerance=self._settings.numeric_tolerance
            )
            alternative_plan = _plan_entries(alternative)
            alternative_report = verify_plan(
                request,
                interpretation,
                alternative_plan,
                tolerance=self._settings.numeric_tolerance,
            )
            if len(alternative_report.violations) < len(report.violations):
                result, plan, report = alternative, alternative_plan, alternative_report
                warnings.append("solver_switched_after_replay_mismatch")

        if not report.ok:
            warnings.append("plan_does_not_fully_satisfy_all_constraints")
            logger.error(
                "plan failed replay verification: %s",
                "; ".join(str(violation) for violation in report.violations[:5]),
            )

        tariff = request.tariff()
        total_grid, total_cost, peak = _totals(plan, tariff)
        summary = _build_summary(interpretation, result, total_grid, total_cost, peak)

        logger.info(
            "scenario=%s source=%s engine=%s cost=%.4f grid=%.4f peak=%.4f elapsed=%.3fs warnings=%d",
            request.scenario_id,
            source,
            result.engine,
            total_cost,
            total_grid,
            peak,
            time.monotonic() - started,
            len(warnings),
        )

        return OptimizeResponse(
            scenario_id=request.scenario_id,
            directive_interpretation=list(interpretation),
            hourly_plan=plan,
            total_grid_kwh=total_grid,
            total_cost_bdt=total_cost,
            peak_grid_kwh=peak,
            plan_summary=summary,
        )

    async def _interpret(
        self, request: ScenarioRequest, *, deadline: float
    ) -> tuple[list[DirectiveInterpretation], list[str], str]:
        """Recover a complete, validated interpretation for every note."""
        notes = request.operator_notes
        slots: list[DirectiveInterpretation | None] = [None] * len(notes)
        problems: list[str] = []
        llm_used = False

        if self._llm is not None:
            raw = await self._safe_call(request, deadline=deadline)
            if raw is not None:
                llm_used = True
                slots, parse_problems = self._parse(raw, request)
                problems.extend(parse_problems)

            if not all(slot is not None for slot in slots):
                remaining = deadline - time.monotonic()
                if remaining > 2.0:
                    retry_raw = await self._safe_call(
                        request,
                        deadline=deadline,
                        feedback=describe_feedback(problems) or ["no valid entry produced"],
                    )
                    if retry_raw is not None:
                        retry_slots, retry_problems = self._parse(retry_raw, request)
                        problems.extend(retry_problems)
                        # Keep anything the retry fixed; keep first-attempt wins too.
                        slots = [
                            retry if retry is not None else first
                            for retry, first in zip(retry_slots, slots)
                        ]

        missing = [index for index, slot in enumerate(slots) if slot is None]
        if missing:
            if self._settings.allow_deterministic_fallback:
                for entry in interpret_notes(notes, request.battery.capacity_kwh):
                    if slots[entry.note_index] is None:
                        slots[entry.note_index] = entry
                        self.counters["fallbacks"] += 1
                        problems.append(
                            f"note {entry.note_index}: rule-based fallback interpretation applied"
                        )
            for index in missing:
                if slots[index] is None:  # pragma: no cover - defensive
                    slots[index] = _no_op(index)

        if problems:
            logger.info("interpretation warnings for %s: %s", request.scenario_id, problems)

        final = [slot if slot is not None else _no_op(i) for i, slot in enumerate(slots)]
        source = "llm" if llm_used and not problems else ("llm+fallback" if llm_used else "fallback")
        return final, problems, source

    async def _safe_call(
        self, request: ScenarioRequest, *, deadline: float, feedback: Sequence[str] | None = None
    ) -> str | None:
        """Call the model, converting every failure into a logged ``None``."""
        if self._llm is None:
            return None
        remaining = deadline - time.monotonic()
        if remaining <= 1.0:
            logger.warning("skipping model call: %.3fs of budget left", remaining)
            return None
        messages = build_messages(
            request.scenario_id, request.operator_notes, request.battery, feedback=feedback
        )
        try:
            self.counters["llm_calls"] += 1
            async with self._semaphore:
                return await asyncio.wait_for(
                    self._llm.complete_json(
                        messages[0]["content"], messages[1]["content"], deadline=deadline
                    ),
                    timeout=remaining,
                )
        except asyncio.TimeoutError:
            logger.warning("model call exceeded the request budget")
        except LLMError as exc:
            logger.warning("model call failed: %s", exc.message)
        except Exception as exc:  # noqa: BLE001 - never let a provider bug 5xx us
            logger.warning("unexpected model failure: %s", type(exc).__name__)
        return None

    @staticmethod
    def _parse(
        raw: str, request: ScenarioRequest
    ) -> tuple[list[DirectiveInterpretation | None], list[str]]:
        try:
            payload = extract_json_payload(raw)
        except ValueError as exc:
            return [None] * len(request.operator_notes), [f"model output not parsable as JSON: {exc}"]
        try:
            return normalize_interpretation(payload, request.operator_notes, request.battery)
        except Exception as exc:  # noqa: BLE001 - malformed shapes must not escape
            return [None] * len(request.operator_notes), [
                f"model output failed normalization: {type(exc).__name__}"
            ]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _no_op(index: int) -> DirectiveInterpretation:
    return DirectiveInterpretation(
        note_index=index,
        applies=False,
        directive_type=DirectiveType.NO_OP,
        structured_adjustment=None,
        explanation="No supported energy directive was extracted from this note.",
    )


def _plan_entries(result: Any) -> list[HourlyPlanEntry]:
    """Convert solver output into the response schema, with exact action labels."""
    entries: list[HourlyPlanEntry] = []
    for hour in range(HORIZON):
        charge = result.charge[hour]
        discharge = result.discharge[hour]
        if charge > _ACTION_TOLERANCE:
            action, magnitude = BatteryAction.CHARGE, charge
        elif discharge > _ACTION_TOLERANCE:
            action, magnitude = BatteryAction.DISCHARGE, discharge
        else:
            action, magnitude = BatteryAction.IDLE, 0.0
        entries.append(
            HourlyPlanEntry(
                hour=hour,
                grid_kwh=round(max(result.grid[hour], 0.0), 6),
                solar_used_kwh=round(max(result.solar_used[hour], 0.0), 6),
                battery_action=action,
                battery_kwh=round(max(magnitude, 0.0), 6),
                battery_energy_after_kwh=round(max(result.soc[hour], 0.0), 6),
            )
        )
    return entries


def _totals(plan: Sequence[HourlyPlanEntry], tariff: Sequence[float]) -> tuple[float, float, float]:
    """Recompute the three reported totals from ``hourly_plan`` itself."""
    total_grid = round(sum(entry.grid_kwh for entry in plan), 6)
    total_cost = round(sum(entry.grid_kwh * tariff[entry.hour] for entry in plan), 6)
    peak = round(max((entry.grid_kwh for entry in plan), default=0.0), 6)
    return total_grid, total_cost, peak


def _format_hours(hours: Sequence[int]) -> str:
    if not hours:
        return "none"
    ordered = sorted(hours)
    ranges: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for hour in ordered[1:]:
        if hour == previous + 1:
            previous = hour
            continue
        ranges.append((start, previous))
        start = previous = hour
    ranges.append((start, previous))
    return ", ".join(f"{low:02d}:00-{high + 1:02d}:00" for low, high in ranges)


def _describe_directive(entry: DirectiveInterpretation) -> str:
    adjustment = entry.structured_adjustment or {}
    hours = adjustment.get("hours", [])
    span = _format_hours(hours)
    kind = entry.directive_type.value
    if kind == "solar_reduction":
        return f"solar limited to {float(adjustment.get('factor', 1.0)):.2f}x for {span}"
    if kind == "minimum_battery_reserve":
        return f"battery reserve >= {float(adjustment.get('minimum_energy_kwh', 0.0)):.0f} kWh for {span}"
    if kind == "no_charge_window":
        return f"charging blocked for {span}"
    if kind == "no_discharge_window":
        return f"discharging blocked for {span}"
    if kind == "max_grid_window":
        return f"grid import <= {float(adjustment.get('max_grid_kwh', 0.0)):.0f} kWh for {span}"
    return "no schedule change"


def _build_summary(
    interpretation: Sequence[DirectiveInterpretation],
    result: Any,
    total_grid: float,
    total_cost: float,
    peak: float,
) -> str:
    """Deterministic human-readable strategy explanation (no model required).

    Deriving this from the solved plan rather than a second model call keeps it
    consistent with the returned numbers and removes a failure mode from the
    critical path.
    """
    applied = [entry for entry in interpretation if entry.applies]
    parts: list[str] = []

    if applied:
        parts.append(
            "Directives applied: "
            + "; ".join(_describe_directive(entry) for entry in applied)
            + "."
        )
    else:
        parts.append("No operator note changed the 24-hour schedule.")

    charge_hours = [hour for hour in range(HORIZON) if result.charge[hour] > _ACTION_TOLERANCE]
    discharge_hours = [hour for hour in range(HORIZON) if result.discharge[hour] > _ACTION_TOLERANCE]
    if charge_hours:
        parts.append(f"Battery charged during {_format_hours(charge_hours)}.")
    if discharge_hours:
        parts.append(f"Battery discharged during {_format_hours(discharge_hours)}.")
    if not charge_hours and not discharge_hours:
        parts.append("Battery idle for the full horizon.")

    parts.append(
        f"Total grid energy {total_grid:.1f} kWh at a cost of {total_cost:.2f} BDT, "
        f"with peak import {peak:.1f} kWh."
    )
    if getattr(result, "relaxations", None):
        parts.append("Note: some base constraints were relaxed to return a feasible plan.")

    return " ".join(parts)[:_SUMMARY_LIMIT]
