"""Independent hour-by-hour replay verifier.

The judge replays the returned plan using the *organizer's ground-truth*
directives, not the directives we reported. If the model builder and the checker
shared an implementation bug, that check would be worthless - so this module
re-derives every rule from the raw request and the raw interpretation entries,
with no import from :mod:`app.optimization.constraints` or
:mod:`app.optimization.solver`.

It consumes only ``(request, interpretation, hourly_plan)`` and reports
violations; it never mutates anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

HORIZON = 24


@dataclass
class Violation:
    """A single rule breach found during replay."""

    code: str
    hour: int | None
    detail: str

    def __str__(self) -> str:  # pragma: no cover - diagnostics helper
        where = f"hour {self.hour}" if self.hour is not None else "plan"
        return f"{self.code} @ {where}: {self.detail}"


@dataclass
class ReplayReport:
    """Outcome of a full replay."""

    violations: list[Violation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    total_grid_kwh: float = 0.0
    total_cost_bdt: float = 0.0
    peak_grid_kwh: float = 0.0

    @property
    def ok(self) -> bool:
        return not self.violations

    def codes(self) -> list[str]:
        return [v.code for v in self.violations]


def _dtype(entry: Any) -> str:
    value = getattr(entry, "directive_type", None)
    return getattr(value, "value", value) or "no_op"


def _active(entry: Any) -> bool:
    if not getattr(entry, "applies", False):
        return False
    return _dtype(entry) != "no_op"


def derive_hourly_rules(
    request: Any, interpretation: Iterable[Any]
) -> dict[str, list]:
    """Re-derive per-hour rule tables straight from the interpretation entries."""
    base_solar = request.solar()
    capacity = request.battery.capacity_kwh
    base_min = request.battery.minimum_energy_kwh
    base_charge = request.battery.max_charge_kwh_per_hour
    base_discharge = request.battery.max_discharge_kwh_per_hour

    effective_solar = list(base_solar)
    soc_min = [base_min] * HORIZON
    charge_cap = [base_charge] * HORIZON
    discharge_cap = [base_discharge] * HORIZON
    grid_cap: list[float | None] = [None] * HORIZON

    for entry in interpretation:
        if not _active(entry):
            continue
        kind = _dtype(entry)
        adjustment = getattr(entry, "structured_adjustment", None) or {}
        hours = [h for h in adjustment.get("hours", []) if isinstance(h, int) and 0 <= h < HORIZON]

        if kind == "solar_reduction":
            factor = float(adjustment.get("factor", 1.0))
            for hour in hours:
                effective_solar[hour] = base_solar[hour] * factor
        elif kind == "minimum_battery_reserve":
            reserve = float(adjustment.get("minimum_energy_kwh", base_min))
            for hour in hours:
                soc_min[hour] = max(soc_min[hour], min(reserve, capacity))
        elif kind == "no_charge_window":
            for hour in hours:
                charge_cap[hour] = 0.0
        elif kind == "no_discharge_window":
            for hour in hours:
                discharge_cap[hour] = 0.0
        elif kind == "max_grid_window":
            cap = float(adjustment.get("max_grid_kwh", math.inf))
            for hour in hours:
                grid_cap[hour] = cap if grid_cap[hour] is None else min(grid_cap[hour], cap)

    return {
        "effective_solar": effective_solar,
        "soc_min": soc_min,
        "charge_cap": charge_cap,
        "discharge_cap": discharge_cap,
        "grid_cap": grid_cap,
    }


def verify_plan(
    request: Any,
    interpretation: Iterable[Any],
    hourly_plan: Iterable[Any],
    *,
    tolerance: float = 0.01,
) -> ReplayReport:
    """Replay the plan hour by hour against every energy and directive rule."""
    report = ReplayReport()
    rules = derive_hourly_rules(request, interpretation)

    plan = list(hourly_plan)
    if len(plan) != HORIZON:
        report.violations.append(
            Violation("plan_length", None, f"expected {HORIZON} hourly entries, found {len(plan)}")
        )
        return report

    hours = [int(entry.hour) for entry in plan]
    if sorted(hours) != list(range(HORIZON)):
        report.violations.append(
            Violation("plan_hours", None, "hourly_plan must contain each hour 0-23 exactly once")
        )
        return report

    by_hour = {int(entry.hour): entry for entry in plan}
    demand = request.demand()
    tariff = request.tariff()
    capacity = request.battery.capacity_kwh
    initial = request.battery.initial_energy_kwh

    previous_soc = initial
    for hour in range(HORIZON):
        entry = by_hour[hour]
        grid = float(entry.grid_kwh)
        solar_used = float(entry.solar_used_kwh)
        action = getattr(entry.battery_action, "value", entry.battery_action)
        battery_kwh = float(entry.battery_kwh)
        soc = float(entry.battery_energy_after_kwh)

        for name, value in (
            ("grid_kwh", grid),
            ("solar_used_kwh", solar_used),
            ("battery_kwh", battery_kwh),
            ("battery_energy_after_kwh", soc),
        ):
            if not math.isfinite(value):
                report.violations.append(Violation("non_finite", hour, f"{name} is not finite"))
            elif value < -tolerance:
                report.violations.append(Violation("negative_value", hour, f"{name}={value}"))

        # Action semantics: exactly one action, magnitude only when acting.
        if action == "idle" and abs(battery_kwh) > tolerance:
            report.violations.append(
                Violation("action_consistency", hour, f"idle hour reports battery_kwh={battery_kwh}")
            )
        if action not in ("charge", "discharge", "idle"):
            report.violations.append(Violation("action_value", hour, f"unknown action {action!r}"))
        if action in ("charge", "discharge") and battery_kwh <= 0.0:
            report.violations.append(
                Violation("action_consistency", hour, f"{action} hour reports battery_kwh={battery_kwh}")
            )

        charge = battery_kwh if action == "charge" else 0.0
        discharge = battery_kwh if action == "discharge" else 0.0

        # Rate limits (including directive-forced zero windows).
        if charge > rules["charge_cap"][hour] + tolerance:
            report.violations.append(
                Violation(
                    "charge_rate",
                    hour,
                    f"charge {charge} exceeds limit {rules['charge_cap'][hour]}",
                )
            )
        if discharge > rules["discharge_cap"][hour] + tolerance:
            report.violations.append(
                Violation(
                    "discharge_rate",
                    hour,
                    f"discharge {discharge} exceeds limit {rules['discharge_cap'][hour]}",
                )
            )

        # Energy balance.
        imbalance = (grid + solar_used + discharge) - (demand[hour] + charge)
        if abs(imbalance) > tolerance:
            report.violations.append(
                Violation("energy_balance", hour, f"imbalance {imbalance:.6f} kWh")
            )

        # Effective-solar cap after solar_reduction.
        if solar_used > rules["effective_solar"][hour] + tolerance:
            report.violations.append(
                Violation(
                    "solar_overuse",
                    hour,
                    f"solar_used {solar_used} exceeds effective "
                    f"{rules['effective_solar'][hour]:.6f}",
                )
            )

        # Storage dynamics.
        expected_soc = previous_soc + charge - discharge
        if abs(soc - expected_soc) > tolerance:
            report.violations.append(
                Violation(
                    "soc_transition",
                    hour,
                    f"reported soc {soc} but transitions give {expected_soc:.6f}",
                )
            )

        # Bounds, including directive-raised reserve.
        if soc > capacity + tolerance:
            report.violations.append(Violation("soc_capacity", hour, f"soc {soc} > capacity {capacity}"))
        if soc < rules["soc_min"][hour] - tolerance:
            report.violations.append(
                Violation(
                    "soc_reserve",
                    hour,
                    f"soc {soc} below active minimum {rules['soc_min'][hour]}",
                )
            )

        # Grid cap.
        cap = rules["grid_cap"][hour]
        if cap is not None and grid > cap + tolerance:
            report.violations.append(
                Violation("grid_cap", hour, f"grid {grid} exceeds cap {cap}")
            )

        previous_soc = soc
        report.total_grid_kwh += grid
        report.total_cost_bdt += grid * tariff[hour]
        report.peak_grid_kwh = max(report.peak_grid_kwh, grid)

    # End-of-day neutrality.
    if abs(previous_soc - initial) > tolerance:
        report.violations.append(
            Violation(
                "end_of_day_neutrality",
                None,
                f"final battery energy {previous_soc} must equal initial {initial}",
            )
        )

    return report
