"""Compile validated directives into deterministic optimization parameters.

This module is the *only* bridge between natural-language interpretation and
mathematics. The optimizer never inspects English text.

Merging semantics for repeated directive types (hidden cases may repeat a type
within one scenario):

* ``solar_reduction``        - factors multiply (successive reductions compound)
* ``minimum_battery_reserve`` - the strictest (largest) reserve wins
* ``max_grid_window``        - the strictest (smallest) cap wins
* ``no_charge_window``       - union of the blocked hours
* ``no_discharge_window``    - union of the blocked hours
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

HORIZON = 24


@dataclass
class DirectiveConstraints:
    """Per-hour modifiers derived from validated directives."""

    solar_factor: list[float] = field(default_factory=lambda: [1.0] * HORIZON)
    reserve_kwh: list[float | None] = field(default_factory=lambda: [None] * HORIZON)
    charge_allowed: list[bool] = field(default_factory=lambda: [True] * HORIZON)
    discharge_allowed: list[bool] = field(default_factory=lambda: [True] * HORIZON)
    grid_cap_kwh: list[float | None] = field(default_factory=lambda: [None] * HORIZON)
    applied: list[tuple[int, str, dict[str, Any]]] = field(default_factory=list)

    def effective_solar(self, base_solar: list[float]) -> list[float]:
        """Apply ``effective_solar[h] = base_solar[h] * factor[h]``."""
        return [base_solar[h] * self.solar_factor[h] for h in range(HORIZON)]

    def has_any(self) -> bool:
        return bool(self.applied)


def _entries(interpretation: Iterable[Any]) -> list[Any]:
    """Return only entries that actually modify the model, in note order."""
    result = []
    for entry in interpretation:
        directive_type = getattr(entry.directive_type, "value", entry.directive_type)
        if directive_type == "no_op" or not entry.applies:
            continue
        result.append(entry)
    return result


def compile_constraints(interpretation: Iterable[Any]) -> DirectiveConstraints:
    """Turn validated interpretation entries into per-hour parameters."""
    constraints = DirectiveConstraints()

    for entry in _entries(interpretation):
        directive_type = getattr(entry.directive_type, "value", entry.directive_type)
        adjustment = entry.structured_adjustment or {}
        hours = [int(h) for h in adjustment.get("hours", []) if 0 <= int(h) < HORIZON]
        if not hours:
            continue

        if directive_type == "solar_reduction":
            factor = float(adjustment["factor"])
            for hour in hours:
                constraints.solar_factor[hour] *= factor
            constraints.applied.append(
                (entry.note_index, directive_type, {"hours": sorted(hours), "factor": factor})
            )

        elif directive_type == "minimum_battery_reserve":
            reserve = float(adjustment["minimum_energy_kwh"])
            for hour in hours:
                current = constraints.reserve_kwh[hour]
                constraints.reserve_kwh[hour] = reserve if current is None else max(current, reserve)
            constraints.applied.append(
                (entry.note_index, directive_type, {"hours": sorted(hours), "minimum_energy_kwh": reserve})
            )

        elif directive_type == "no_charge_window":
            for hour in hours:
                constraints.charge_allowed[hour] = False
            constraints.applied.append((entry.note_index, directive_type, {"hours": sorted(hours)}))

        elif directive_type == "no_discharge_window":
            for hour in hours:
                constraints.discharge_allowed[hour] = False
            constraints.applied.append((entry.note_index, directive_type, {"hours": sorted(hours)}))

        elif directive_type == "max_grid_window":
            cap = float(adjustment["max_grid_kwh"])
            for hour in hours:
                current = constraints.grid_cap_kwh[hour]
                constraints.grid_cap_kwh[hour] = cap if current is None else min(current, cap)
            constraints.applied.append(
                (entry.note_index, directive_type, {"hours": sorted(hours), "max_grid_kwh": cap})
            )

    return constraints
