"""Layer L4: final policy assertions on a normalized interpretation list.

These checks run after normalization and are also used directly by the test
suite. A problem here means the interpretation must not be trusted, so the
pipeline records it and falls back rather than sending bad constraints to the
optimizer.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.schemas import DirectiveType, BatterySpec

HORIZON = 24
_NO_OP_ADJUSTMENT_KEYS = {"hours", "factor", "minimum_energy_kwh", "max_grid_kwh"}


def check_interpretation(
    entries: Sequence[Any], note_count: int, battery: BatterySpec
) -> list[str]:
    """Return a list of policy problems; empty means the list is trustworthy."""
    problems: list[str] = []

    if len(entries) != note_count:
        problems.append(f"expected {note_count} entries, found {len(entries)}")

    seen: set[int] = set()
    for position, entry in enumerate(entries):
        if entry is None:
            problems.append(f"position {position} has no interpretation")
            continue

        index = entry.note_index
        if index in seen:
            problems.append(f"note_index {index} appears more than once")
        seen.add(index)

        directive_type = getattr(entry.directive_type, "value", entry.directive_type)
        applies = bool(entry.applies)
        adjustment = entry.structured_adjustment

        if directive_type == DirectiveType.NO_OP.value:
            if applies:
                problems.append(f"note {index}: no_op must use applies=false")
            if adjustment is not None:
                problems.append(f"note {index}: no_op must use a null structured_adjustment")
            continue

        if not applies:
            problems.append(f"note {index}: {directive_type} must use applies=true")
        if not isinstance(adjustment, dict):
            problems.append(f"note {index}: {directive_type} requires a structured_adjustment")
            continue

        hours = adjustment.get("hours")
        if not isinstance(hours, list) or not hours:
            problems.append(f"note {index}: hours must be a non-empty list")
        else:
            if any(not isinstance(hour, int) or isinstance(hour, bool) for hour in hours):
                problems.append(f"note {index}: hours must be integers")
            elif hours != sorted(set(hours)):
                problems.append(f"note {index}: hours must be ascending and unique")
            elif any(not 0 <= hour < HORIZON for hour in hours):
                problems.append(f"note {index}: hours must be within 0-23")

        if directive_type == DirectiveType.SOLAR_REDUCTION.value:
            factor = adjustment.get("factor")
            if not isinstance(factor, (int, float)) or isinstance(factor, bool):
                problems.append(f"note {index}: solar_reduction requires a numeric factor")
            elif not 0.0 <= float(factor) <= 1.0:
                problems.append(f"note {index}: factor must lie within [0, 1]")
            extra = set(adjustment) - {"hours", "factor"}
            if extra:
                problems.append(f"note {index}: unexpected keys {sorted(extra)}")

        elif directive_type == DirectiveType.MINIMUM_BATTERY_RESERVE.value:
            reserve = adjustment.get("minimum_energy_kwh")
            if not isinstance(reserve, (int, float)) or isinstance(reserve, bool):
                problems.append(f"note {index}: minimum_battery_reserve requires a numeric value")
            elif reserve < 0:
                problems.append(f"note {index}: reserve must be non-negative")
            elif float(reserve) > battery.capacity_kwh:
                problems.append(f"note {index}: reserve must not exceed battery capacity")
            extra = set(adjustment) - {"hours", "minimum_energy_kwh"}
            if extra:
                problems.append(f"note {index}: unexpected keys {sorted(extra)}")

        elif directive_type == DirectiveType.MAX_GRID_WINDOW.value:
            cap = adjustment.get("max_grid_kwh")
            if not isinstance(cap, (int, float)) or isinstance(cap, bool):
                problems.append(f"note {index}: max_grid_window requires a numeric cap")
            elif cap < 0:
                problems.append(f"note {index}: grid cap must be non-negative")
            extra = set(adjustment) - {"hours", "max_grid_kwh"}
            if extra:
                problems.append(f"note {index}: unexpected keys {sorted(extra)}")

        elif directive_type in (
            DirectiveType.NO_CHARGE_WINDOW.value,
            DirectiveType.NO_DISCHARGE_WINDOW.value,
        ):
            extra = set(adjustment) - {"hours"}
            if extra:
                problems.append(f"note {index}: unexpected keys {sorted(extra)}")

        else:
            problems.append(f"note {index}: unsupported directive_type {directive_type!r}")

    return problems
