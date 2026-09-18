"""Layers L1-L4: parse, normalize, and repair raw model output.

The model's structured output is treated as untrusted input until it has passed
through this module. Repairs are deliberately conservative:

* values that are merely *sloppy* are fixed (unsorted or duplicate hours, a
  factor of ``1.2``, a reserve above battery capacity, ``"0.2"``-style strings);
* values that are *structurally wrong* (unknown directive type, missing hours,
  non-numeric factor) invalidate the entry so the caller can re-ask the model or
  hand the note to the rule-based interpreter.

Rejecting rather than guessing matters because a silently invented constraint
would corrupt the optimization and is explicitly forbidden by the spec.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Sequence

from app.schemas import BatterySpec, DirectiveInterpretation, DirectiveType

HORIZON = 24

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_ENTRY_KEYS = (
    "directive_interpretation",
    "directives",
    "interpretations",
    "results",
    "entries",
    "notes",
)


# ---------------------------------------------------------------------------
# L1 - transport / parse
# ---------------------------------------------------------------------------


def _first_balanced(text: str) -> str | None:
    """Return the first balanced ``{...}`` or ``[...]`` block in ``text``."""
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        in_string = False
        escaped = False
        for position in range(start, len(text)):
            char = text[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opener:
                depth += 1
            elif char == closer:
                depth -= 1
                if depth == 0:
                    candidate = text[start : position + 1]
                    try:
                        json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    return candidate
    return None


def extract_json_payload(text: str) -> Any:
    """Extract JSON from model output, tolerating fences and surrounding prose."""
    if text is None:
        raise ValueError("model returned no content")
    stripped = text.strip()
    if not stripped:
        raise ValueError("model returned empty content")

    fence = _FENCE_RE.search(stripped)
    if fence:
        stripped = fence.group(1).strip()

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    block = _first_balanced(stripped)
    if block is None:
        raise ValueError("no JSON object or array found in model output")
    return json.loads(block)


def extract_entries(payload: Any) -> list[dict[str, Any]]:
    """Locate the list of interpretation entries inside a decoded payload."""
    if isinstance(payload, dict):
        for key in _ENTRY_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if "note_index" in payload:
            return [payload]
        for value in payload.values():
            if isinstance(value, list) and any(
                isinstance(item, dict) and "note_index" in item for item in value
            ):
                return [item for item in value if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


# ---------------------------------------------------------------------------
# L3 - coercion helpers
# ---------------------------------------------------------------------------


def _finite_float(value: Any) -> float | None:
    """Coerce to a finite float, or ``None`` when impossible."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


def _coerce_bool(value: Any) -> bool | None:
    """Coerce to a bool, or ``None`` when ambiguous."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "y", "1"):
            return True
        if lowered in ("false", "no", "n", "0"):
            return False
    return None


def _coerce_index(value: Any) -> int | None:
    """Coerce a ``note_index`` to a non-negative int."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and float(value).is_integer():
        return int(value) if value >= 0 else None
    if isinstance(value, str):
        try:
            parsed = int(value.strip())
        except ValueError:
            return None
        return parsed if parsed >= 0 else None
    return None


def normalize_hours(value: Any) -> list[int] | None:
    """Return ascending unique in-range hours, or ``None`` when unusable.

    Sorting and de-duplication are safe repairs; an out-of-range or non-integer
    value makes the whole list unusable.
    """
    if not isinstance(value, (list, tuple)) or not value:
        return None
    collected: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            return None
        if isinstance(item, int):
            hour = item
        elif isinstance(item, float) and float(item).is_integer():
            hour = int(item)
        elif isinstance(item, str):
            try:
                hour = int(item.strip())
            except ValueError:
                return None
        else:
            return None
        if not 0 <= hour < HORIZON:
            return None
        collected.add(hour)
    if not collected:
        return None
    return sorted(collected)


# ---------------------------------------------------------------------------
# L2/L4 - entry normalization
# ---------------------------------------------------------------------------


def _default_explanation(directive_type: DirectiveType) -> str:
    return {
        DirectiveType.NO_OP: "No effect on the 24-hour energy schedule.",
        DirectiveType.SOLAR_REDUCTION: "Usable solar is reduced for the listed hours.",
        DirectiveType.MINIMUM_BATTERY_RESERVE: "A higher battery reserve applies for the listed hours.",
        DirectiveType.NO_CHARGE_WINDOW: "Battery charging is unavailable for the listed hours.",
        DirectiveType.NO_DISCHARGE_WINDOW: "Battery discharging is unavailable for the listed hours.",
        DirectiveType.MAX_GRID_WINDOW: "Grid import is capped for the listed hours.",
    }[directive_type]


def normalize_entry(
    raw: dict[str, Any], note_index: int, battery: BatterySpec
) -> tuple[DirectiveInterpretation | None, list[str]]:
    """Validate and repair one raw entry. ``None`` means "unusable"."""
    problems: list[str] = []

    raw_type = raw.get("directive_type")
    if not isinstance(raw_type, str):
        return None, [f"note {note_index}: directive_type is missing or not a string"]
    try:
        directive_type = DirectiveType(raw_type.strip().lower())
    except ValueError:
        return None, [
            f"note {note_index}: unsupported directive_type {raw_type!r} (invention rejected)"
        ]

    explanation = str(raw.get("explanation") or "").strip()

    # no_op is the only directive permitted with applies=false and null adjustment.
    if directive_type is DirectiveType.NO_OP:
        return (
            DirectiveInterpretation(
                note_index=note_index,
                applies=False,
                directive_type=directive_type,
                structured_adjustment=None,
                explanation=explanation or _default_explanation(directive_type),
            ),
            problems,
        )

    applies = _coerce_bool(raw.get("applies"))
    if applies is False:
        problems.append(
            f"note {note_index}: applies=false is only legal for no_op; coerced to no_op"
        )
        return (
            DirectiveInterpretation(
                note_index=note_index,
                applies=False,
                directive_type=DirectiveType.NO_OP,
                structured_adjustment=None,
                explanation=explanation or _default_explanation(DirectiveType.NO_OP),
            ),
            problems,
        )

    adjustment = raw.get("structured_adjustment")
    if not isinstance(adjustment, dict):
        return None, [f"note {note_index}: structured_adjustment must be an object"]

    hours = normalize_hours(adjustment.get("hours"))
    if hours is None:
        return None, [f"note {note_index}: hours must be unique integers 0-23"]

    if directive_type is DirectiveType.SOLAR_REDUCTION:
        factor = _finite_float(adjustment.get("factor"))
        if factor is None:
            return None, [f"note {note_index}: solar_reduction requires a numeric factor"]
        clamped = min(max(factor, 0.0), 1.0)
        if clamped != factor:
            problems.append(f"note {note_index}: factor {factor} clamped to {clamped}")
        repaired: dict[str, Any] = {"hours": hours, "factor": round(clamped, 6)}

    elif directive_type is DirectiveType.MINIMUM_BATTERY_RESERVE:
        reserve = _finite_float(adjustment.get("minimum_energy_kwh"))
        if reserve is None:
            return None, [f"note {note_index}: minimum_battery_reserve requires minimum_energy_kwh"]
        if reserve < 0:
            problems.append(f"note {note_index}: negative reserve raised to 0")
            reserve = 0.0
        if reserve > battery.capacity_kwh:
            problems.append(
                f"note {note_index}: reserve {reserve} exceeds capacity; clamped to "
                f"{battery.capacity_kwh}"
            )
            reserve = battery.capacity_kwh
        repaired = {"hours": hours, "minimum_energy_kwh": round(reserve, 6)}

    elif directive_type is DirectiveType.MAX_GRID_WINDOW:
        cap = _finite_float(adjustment.get("max_grid_kwh"))
        if cap is None:
            return None, [f"note {note_index}: max_grid_window requires max_grid_kwh"]
        if cap < 0:
            problems.append(f"note {note_index}: negative grid cap raised to 0")
            cap = 0.0
        repaired = {"hours": hours, "max_grid_kwh": round(cap, 6)}

    else:  # no_charge_window / no_discharge_window
        repaired = {"hours": hours}

    return (
        DirectiveInterpretation(
            note_index=note_index,
            applies=True,
            directive_type=directive_type,
            structured_adjustment=repaired,
            explanation=explanation or _default_explanation(directive_type),
        ),
        problems,
    )


def normalize_interpretation(
    payload: Any, notes: Sequence[str], battery: BatterySpec
) -> tuple[list[DirectiveInterpretation | None], list[str]]:
    """Normalize a decoded payload into one slot per operator note.

    Returns a list whose length equals the note count. ``None`` entries mark
    notes the model failed to interpret correctly; the caller fills them from
    the deterministic rule-based interpreter.
    """
    count = len(notes)
    problems: list[str] = []
    by_index: dict[int, DirectiveInterpretation] = {}

    for raw in extract_entries(payload):
        index = _coerce_index(raw.get("note_index"))
        if index is None or index >= count:
            problems.append(f"entry with unusable note_index {raw.get('note_index')!r} ignored")
            continue
        if index in by_index:
            problems.append(f"duplicate note_index {index} ignored")
            continue
        entry, entry_problems = normalize_entry(raw, index, battery)
        problems.extend(entry_problems)
        if entry is not None:
            by_index[index] = entry

    missing = [index for index in range(count) if index not in by_index]
    if missing:
        problems.append(f"no usable interpretation for notes {missing}")

    return [by_index.get(index) for index in range(count)], problems
