"""Rule-based operator-note interpreter.

Role in the architecture
------------------------
This is **not** the primary interpreter and must never be the sole one - the
challenge requires a language model in the operator-note interpretation path.
It exists as the last stage of the recovery ladder:

    LLM  ->  guardrails  ->  retry LLM  ->  THIS  ->  no_op fallback

It also gives the test suite a deterministic, offline interpretation path so the
optimizer and replay verifier can be exercised without a provider key.

Conventions implemented (matching the canonical Problem Statement)
-----------------------------------------------------------------
* Time windows are **start-inclusive and end-exclusive**, in whole hours:
  "1 PM to 3 PM" -> ``[13, 14]``.
* ``factor`` is the *usable fraction remaining*:
  - "X% of the forecast"      -> ``X/100``
  - "drop to X%"              -> ``X/100``
  - "X% reduction" / "by X%"  -> ``1 - X/100``
  - "about half"              -> ``0.5``
* Relative reserves are resolved against ``capacity_kwh``:
  "50% of the battery capacity" -> ``0.5 * capacity``.
"""

from __future__ import annotations

import re
from typing import Iterable

from app.schemas import DirectiveInterpretation, DirectiveType

HORIZON = 24

# ---------------------------------------------------------------------------
# Vocabulary
#
# Word-boundary patterns rather than raw substring checks: "import" must not
# fire on "important", and "charge" must not fire on "discharge".
# ---------------------------------------------------------------------------

_SOLAR_RE = re.compile(
    r"\b(solar|photovoltaic|pv|panels?|rooftop|irradiance|inverter|sunlight)\b",
    re.IGNORECASE,
)
_GRID_RE = re.compile(
    r"\b(grid|import|imports|intake|draw|feeder|transformer|substation|utility)\b",
    re.IGNORECASE,
)
_CHARGE_RE = re.compile(r"\b(charg\w*|recharg\w*)\b", re.IGNORECASE)
_DISCHARGE_RE = re.compile(r"\bdischarg\w*\b", re.IGNORECASE)
_RESERVE_RE = re.compile(
    r"\b(reserve|reserves|remaining|remain|kept|keep|stored|state of charge|"
    r"minimum|level|left)\b",
    re.IGNORECASE,
)

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_FRACTION_WORDS = {
    "half": 0.5,
    "third": 1.0 / 3.0,
    "quarter": 0.25,
    "fourth": 0.25,
    "fifth": 0.2,
    "sixth": 1.0 / 6.0,
    "seventh": 1.0 / 7.0,
    "eighth": 0.125,
    "ninth": 1.0 / 9.0,
    "tenth": 0.1,
    "twentieth": 0.05,
}

# ---------------------------------------------------------------------------
# Time parsing
# ---------------------------------------------------------------------------

_WORD_TIME = "|".join(_NUMBER_WORDS)
_CLOCK_RANGE_RE = re.compile(
    r"\b(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*(a\.?m\.?|p\.?m\.?)",
    re.IGNORECASE,
)
_TIME_TOKEN_RE = re.compile(
    r"(?P<hms>\b(?P<h1>\d{1,2}):(?P<m1>\d{2})\s*(?P<ap1>a\.?m\.?|p\.?m\.?)?)"
    r"|(?P<hm>\b(?P<h2>\d{1,2})\s*(?P<ap2>a\.?m\.?|p\.?m\.?))"
    r"|(?P<wordtime>\b(?P<w1>noon|midnight|midday)\b)"
    r"|(?P<numtime>\b(?P<w2>" + _WORD_TIME + r")\s*(?P<ap3>a\.?m\.?|p\.?m\.?))",
    re.IGNORECASE,
)
_HOUR_MENTION_RE = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*hour", re.IGNORECASE)


def _to_24h(hour: int, meridiem: str | None) -> int | None:
    """Convert a clock reading to 0-23 using the 12-hour convention."""
    if meridiem is None:
        return hour if 0 <= hour < 24 else None
    marker = meridiem.lower().replace(".", "")
    if not 1 <= hour <= 12:
        return None
    if marker == "am":
        return 0 if hour == 12 else hour
    return 12 if hour == 12 else hour + 12


def _find_times(text: str) -> list[int]:
    """Return clock hours in order of appearance."""
    hours: list[int] = []
    consumed: list[tuple[int, int]] = []

    # "1-3 PM" style ranges: both endpoints share the trailing meridiem.
    for match in _CLOCK_RANGE_RE.finditer(text):
        first = _to_24h(int(match.group(1)), match.group(3))
        second = _to_24h(int(match.group(2)), match.group(3))
        if first is None or second is None:
            continue
        hours.extend([first, second])
        consumed.append(match.span())

    def _overlaps(span: tuple[int, int]) -> bool:
        return any(span[0] < end and start < span[1] for start, end in consumed)

    for match in _TIME_TOKEN_RE.finditer(text):
        if _overlaps(match.span()):
            continue
        if match.group("hms"):
            value = _to_24h(int(match.group("h1")), match.group("ap1"))
        elif match.group("hm"):
            value = _to_24h(int(match.group("h2")), match.group("ap2"))
        elif match.group("wordtime"):
            token = match.group("w1").lower()
            value = {"noon": 12, "midday": 12, "midnight": 0}[token]
        else:
            token = match.group("w2").lower()
            value = _to_24h(_NUMBER_WORDS[token], match.group("ap3"))
        if value is not None:
            hours.append(value)
            consumed.append(match.span())

    return hours


def _window(text: str) -> list[int] | None:
    """Derive the affected hours from the note's time expressions."""
    times = _find_times(text)
    if len(times) >= 2:
        start, end = times[0], times[1]
        if end > start:
            return list(range(start, end))
        if end == start:
            return [start]
        # Wraps past midnight: split the window.
        return list(range(start, HORIZON)) + list(range(0, end))
    if len(times) == 1:
        hour = times[0]
        return [hour] if hour == HORIZON - 1 else [hour, hour + 1]

    hours = [int(m.group(1)) for m in _HOUR_MENTION_RE.finditer(text)]
    if hours:
        return sorted({h for h in hours if 0 <= h < HORIZON})
    return None


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------

_KWH_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:kwh|kilowatt[- ]?hours?|units?)", re.IGNORECASE)
_PCT_OF_CAPACITY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of|off)\s*(?:the\s+)?"
    r"(?:battery\s+)?(?:capacity|storage|pack)\b",
    re.IGNORECASE,
)
_PCT_OF_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:of|off)\b", re.IGNORECASE)
_PCT_REDUCTION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*"
    r"(?:reduction|less|lower|loss|drop|decrease|reduced)\b",
    re.IGNORECASE,
)
_REDUCE_BY_RE = re.compile(
    r"\b(?:reduc\w*|drop\w*|decreas\w*|los\w*|lower\w*|down|fall\w*|declin\w*)"
    r"\s*(?:by|about|roughly|approximately)\s*(\d+(?:\.\d+)?)\s*(?:%|percent)",
    re.IGNORECASE,
)
_DROP_TO_RE = re.compile(
    r"\b(?:drop\w*|fall\w*|declin\w*|decreas\w*|down)\s*(?:to|to about|to roughly|to around|at)"
    r"\s*(\d+(?:\.\d+)?)\s*(?:%|percent)",
    re.IGNORECASE,
)
_FRACTION_RE = re.compile(
    r"\b(?:(?P<count>[a-z]+)[-\s])?(?P<base>" + "|".join(_FRACTION_WORDS) + r")\b",
    re.IGNORECASE,
)


def _fraction_value(text: str) -> float | None:
    """Resolve wording like "half", "about half", "one-fifth", "two-thirds"."""
    for match in _FRACTION_RE.finditer(text):
        base = _FRACTION_WORDS[match.group("base").lower()]
        count_token = (match.group("count") or "").lower()
        if not count_token or count_token == "a":
            multiplier = 1
        elif count_token in _NUMBER_WORDS:
            multiplier = _NUMBER_WORDS[count_token]
        else:
            multiplier = 1
        return min(max(multiplier * base, 0.0), 1.0)
    return None


def solar_factor(text: str) -> float | None:
    """Extract the *usable fraction remaining* from a solar note."""
    match = _DROP_TO_RE.search(text)
    if match:
        return min(max(float(match.group(1)) / 100.0, 0.0), 1.0)

    match = _PCT_REDUCTION_RE.search(text)
    if match:
        return min(max(1.0 - float(match.group(1)) / 100.0, 0.0), 1.0)

    match = _REDUCE_BY_RE.search(text)
    if match:
        return min(max(1.0 - float(match.group(1)) / 100.0, 0.0), 1.0)

    match = _PCT_OF_RE.search(text)
    if match:
        return min(max(float(match.group(1)) / 100.0, 0.0), 1.0)

    return _fraction_value(text)


def _kwh_value(text: str) -> float | None:
    match = _KWH_RE.search(text)
    if match:
        return float(match.group(1))
    return None


def _reserve_value(text: str, capacity_kwh: float) -> float | None:
    match = _PCT_OF_CAPACITY_RE.search(text)
    if match:
        return min(max(float(match.group(1)) / 100.0 * capacity_kwh, 0.0), capacity_kwh)
    value = _kwh_value(text)
    if value is not None:
        return min(max(value, 0.0), capacity_kwh)
    return None


def _grid_cap_value(text: str) -> float | None:
    value = _kwh_value(text)
    if value is None:
        return None
    return max(value, 0.0)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _hit(pattern: re.Pattern[str], text: str) -> bool:
    return pattern.search(text) is not None


def classify(text: str, capacity_kwh: float) -> DirectiveInterpretation:
    """Interpret one note deterministically, falling back to ``no_op``."""
    lowered = text.lower()
    hours = _window(lowered)

    def _no_op(reason: str) -> DirectiveInterpretation:
        return DirectiveInterpretation(
            note_index=0,
            applies=False,
            directive_type=DirectiveType.NO_OP,
            structured_adjustment=None,
            explanation=f"Rule-based fallback: {reason}",
        )

    def _build(directive_type: DirectiveType, adjustment: dict, reason: str) -> DirectiveInterpretation:
        return DirectiveInterpretation(
            note_index=0,
            applies=True,
            directive_type=directive_type,
            structured_adjustment=adjustment,
            explanation=f"Rule-based fallback: {reason}",
        )

    if hours is None:
        return _no_op("no actionable time window found in the note")

    # Grid caps are checked before solar/reserve: an import limit is explicit.
    if _hit(_GRID_RE, lowered):
        cap = _grid_cap_value(lowered)
        if cap is not None:
            return _build(
                DirectiveType.MAX_GRID_WINDOW,
                {"hours": hours, "max_grid_kwh": round(cap, 6)},
                f"grid import capped at {cap} kWh for hours {hours}",
            )

    if _hit(_SOLAR_RE, lowered):
        factor = solar_factor(lowered)
        if factor is not None:
            return _build(
                DirectiveType.SOLAR_REDUCTION,
                {"hours": hours, "factor": round(factor, 6)},
                f"usable solar limited to a factor of {round(factor, 6)} for hours {hours}",
            )

    if _hit(_RESERVE_RE, lowered):
        value = _reserve_value(lowered, capacity_kwh)
        if value is not None:
            return _build(
                DirectiveType.MINIMUM_BATTERY_RESERVE,
                {"hours": hours, "minimum_energy_kwh": round(value, 6)},
                f"battery reserve raised to {round(value, 6)} kWh for hours {hours}",
            )

    # Discharge is checked before charge: "discharge" never contains a
    # word-boundary "charge", but keeping the order explicit avoids surprises.
    if _hit(_DISCHARGE_RE, lowered):
        return _build(
            DirectiveType.NO_DISCHARGE_WINDOW,
            {"hours": hours},
            f"battery discharging unavailable for hours {hours}",
        )

    if _hit(_CHARGE_RE, lowered):
        return _build(
            DirectiveType.NO_CHARGE_WINDOW,
            {"hours": hours},
            f"battery charging unavailable for hours {hours}",
        )

    return _no_op("the note does not reference a supported energy operating condition")


def interpret_note(text: str, capacity_kwh: float = 0.0) -> DirectiveInterpretation:
    """Interpret a single note. ``note_index`` is always 0 and set by the caller."""
    entry = classify(text, capacity_kwh)
    return entry


def interpret_notes(
    notes: Iterable[str], capacity_kwh: float = 0.0
) -> list[DirectiveInterpretation]:
    """Interpret every note, filling ``note_index`` in order."""
    results: list[DirectiveInterpretation] = []
    for index, note in enumerate(notes):
        entry = classify(str(note), capacity_kwh)
        results.append(entry.model_copy(update={"note_index": index}))
    return results
