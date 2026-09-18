"""Guardrail behaviour: parsing, repair, and rejection of untrusted model output."""

from __future__ import annotations

import json

import pytest

from app.guardrails.normalize import (
    extract_entries,
    extract_json_payload,
    normalize_entry,
    normalize_hours,
    normalize_interpretation,
)
from app.guardrails.validate import check_interpretation
from app.schemas import BatterySpec, DirectiveType

BATTERY = BatterySpec(
    capacity_kwh=200.0,
    initial_energy_kwh=100.0,
    minimum_energy_kwh=40.0,
    max_charge_kwh_per_hour=50.0,
    max_discharge_kwh_per_hour=50.0,
)
NOTES = ["first note", "second note", "third note"]


# ---------------------------------------------------------------------------
# L1 - parsing
# ---------------------------------------------------------------------------


def test_extract_json_handles_plain_payload() -> None:
    payload = {"directive_interpretation": [{"note_index": 0}]}
    assert extract_json_payload(json.dumps(payload)) == payload


def test_extract_json_handles_markdown_fence() -> None:
    raw = 'Here you go:\n```json\n{"a": 1}\n```\n'
    assert extract_json_payload(raw) == {"a": 1}


def test_extract_json_handles_surrounding_prose() -> None:
    raw = 'Sure! {"directive_interpretation": []} Hope that helps.'
    assert extract_json_payload(raw) == {"directive_interpretation": []}


def test_extract_json_handles_bare_array() -> None:
    assert extract_json_payload('[{"note_index": 0}]') == [{"note_index": 0}]


def test_extract_json_ignores_braces_inside_strings() -> None:
    raw = '{"explanation": "a } brace inside a string", "note_index": 0}'
    assert extract_json_payload(raw)["note_index"] == 0


def test_extract_json_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        extract_json_payload("I cannot help with that.")


def test_extract_json_rejects_empty() -> None:
    with pytest.raises(ValueError):
        extract_json_payload("   ")


def test_extract_entries_reads_alternate_keys() -> None:
    payload = {"directives": [{"note_index": 0, "directive_type": "no_op"}]}
    assert len(extract_entries(payload)) == 1


# ---------------------------------------------------------------------------
# L3 - hours normalization
# ---------------------------------------------------------------------------


def test_hours_are_sorted_and_deduplicated() -> None:
    assert normalize_hours([14, 13, 14, 12]) == [12, 13, 14]


def test_hours_accept_numeric_strings_and_float_ints() -> None:
    assert normalize_hours(["3", 4.0]) == [3, 4]


@pytest.mark.parametrize("value", [[], [24], [-1], ["noon"], [True], None, "13"])
def test_hours_reject_unusable_values(value) -> None:
    assert normalize_hours(value) is None


# ---------------------------------------------------------------------------
# L2/L4 - entry normalization
# ---------------------------------------------------------------------------


def _entry(**overrides) -> dict:
    base = {
        "note_index": 0,
        "applies": True,
        "directive_type": "solar_reduction",
        "structured_adjustment": {"hours": [13, 14], "factor": 0.2},
        "explanation": "test",
    }
    base.update(overrides)
    return base


def test_valid_entry_passes_through() -> None:
    entry, problems = normalize_entry(_entry(), 0, BATTERY)
    assert entry is not None
    assert entry.directive_type is DirectiveType.SOLAR_REDUCTION
    assert entry.structured_adjustment == {"hours": [13, 14], "factor": 0.2}
    assert problems == []


def test_factor_above_one_is_clamped() -> None:
    entry, problems = normalize_entry(_entry(structured_adjustment={"hours": [1], "factor": 1.5}), 0, BATTERY)
    assert entry is not None
    assert entry.structured_adjustment["factor"] == 1.0
    assert any("clamped" in problem for problem in problems)


def test_factor_accepts_numeric_string() -> None:
    entry, _ = normalize_entry(_entry(structured_adjustment={"hours": [1], "factor": "0.25"}), 0, BATTERY)
    assert entry is not None
    assert entry.structured_adjustment["factor"] == 0.25


def test_reserve_above_capacity_is_clamped() -> None:
    raw = _entry(
        directive_type="minimum_battery_reserve",
        structured_adjustment={"hours": [18], "minimum_energy_kwh": 900},
    )
    entry, problems = normalize_entry(raw, 0, BATTERY)
    assert entry is not None
    assert entry.structured_adjustment["minimum_energy_kwh"] == BATTERY.capacity_kwh
    assert any("capacity" in problem for problem in problems)


def test_negative_grid_cap_is_raised_to_zero() -> None:
    raw = _entry(
        directive_type="max_grid_window",
        structured_adjustment={"hours": [18], "max_grid_kwh": -5},
    )
    entry, _ = normalize_entry(raw, 0, BATTERY)
    assert entry is not None
    assert entry.structured_adjustment["max_grid_kwh"] == 0.0


def test_no_op_forces_false_and_null() -> None:
    raw = _entry(directive_type="no_op", applies=True, structured_adjustment={"hours": [1]})
    entry, _ = normalize_entry(raw, 2, BATTERY)
    assert entry is not None
    assert entry.applies is False
    assert entry.structured_adjustment is None
    assert entry.directive_type is DirectiveType.NO_OP


def test_applies_false_with_real_type_is_coerced_to_no_op() -> None:
    raw = _entry(applies=False)
    entry, problems = normalize_entry(raw, 0, BATTERY)
    assert entry is not None
    assert entry.directive_type is DirectiveType.NO_OP
    assert entry.applies is False
    assert any("applies=false" in problem for problem in problems)


def test_unknown_directive_type_is_rejected_not_invented() -> None:
    raw = _entry(directive_type="battery_swap_window")
    entry, problems = normalize_entry(raw, 0, BATTERY)
    assert entry is None
    assert any("unsupported directive_type" in problem for problem in problems)


def test_missing_factor_is_rejected() -> None:
    entry, problems = normalize_entry(_entry(structured_adjustment={"hours": [1]}), 0, BATTERY)
    assert entry is None
    assert any("factor" in problem for problem in problems)


def test_missing_adjustment_is_rejected() -> None:
    entry, _ = normalize_entry(_entry(structured_adjustment=None), 0, BATTERY)
    assert entry is None


def test_out_of_range_hour_invalidates_entry() -> None:
    entry, _ = normalize_entry(_entry(structured_adjustment={"hours": [25], "factor": 0.5}), 0, BATTERY)
    assert entry is None


def test_no_charge_window_ignores_extra_keys() -> None:
    raw = _entry(
        directive_type="no_charge_window",
        structured_adjustment={"hours": [14, 15], "factor": 0.9},
    )
    entry, _ = normalize_entry(raw, 0, BATTERY)
    assert entry is not None
    assert entry.structured_adjustment == {"hours": [14, 15]}


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_missing_notes_are_reported_as_gaps() -> None:
    payload = {"directive_interpretation": [_entry(note_index=0)]}
    slots, problems = normalize_interpretation(payload, NOTES, BATTERY)
    assert len(slots) == 3
    assert slots[0] is not None
    assert slots[1] is None and slots[2] is None
    assert any("no usable interpretation" in problem for problem in problems)


def test_out_of_range_and_duplicate_indices_are_dropped() -> None:
    payload = {
        "directive_interpretation": [
            _entry(note_index=9),
            _entry(note_index=1),
            _entry(note_index=1),
        ]
    }
    slots, problems = normalize_interpretation(payload, NOTES, BATTERY)
    assert slots[1] is not None
    assert slots[0] is None and slots[2] is None
    assert any("unusable note_index" in problem for problem in problems)
    assert any("duplicate note_index" in problem for problem in problems)


def test_full_coverage_returns_no_gap_problems() -> None:
    payload = {
        "directive_interpretation": [
            _entry(note_index=0),
            _entry(note_index=1, directive_type="no_op", applies=False, structured_adjustment=None),
            _entry(note_index=2, directive_type="no_charge_window", structured_adjustment={"hours": [2]}),
        ]
    }
    slots, problems = normalize_interpretation(payload, NOTES, BATTERY)
    assert all(slot is not None for slot in slots)
    assert not any("no usable interpretation" in problem for problem in problems)


# ---------------------------------------------------------------------------
# policy assertions
# ---------------------------------------------------------------------------


def test_policy_check_accepts_a_clean_list() -> None:
    payload = {
        "directive_interpretation": [
            _entry(note_index=0),
            _entry(note_index=1, directive_type="no_op", applies=False, structured_adjustment=None),
        ]
    }
    slots, _ = normalize_interpretation(payload, ["a", "b"], BATTERY)
    assert check_interpretation(slots, 2, BATTERY) == []


def test_policy_check_flags_unsorted_hours() -> None:
    from app.schemas import DirectiveInterpretation

    bad = DirectiveInterpretation(
        note_index=0,
        applies=True,
        directive_type=DirectiveType.NO_CHARGE_WINDOW,
        structured_adjustment={"hours": [15, 14]},
        explanation="unsorted",
    )
    problems = check_interpretation([bad], 1, BATTERY)
    assert any("ascending" in problem for problem in problems)


def test_policy_check_flags_missing_entry() -> None:
    problems = check_interpretation([None], 1, BATTERY)
    assert problems
