#!/usr/bin/env python
"""50-case validation suite for a deployed GridWise service.

Runs a broad, logically-structured battery of cases against a live base URL and
scores each one the way the judge does:

1. HTTP status matches the contract (200 / 400 / 422)
2. response schema is complete and well-formed
3. `directive_interpretation` matches the case's ground truth
4. the returned `hourly_plan` **replays clean against the ground-truth directives**
   (this is the judge's real method - a correct parse with an unapplied directive fails)
5. reported cost is not worse than the optimum computed locally for the same
   scenario + ground truth, i.e. `min(1, optimal/team)` stays at 1.0

Usage:
    python tools/suite_50.py                                  # default deployed URL
    python tools/suite_50.py --url http://127.0.0.1:8000
    python tools/suite_50.py --category TIME                  # one group
    python tools/suite_50.py --only WIN-NOON,SR-PCT-OF
    python tools/suite_50.py --dump tools/suite_50.json       # export the case pack
    python tools/suite_50.py --verbose

Exit code is non-zero if any non-advisory case fails.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx  # noqa: E402

from app.optimization.constraints import compile_constraints  # noqa: E402
from app.optimization.replay import verify_plan  # noqa: E402
from app.optimization.solver import optimize  # noqa: E402
from app.schemas import DirectiveInterpretation, HourlyPlanEntry, ScenarioRequest  # noqa: E402

DEFAULT_URL = "https://bup-hackathon-00ea.onrender.com"
COST_TOLERANCE = 0.01
FACTOR_TOLERANCE = 0.02
VALUE_TOLERANCE = 0.5

REQUIRED_RESPONSE_KEYS = {
    "scenario_id",
    "directive_interpretation",
    "hourly_plan",
    "total_grid_kwh",
    "total_cost_bdt",
    "peak_grid_kwh",
    "plan_summary",
}
PLAN_FIELDS = (
    "hour",
    "grid_kwh",
    "solar_used_kwh",
    "battery_action",
    "battery_kwh",
    "battery_energy_after_kwh",
)

# ---------------------------------------------------------------------------
# scenario construction
# ---------------------------------------------------------------------------

_BASE_DEMAND = [
    95, 90, 88, 88, 92, 100, 115, 135, 155, 170, 180, 188,
    192, 188, 178, 172, 182, 198, 214, 228, 218, 190, 152, 120,
]
_BASE_SOLAR = [
    0, 0, 0, 0, 0, 4, 18, 45, 80, 120, 155, 175,
    188, 180, 158, 120, 68, 26, 4, 0, 0, 0, 0, 0,
]
_BASE_TARIFF = [
    6, 6, 5, 5, 5, 6, 8, 10, 12, 14, 16, 17,
    16, 15, 14, 15, 19, 24, 30, 34, 31, 21, 11, 8,
]


def campus_hours(
    *,
    demand_scale: float = 1.0,
    solar_scale: float = 1.0,
    solar_zero: bool = False,
    tariff_scale: float = 1.0,
    tariff_override: list[float] | None = None,
    demand_override: dict[int, float] | None = None,
) -> list[dict[str, Any]]:
    """A realistic 24-hour campus profile, optionally perturbed."""
    hours: list[dict[str, Any]] = []
    for hour in range(24):
        if solar_zero:
            solar = 0.0
        else:
            solar = round(_BASE_SOLAR[hour] * solar_scale, 2)
        if tariff_override is not None:
            tariff = round(tariff_override[hour], 2)
        else:
            tariff = round(_BASE_TARIFF[hour] * tariff_scale, 2)
        demand = round(_BASE_DEMAND[hour] * demand_scale, 2)
        if demand_override and hour in demand_override:
            demand = round(demand_override[hour], 2)
        hours.append(
            {
                "hour": hour,
                "demand_kwh": demand,
                "solar_kwh": solar,
                "tariff_bdt_per_kwh": tariff,
            }
        )
    return hours


def battery(
    capacity: float = 220.0,
    initial: float = 110.0,
    minimum: float = 40.0,
    max_charge: float = 55.0,
    max_discharge: float = 55.0,
) -> dict[str, float]:
    return {
        "capacity_kwh": capacity,
        "initial_energy_kwh": initial,
        "minimum_energy_kwh": minimum,
        "max_charge_kwh_per_hour": max_charge,
        "max_discharge_kwh_per_hour": max_discharge,
    }


# ---------------------------------------------------------------------------
# expectations
# ---------------------------------------------------------------------------

# A directive spec is ("type", {adjustment}) or None for a no_op note.
Spec = tuple[str, dict[str, Any]] | None


def truth(specs: list[Spec]) -> list[dict[str, Any]]:
    """Expand compact specs into ground-truth interpretation entries."""
    entries: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        if spec is None:
            entries.append(
                {
                    "note_index": index,
                    "applies": False,
                    "directive_type": "no_op",
                    "structured_adjustment": None,
                    "explanation": "",
                }
            )
        else:
            kind, adjustment = spec
            entries.append(
                {
                    "note_index": index,
                    "applies": True,
                    "directive_type": kind,
                    "structured_adjustment": adjustment,
                    "explanation": "",
                }
            )
    return entries


def solar(hours: list[int], factor: float) -> Spec:
    return ("solar_reduction", {"hours": hours, "factor": factor})


def no_charge(hours: list[int]) -> Spec:
    return ("no_charge_window", {"hours": hours})


def no_discharge(hours: list[int]) -> Spec:
    return ("no_discharge_window", {"hours": hours})


def reserve(hours: list[int], kwh: float) -> Spec:
    return ("minimum_battery_reserve", {"hours": hours, "minimum_energy_kwh": kwh})


def grid_cap(hours: list[int], kwh: float) -> Spec:
    return ("max_grid_window", {"hours": hours, "max_grid_kwh": kwh})


# A per-run suffix appended to scenario_id. The service caches by canonical
# request, so replaying the same cases would otherwise measure cache hits
# instead of real model latency.
_RUN_SALT = ""


def _scenario_id(case_id: str) -> str:
    return f"{case_id}{_RUN_SALT}"


@dataclass
class Case:
    """One suite case."""

    id: str
    category: str
    description: str
    notes: list[str] | None = None
    hours: list[dict[str, Any]] | None = None
    battery_spec: dict[str, float] | None = None
    expected: list[dict[str, Any]] | None = None
    expect_status: int = 200
    method: str = "POST"
    path: str = "/optimize-energy"
    raw_body: str | None = None
    advisory: bool = False
    # computed at run time
    optimal_cost: float | None = None
    local_status: str = ""

    def payload(self) -> dict[str, Any]:
        return {
            "scenario_id": _scenario_id(self.id),
            "operator_notes": self.notes or [],
            "hours": self.hours or [],
            "battery": self.battery_spec or battery(),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "description": self.description,
            "method": self.method,
            "path": self.path,
            "advisory": self.advisory,
            "expect_status": self.expect_status,
            "request": self.payload() if self.method == "POST" else None,
            "expected_directive_interpretation": self.expected,
        }


# ---------------------------------------------------------------------------
# the 50 cases
# ---------------------------------------------------------------------------


def build_cases() -> list[Case]:
    cases: list[Case] = []
    add = cases.append

    # ---- A. single directive, core semantics (1-13) ----------------------
    add(Case(
        "SR-PCT-OF", "SOLAR",
        "Percentage *of forecast* means the fraction remaining, not the reduction.",
        ["Rooftop PV output will be limited to about 30% of the forecast between 10 AM and 1 PM."],
        campus_hours(), battery(),
        truth([solar([10, 11, 12], 0.3)]),
    ))
    add(Case(
        "SR-PCT-RED", "SOLAR",
        "Percentage *reduction* must be inverted: 45% off leaves 0.55.",
        ["Expect a 45% reduction in solar generation from 9 AM until noon."],
        campus_hours(), battery(),
        truth([solar([9, 10, 11], 0.55)]),
    ))
    add(Case(
        "SR-HALF", "SOLAR",
        "Fractional wording: 'roughly half of normal output'.",
        ["Clouds will leave roughly half of the normal solar output from 11 AM to 2 PM."],
        campus_hours(), battery(),
        truth([solar([11, 12, 13], 0.5)]),
    ))
    add(Case(
        "SR-DROP-TO", "SOLAR",
        "'drop to X%' is a remainder, not a reduction.",
        ["Solar production will drop to about 15% during the 2 PM to 4 PM window."],
        campus_hours(), battery(),
        truth([solar([14, 15], 0.15)]),
    ))
    add(Case(
        "NC-DO-NOT", "CHARGE",
        "Explicit prohibition on charging.",
        ["Do not charge the battery between 1 PM and 4 PM."],
        campus_hours(), battery(),
        truth([no_charge([13, 14, 15])]),
    ))
    add(Case(
        "NC-OFFLINE", "CHARGE",
        "Indirect availability wording: charger offline for servicing.",
        ["The battery charger will be offline from 3 AM until 6 AM for servicing."],
        campus_hours(), battery(),
        truth([no_charge([3, 4, 5])]),
    ))
    add(Case(
        "ND-MUST-NOT", "DISCHARGE",
        "Explicit prohibition on discharging.",
        ["The battery must not discharge from 5 PM until 8 PM during protection testing."],
        campus_hours(), battery(),
        truth([no_discharge([17, 18, 19])]),
    ))
    add(Case(
        "ND-DISABLED", "DISCHARGE",
        "Passive-voice availability wording.",
        ["Discharging is disabled between 6 PM and 9 PM while the inverter is recalibrated."],
        campus_hours(), battery(),
        truth([no_discharge([18, 19, 20])]),
    ))
    add(Case(
        "RES-KWH", "RESERVE",
        "Absolute reserve in kWh.",
        ["Keep at least 120 kWh in the battery from 6 PM until 9 PM."],
        campus_hours(), battery(),
        truth([reserve([18, 19, 20], 120)]),
    ))
    add(Case(
        "RES-PCT", "RESERVE",
        "Relative reserve requires the request's capacity_kwh (60% of 200 = 120).",
        ["Hold at least 60% of battery capacity from 5 PM until 8 PM for the audit."],
        campus_hours(), battery(capacity=200.0),
        truth([reserve([17, 18, 19], 120)]),
    ))
    add(Case(
        "GRID-KWH", "GRID",
        "Hourly grid-import cap that forces battery pre-charging into the peak.",
        ["Grid import must not exceed 180 kWh in any hour from 6 PM until 9 PM."],
        campus_hours(), battery(),
        truth([grid_cap([18, 19, 20], 180)]),
    ))
    add(Case(
        "GRID-TX", "GRID",
        "Tight equipment cap: feasible only with near-maximum discharge.",
        ["The transformer limits grid intake to 175 kWh between 7 PM and 10 PM."],
        campus_hours(), battery(),
        truth([grid_cap([19, 20, 21], 175)]),
    ))
    add(Case(
        "NOOP-ADMIN", "DISTRACTOR",
        "Single obviously irrelevant administrative note.",
        ["The sports office moved next month's registration deadline."],
        campus_hours(), battery(),
        truth([None]),
    ))

    # ---- B. time-window edges (14-20) -----------------------------------
    add(Case(
        "WIN-SINGLE", "TIME",
        "A single-hour window must not expand to two hours.",
        ["Keep at least 90 kWh in the battery during the 7 PM hour."],
        campus_hours(), battery(),
        truth([reserve([19], 90)]),
    ))
    add(Case(
        "WIN-TO-MIDNIGHT", "TIME",
        "Window ending at midnight: 10 PM until midnight -> [22, 23].",
        ["Do not charge the battery from 10 PM until midnight."],
        campus_hours(), battery(),
        truth([no_charge([22, 23])]),
    ))
    add(Case(
        "WIN-FROM-MIDNIGHT", "TIME",
        "Window starting at hour 0.",
        ["Battery charging is unavailable from midnight until 3 AM."],
        campus_hours(), battery(),
        truth([no_charge([0, 1, 2])]),
    ))
    add(Case(
        "WIN-NOON", "TIME",
        "'noon' must resolve to hour 12, end-exclusive.",
        ["Solar output will be reduced to 40% from noon until 2 PM."],
        campus_hours(), battery(),
        truth([solar([12, 13], 0.4)]),
    ))
    add(Case(
        "WIN-LONG", "TIME",
        "Six-hour daylight window.",
        ["Grid import is capped at 130 kWh from 11 AM until 5 PM."],
        campus_hours(), battery(),
        truth([grid_cap([11, 12, 13, 14, 15, 16], 130)]),
    ))
    add(Case(
        "WIN-EARLY", "TIME",
        "Pre-dawn window crossing the cheap tariff block.",
        ["The charger is isolated from 2 AM until 5 AM."],
        campus_hours(), battery(),
        truth([no_charge([2, 3, 4])]),
    ))
    add(Case(
        "WIN-LATE", "TIME",
        "Late-evening two-hour window.",
        ["The battery must not discharge from 9 PM until 11 PM."],
        campus_hours(), battery(),
        truth([no_discharge([21, 22])]),
    ))

    # ---- C. multiple directives (21-30) ---------------------------------
    add(Case(
        "M-SOLAR-CHARGE", "MULTI",
        "Two different directive types in one scenario.",
        [
            "Solar output will drop to 50% of forecast between 10 AM and 1 PM.",
            "The charging circuit is unavailable from 2 PM until 4 PM.",
        ],
        campus_hours(), battery(),
        truth([solar([10, 11, 12], 0.5), no_charge([14, 15])]),
    ))
    add(Case(
        "M-RES-GRID", "MULTI",
        "Reserve plus grid cap covering the evening peak together.",
        [
            "Keep at least 100 kWh in the battery from 6 PM until 10 PM.",
            "Grid intake must stay at or below 180 kWh from 7 PM until 10 PM.",
        ],
        campus_hours(), battery(),
        truth([reserve([18, 19, 20, 21], 100), grid_cap([19, 20, 21], 180)]),
    ))
    add(Case(
        "M-NC-ND", "MULTI",
        "Both battery actions blocked, in disjoint windows.",
        [
            "Battery charging is disabled from 11 AM until 1 PM.",
            "Do not discharge the battery from 5 PM until 7 PM.",
        ],
        campus_hours(), battery(),
        truth([no_charge([11, 12]), no_discharge([17, 18])]),
    ))
    add(Case(
        "M-TRIPLE", "MULTI",
        "Three notes: one distractor must be discarded.",
        [
            "Panel cleaning will leave about 25% of normal solar from noon until 3 PM.",
            "Hold at least 130 kWh in reserve from 6 PM until 9 PM.",
            "The library is extending its opening hours next week.",
        ],
        campus_hours(), battery(),
        truth([solar([12, 13, 14], 0.25), reserve([18, 19, 20], 130), None]),
    ))
    add(Case(
        "M-GRID-NOOP", "MULTI",
        "One real directive plus one distractor.",
        [
            "The feeder caps grid import at 185 kWh per hour from 6 PM until 9 PM.",
            "Cafeteria menus rotate on Monday.",
        ],
        campus_hours(), battery(),
        truth([grid_cap([18, 19, 20], 185), None]),
    ))
    add(Case(
        "M-RES-ND", "MULTI",
        "Reserve that must hold through a no-discharge window: battery must be pre-charged.",
        [
            "Keep at least 80 kWh in the battery from 5 PM until 9 PM.",
            "The battery must not discharge between 6 PM and 8 PM.",
        ],
        campus_hours(), battery(capacity=240.0, initial=130.0, max_charge=60.0, max_discharge=60.0),
        truth([reserve([17, 18, 19, 20], 80), no_discharge([18, 19])]),
    ))
    add(Case(
        "M-SOLAR2", "MULTI",
        "Two solar reductions in disjoint windows (merge semantics).",
        [
            "Solar output will fall to 50% between 8 AM and 10 AM.",
            "A second reduction to 50% applies from 1 PM until 3 PM.",
        ],
        campus_hours(), battery(),
        truth([solar([8, 9], 0.5), solar([13, 14], 0.5)]),
    ))
    add(Case(
        "M-RES2", "MULTI",
        "Two reserves must merge by taking the strictest value per hour.",
        [
            "Keep at least 90 kWh in the battery from 5 PM until 8 PM.",
            "A stricter 140 kWh reserve applies from 7 PM until 10 PM.",
        ],
        campus_hours(), battery(capacity=260.0, initial=140.0, max_charge=65.0, max_discharge=65.0),
        truth([reserve([17, 18, 19], 90), reserve([19, 20, 21], 140)]),
    ))
    add(Case(
        "M-GRID2", "MULTI",
        "Two grid caps must merge by taking the tightest value per hour.",
        [
            "Grid import is capped at 182 kWh from 6 PM until 8 PM.",
            "A second cap of 190 kWh applies from 7 PM until 9 PM.",
        ],
        campus_hours(), battery(),
        truth([grid_cap([18, 19], 182), grid_cap([19, 20], 190)]),
    ))
    add(Case(
        "M-FULL", "MULTI",
        "Three simultaneous hard constraints around the evening peak.",
        [
            "Keep at least 100 kWh in the battery from 6 PM until 9 PM.",
            "Grid intake must not exceed 170 kWh between 7 PM and 9 PM.",
            "Battery charging is unavailable from 1 PM until 4 PM.",
        ],
        campus_hours(), battery(capacity=240.0, initial=130.0, max_charge=60.0, max_discharge=60.0),
        truth([reserve([18, 19, 20], 100), grid_cap([19, 20], 170), no_charge([13, 14, 15])]),
    ))

    # ---- D. paraphrase robustness (31-36) --------------------------------
    add(Case(
        "P-NC-A", "PARAPHRASE",
        "Charging prohibition, formal wording.",
        ["Charging is not permitted from 1 PM to 3 PM."],
        campus_hours(), battery(),
        truth([no_charge([13, 14])]),
    ))
    add(Case(
        "P-NC-B", "PARAPHRASE",
        "Charging prohibition, terse wording.",
        ["No battery charging between 1 PM and 3 PM."],
        campus_hours(), battery(),
        truth([no_charge([13, 14])]),
    ))
    add(Case(
        "P-ND-A", "PARAPHRASE",
        "Discharge prohibition, permissive-verb wording.",
        ["The battery may not discharge from 5 PM until 7 PM."],
        campus_hours(), battery(),
        truth([no_discharge([17, 18])]),
    ))
    add(Case(
        "P-ND-B", "PARAPHRASE",
        "Discharge prohibition, imperative wording.",
        ["Prevent battery discharge between 5 PM and 7 PM."],
        campus_hours(), battery(),
        truth([no_discharge([17, 18])]),
    ))
    add(Case(
        "P-RES-A", "PARAPHRASE",
        "Reserve expressed as a floor, not a 'keep at least'.",
        ["Maintain a minimum of 100 kWh in the battery from 6 PM to 9 PM."],
        campus_hours(), battery(),
        truth([reserve([18, 19, 20], 100)]),
    ))
    add(Case(
        "P-RES-B", "PARAPHRASE",
        "Reserve expressed as a prohibition on falling below a level.",
        ["The battery should not fall below 100 kWh between 6 PM and 9 PM."],
        campus_hours(), battery(),
        truth([reserve([18, 19, 20], 100)]),
    ))

    # ---- E. distractor discrimination (37-40) ---------------------------
    add(Case(
        "N-LIBRARY", "DISTRACTOR",
        "Administrative note that mentions 'hours' - must not become a time window.",
        ["The library is extending its opening hours next week."],
        campus_hours(), battery(),
        truth([None]),
    ))
    add(Case(
        "N-SEMINAR", "DISTRACTOR",
        "Booking change with a date reference - must be ignored.",
        ["A seminar room booking was moved to next week."],
        campus_hours(), battery(),
        truth([None]),
    ))
    add(Case(
        "N-ELEVATOR", "DISTRACTOR",
        "Facilities maintenance that has nothing to do with energy.",
        ["Annual elevator maintenance is scheduled for next month."],
        campus_hours(), battery(),
        truth([None]),
    ))
    add(Case(
        "N-MIXED-DISTRACT", "DISTRACTOR",
        "Two distractors and one real directive must be separated correctly.",
        [
            "The student affairs office will publish club notices tomorrow.",
            "Do not charge the battery from 2 PM until 4 PM.",
            "A new cafeteria vendor starts next quarter.",
        ],
        campus_hours(), battery(),
        truth([None, no_charge([14, 15]), None]),
    ))

    # ---- F. energy / numeric stress (41-46) -----------------------------
    add(Case(
        "X-NO-SOLAR", "STRESS",
        "Zero solar all day; a solar directive still must be reported and applied.",
        ["Solar output will be reduced to 20% between 9 AM and 11 AM."],
        campus_hours(solar_zero=True), battery(),
        truth([solar([9, 10], 0.2)]),
    ))
    add(Case(
        "X-SOLAR-SURPLUS", "STRESS",
        "Heavy solar requiring curtailment; no directives at all.",
        ["The grounds team will repaint the car park lines this weekend."],
        campus_hours(solar_scale=1.8), battery(),
        truth([None]),
    ))
    add(Case(
        "X-FLAT-TARIFF", "STRESS",
        "Flat tariff removes arbitrage value; plan must stay valid and neutral.",
        ["The signage committee approved a new font for campus boards."],
        campus_hours(tariff_override=[12.0] * 24), battery(),
        truth([None]),
    ))
    add(Case(
        "X-EXTREME-PEAK", "STRESS",
        "One very expensive hour forces maximum discharge there.",
        ["Bicycle parking is being resurfaced near the engineering building."],
        campus_hours(tariff_override=[8.0] * 19 + [70.0] + [8.0] * 4), battery(),
        truth([None]),
    ))
    add(Case(
        "X-TINY-BATTERY", "STRESS",
        "Small battery; reserve must be clamped inside capacity and stay feasible.",
        ["Keep at least 50 kWh in the battery from 6 PM until 8 PM."],
        campus_hours(), battery(capacity=60.0, initial=30.0, minimum=10.0, max_charge=25.0, max_discharge=25.0),
        truth([reserve([18, 19], 50)]),
    ))
    add(Case(
        "X-ZERO-DISCHARGE-RATE", "STRESS",
        "Discharge rate of zero; battery can only charge, so it must stay idle.",
        ["The IT department will roll out a new printing quota in October."],
        campus_hours(), battery(max_discharge=0.0),
        truth([None]),
    ))

    # ---- G. API contract (47-50) ----------------------------------------
    add(Case(
        "E-HEALTH", "API", "Readiness probe returns exactly status=ok.",
        method="GET", path="/health", expect_status=200,
    ))
    add(Case(
        "E-MALFORMED-JSON", "API", "Malformed JSON body must be rejected with 400.",
        raw_body="{not valid json", expect_status=400,
    ))
    add(Case(
        "E-TOO-MANY-NOTES", "API", "Four operator notes exceed the documented maximum.",
        notes=["a", "b", "c", "d"], hours=campus_hours(), battery_spec=battery(),
        expect_status=400,
    ))
    add(Case(
        "E-SEMANTIC-INVALID", "API",
        "initial_energy_kwh above capacity is well-formed but semantically invalid -> 422.",
        notes=["A note that is never reached."], hours=campus_hours(),
        battery_spec={**battery(), "initial_energy_kwh": 9999.0},
        expect_status=422,
    ))

    return cases


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


@dataclass
class Outcome:
    case: Case
    status: int | None = None
    latency_ms: float = 0.0
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cost: float | None = None
    optimal: float | None = None
    fallback_entries: int = 0
    entry_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures


def compare_directives(actual: Any, expected: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    if not isinstance(actual, list):
        return ["directive_interpretation is not an array"]
    if len(actual) != len(expected):
        return [f"expected {len(expected)} entries, got {len(actual)}"]

    for got, want in zip(actual, expected):
        tag = f"note {want['note_index']}"
        if not isinstance(got, dict):
            problems.append(f"{tag}: entry is not an object")
            continue
        if got.get("note_index") != want["note_index"]:
            problems.append(f"{tag}: note_index={got.get('note_index')}")
        if bool(got.get("applies")) != want["applies"]:
            problems.append(f"{tag}: applies={got.get('applies')} want {want['applies']}")
        if got.get("directive_type") != want["directive_type"]:
            problems.append(f"{tag}: type={got.get('directive_type')} want {want['directive_type']}")
            continue

        got_adjustment = got.get("structured_adjustment")
        want_adjustment = want["structured_adjustment"]
        if want_adjustment is None:
            if got_adjustment is not None:
                problems.append(f"{tag}: adjustment should be null")
            continue
        if not isinstance(got_adjustment, dict):
            problems.append(f"{tag}: adjustment is not an object")
            continue
        if set(got_adjustment) != set(want_adjustment):
            problems.append(
                f"{tag}: adjustment keys {sorted(got_adjustment)} want {sorted(want_adjustment)}"
            )
            continue
        for key, want_value in want_adjustment.items():
            got_value = got_adjustment[key]
            if key == "hours":
                if list(got_value) != list(want_value):
                    problems.append(f"{tag}: hours {got_value} want {want_value}")
            elif key == "factor":
                if abs(float(got_value) - float(want_value)) > FACTOR_TOLERANCE:
                    problems.append(f"{tag}: factor {got_value} want {want_value}")
            else:
                if abs(float(got_value) - float(want_value)) > VALUE_TOLERANCE:
                    problems.append(f"{tag}: {key}={got_value} want {want_value}")
    return problems


def compute_optimum(case: Case) -> tuple[float | None, str]:
    """Local reference optimum for the scenario + ground truth."""
    if case.expected is None or case.method != "POST":
        return None, ""
    request = ScenarioRequest(**case.payload())
    entries = [DirectiveInterpretation(**entry) for entry in case.expected]
    result = optimize(request, compile_constraints(entries))
    return result.objective, result.status


def evaluate_body(case: Case, body: dict[str, Any], outcome: Outcome) -> None:
    if not REQUIRED_RESPONSE_KEYS.issubset(body):
        missing = sorted(REQUIRED_RESPONSE_KEYS - set(body))
        outcome.failures.append(f"missing response keys {missing}")
        return
    if body.get("scenario_id") != case.payload()["scenario_id"]:
        outcome.failures.append("scenario_id not echoed")

    plan = body.get("hourly_plan")
    if not isinstance(plan, list) or len(plan) != 24:
        outcome.failures.append(f"hourly_plan length {len(plan) if isinstance(plan, list) else 'n/a'}")
        return

    # Latency-independent arithmetic consistency.
    tariff = {entry["hour"]: entry["tariff_bdt_per_kwh"] for entry in (case.hours or [])}
    try:
        total_grid = sum(float(entry["grid_kwh"]) for entry in plan)
        total_cost = sum(float(entry["grid_kwh"]) * tariff[entry["hour"]] for entry in plan)
        peak = max(float(entry["grid_kwh"]) for entry in plan)
    except (KeyError, TypeError, ValueError) as exc:
        outcome.failures.append(f"plan not numeric: {type(exc).__name__}")
        return
    outcome.cost = float(body["total_cost_bdt"])
    for name, recomputed in (("total_grid_kwh", total_grid), ("total_cost_bdt", total_cost), ("peak_grid_kwh", peak)):
        if abs(float(body[name]) - recomputed) > COST_TOLERANCE:
            outcome.failures.append(f"{name} {body[name]} != recomputed {recomputed:.4f}")

    outcome.entry_count = len(body.get("directive_interpretation") or [])
    outcome.fallback_entries = sum(
        1
        for entry in (body.get("directive_interpretation") or [])
        if isinstance(entry, dict)
        and isinstance(entry.get("explanation"), str)
        and entry["explanation"].startswith("Rule-based fallback")
    )

    if case.expected is not None:
        outcome.failures.extend(compare_directives(body.get("directive_interpretation"), case.expected))

        # The judge's real method: replay the plan against GROUND TRUTH.
        request = ScenarioRequest(**case.payload())
        entries = [DirectiveInterpretation(**entry) for entry in case.expected]
        try:
            plan_entries = [
                HourlyPlanEntry(**{key: entry[key] for key in PLAN_FIELDS if key in entry})
                for entry in plan
            ]
        except Exception as exc:  # noqa: BLE001
            outcome.failures.append(f"plan entries malformed: {type(exc).__name__}")
            return
        report = verify_plan(request, entries, plan_entries, tolerance=COST_TOLERANCE)
        if not report.ok:
            codes = sorted({violation.code for violation in report.violations})
            outcome.failures.append(f"replay vs ground truth: {','.join(codes)}")

    if outcome.optimal is not None and outcome.cost is not None:
        if outcome.cost > outcome.optimal + COST_TOLERANCE:
            ratio = outcome.optimal / outcome.cost if outcome.cost else 0.0
            outcome.failures.append(
                f"cost {outcome.cost:.2f} above optimum {outcome.optimal:.2f} (ratio {ratio:.4f})"
            )


def run_case(client: httpx.Client, case: Case, verbose: bool) -> Outcome:
    outcome = Outcome(case=case)
    if case.expected is not None and case.method == "POST":
        outcome.optimal, outcome.local_status = compute_optimum(case)
        if outcome.local_status != "optimal":
            outcome.notes.append(f"local reference solve status={outcome.local_status}")

    started = time.perf_counter()
    try:
        if case.method == "GET":
            response = client.get(case.path)
        elif case.raw_body is not None:
            response = client.post(
                case.path, content=case.raw_body, headers={"Content-Type": "application/json"}
            )
        else:
            response = client.post(case.path, json=case.payload())
    except httpx.HTTPError as exc:
        outcome.latency_ms = (time.perf_counter() - started) * 1000.0
        outcome.failures.append(f"transport error: {type(exc).__name__}")
        return outcome
    outcome.latency_ms = (time.perf_counter() - started) * 1000.0
    outcome.status = response.status_code

    if response.status_code != case.expect_status:
        outcome.failures.append(f"status {response.status_code} want {case.expect_status}")
        return outcome

    try:
        body = response.json()
    except ValueError:
        outcome.failures.append("response is not JSON")
        return outcome

    if case.expect_status != 200:
        if not isinstance(body, dict) or "error" not in body:
            outcome.failures.append("error body missing 'error' key")
        return outcome

    if case.method == "GET":
        if body.get("status") != "ok":
            outcome.failures.append(f"health status={body.get('status')!r}")
        return outcome

    evaluate_body(case, body, outcome)
    if verbose and outcome.failures:
        for failure in outcome.failures:
            print(f"      ! {failure}")
    return outcome


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def warm_up(client: httpx.Client, attempts: int = 10, delay: float = 10.0) -> bool:
    """Render free instances sleep; poll /health until ready."""
    for attempt in range(attempts):
        try:
            response = client.get("/health")
            if response.status_code == 200 and response.json().get("status") == "ok":
                if attempt:
                    print(f"service woke after ~{attempt * delay:.0f}s")
                return True
        except httpx.HTTPError:
            pass
        if attempt < attempts - 1:
            print(f"  waiting for the service to become ready ({attempt + 1}/{attempts})...")
            time.sleep(delay)
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"base URL (default {DEFAULT_URL})")
    parser.add_argument("--category", help="run only one category")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--limit", type=int, help="run at most N cases")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout (s)")
    parser.add_argument("--dump", help="write the case pack to this JSON path and exit")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-out", help="write machine-readable results here")
    parser.add_argument(
        "--salt",
        default="",
        help="suffix appended to scenario_id to defeat the service cache and measure true latency",
    )
    args = parser.parse_args(argv)

    global _RUN_SALT
    _RUN_SALT = args.salt or ""

    cases = build_cases()
    assert len(cases) == 50, f"expected 50 cases, built {len(cases)}"

    if args.dump:
        Path(args.dump).write_text(
            json.dumps([case.to_json() for case in cases], indent=2), encoding="utf-8"
        )
        print(f"wrote {len(cases)} cases to {args.dump}")
        return 0

    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        cases = [case for case in cases if case.id in wanted]
    if args.category:
        cases = [case for case in cases if case.category.upper() == args.category.upper()]
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("no cases selected")
        return 2

    print(f"target: {args.url}")
    print(f"cases:  {len(cases)}")

    outcomes: list[Outcome] = []
    with httpx.Client(base_url=args.url, timeout=httpx.Timeout(args.timeout, connect=15.0)) as client:
        if not warm_up(client):
            print("service did not become ready; aborting")
            return 2

        current = ""
        for case in cases:
            if case.category != current:
                current = case.category
                print(f"\n--- {current} ---")
            outcome = run_case(client, case, args.verbose)
            outcomes.append(outcome)
            mark = "PASS" if outcome.ok else ("WARN" if case.advisory else "FAIL")
            detail = ""
            if outcome.cost is not None and outcome.optimal is not None:
                detail = f" cost={outcome.cost:>10.2f} opt={outcome.optimal:>10.2f}"
            elif outcome.cost is not None:
                detail = f" cost={outcome.cost:>10.2f}"
            if outcome.fallback_entries and outcome.entry_count:
                detail += f" fallback={outcome.fallback_entries}/{outcome.entry_count}"
            print(
                f"{case.id:<22}{mark:<5}status={str(outcome.status):<4}"
                f"{outcome.latency_ms:>7.0f}ms{detail}"
            )
            if not outcome.ok and not args.verbose:
                for failure in outcome.failures:
                    print(f"      ! {failure}")
            for note in outcome.notes:
                print(f"      ~ {note}")

    scored = [o for o in outcomes if not o.case.advisory]
    failures = [o for o in scored if not o.ok]
    latencies = sorted(o.latency_ms for o in outcomes if o.status is not None)

    print("\n" + "=" * 72)
    by_category: dict[str, list[Outcome]] = {}
    for outcome in outcomes:
        by_category.setdefault(outcome.case.category, []).append(outcome)
    for category, group in by_category.items():
        passed = sum(1 for o in group if o.ok)
        print(f"  {category:<12} {passed}/{len(group)} passed")

    if latencies:
        p95 = latencies[min(len(latencies) - 1, int(round(0.95 * (len(latencies) - 1))))]
        print(
            f"\n  latency: min={latencies[0]:.0f}ms  median={statistics.median(latencies):.0f}ms  "
            f"p95={p95:.0f}ms  max={latencies[-1]:.0f}ms"
        )
    print(f"  total:   {len(outcomes) - len(failures)}/{len(outcomes)} passed")
    if failures:
        print(f"  FAILED:  {', '.join(o.case.id for o in failures)}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                [
                    {
                        "id": o.case.id,
                        "category": o.case.category,
                        "status": o.status,
                        "ok": o.ok,
                        "latency_ms": round(o.latency_ms, 1),
                        "cost": o.cost,
                        "optimal": o.optimal,
                        "failures": o.failures,
                        "notes": o.notes,
                    }
                    for o in outcomes
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  results written to {args.json_out}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
