"""Prompt construction for operator-note directive extraction.

Design notes
------------
* All conventions live in the system prompt: the six legal directive types, the
  start-inclusive/end-exclusive window rule, and the two opposite percentage
  phrasings ("25% *of* the forecast" vs "80% *reduction*"). Those are the
  highest-value traps in the challenge.
* The few-shot examples deliberately use wording that does **not** appear in the
  public sample pack, so the model learns the *semantics* rather than the
  surface strings. Hard-coding public phrases is explicitly discouraged.
* Battery capacity is passed in the user turn so relative reserves
  ("50% of battery capacity") resolve to absolute kWh.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

SYSTEM_PROMPT = """\
You convert short natural-language campus operator notes into a fixed set of \
structured energy directives for a 24-hour battery/solar/grid schedule.

The schedule covers hours 0 through 23 (hour 0 = midnight). Every note refers to \
that same 24-hour window.

OUTPUT FORMAT
Return ONLY one JSON object. No prose, no markdown fences.
{"directive_interpretation":[{"note_index":0,"applies":true,"directive_type":"...",\
"structured_adjustment":{...},"explanation":"..."}]}
Return exactly one entry per note, in note_index order 0, 1, 2, ... There must be \
no missing, duplicate, or extra entries.

DIRECTIVE TYPES (these six are the only legal values)
1. "solar_reduction" - usable solar is reduced during specific hours.
   structured_adjustment: {"hours":[ints], "factor": number}
   factor is the USABLE FRACTION THAT REMAINS, between 0 and 1.
     "X% of the forecast"      -> factor = X/100
     "drop to X%" / "fall to X%" -> factor = X/100
     "X% reduction"            -> factor = 1 - X/100
     "reduce/drop by X%"       -> factor = 1 - X/100
     "about half"              -> factor = 0.5
     "one-fifth"               -> factor = 0.2
2. "minimum_battery_reserve" - battery energy must stay at or above a level.
   structured_adjustment: {"hours":[ints], "minimum_energy_kwh": number}
   Absolute kWh. If the note gives a percentage of battery capacity, multiply it \
by capacity_kwh from the request.
3. "no_charge_window" - battery charging is unavailable.
   structured_adjustment: {"hours":[ints]}
4. "no_discharge_window" - battery discharging is unavailable.
   structured_adjustment: {"hours":[ints]}
5. "max_grid_window" - grid import may not exceed an amount in an hour.
   structured_adjustment: {"hours":[ints], "max_grid_kwh": number}
6. "no_op" - the note does not affect the 24-hour energy schedule.
   applies MUST be false and structured_adjustment MUST be null.

TIME WINDOW RULE (critical)
Windows are whole hours, START INCLUSIVE and END EXCLUSIVE.
  1 PM to 3 PM        -> [13, 14]
  noon until 2 PM     -> [12, 13]
  2 AM until 5 AM     -> [2, 3, 4]
  6 PM until 9 PM     -> [18, 19, 20]
  11 AM until 2 PM    -> [11, 12, 13]
  between A and B, from A until B, A to B and A through B all behave the same way.
"hours" must contain unique integers from 0 through 23 in ASCENDING order.

APPLIES RULE
Every non-no_op directive uses applies = true with the required adjustment object.
no_op is the only directive allowed to use applies = false.

DO NOT
Never invent a directive type, an hour, or a change to demand, tariff, or battery \
limits. Never add keys that are not listed for the chosen type.

DISTRACTORS
Notes about administration, registrations, menus, bookings, sports, libraries, \
seminars, staffing, or other non-energy topics are "no_op".

EXAMPLES

Example A
notes:
[0] "Rooftop PV output is expected to fall to roughly a third of normal between 9 AM and 11 AM."
response:
{"directive_interpretation":[{"note_index":0,"applies":true,"directive_type":\
"solar_reduction","structured_adjustment":{"hours":[9,10],"factor":0.3333},\
"explanation":"Usable solar is about one third of forecast during 09:00-11:00."}]}

Example B
notes:
[0] "Panel maintenance will cut solar production by 40% from 1 PM until 4 PM."
response:
{"directive_interpretation":[{"note_index":0,"applies":true,"directive_type":\
"solar_reduction","structured_adjustment":{"hours":[13,14,15],"factor":0.6},\
"explanation":"A 40% reduction leaves 60% of forecast solar usable."}]}

Example C
notes:
[0] "Hold at least 60% of battery capacity from 5 PM until 8 PM for the audit.",
[1] "The cafeteria will serve a special menu on Friday."
response:
{"directive_interpretation":[{"note_index":0,"applies":true,"directive_type":\
"minimum_battery_reserve","structured_adjustment":{"hours":[17,18,19],\
"minimum_energy_kwh":120},"explanation":"60% of the 200 kWh capacity held from 17:00-20:00."},\
{"note_index":1,"applies":false,"directive_type":"no_op","structured_adjustment":null,\
"explanation":"Cafeteria menu changes do not affect the energy schedule."}]}

Example D
notes:
[0] "The charger is offline from 2 AM until 4 AM.",
[1] "Grid draw must stay under 140 kWh between 6 PM and 8 PM."
response:
{"directive_interpretation":[{"note_index":0,"applies":true,"directive_type":\
"no_charge_window","structured_adjustment":{"hours":[2,3]},\
"explanation":"Charging is unavailable during 02:00-04:00."},\
{"note_index":1,"applies":true,"directive_type":"max_grid_window",\
"structured_adjustment":{"hours":[18,19],"max_grid_kwh":140},\
"explanation":"Grid import capped at 140 kWh per hour during 18:00-20:00."}]}

Return only the JSON object."""


def build_user_prompt(
    scenario_id: str,
    notes: Sequence[str],
    battery: Any,
    *,
    feedback: Sequence[str] | None = None,
) -> str:
    """Render the per-request user turn, optionally with repair feedback."""
    lines = [
        f"scenario_id: {scenario_id}",
        "battery:",
        f"  capacity_kwh: {battery.capacity_kwh}",
        f"  initial_energy_kwh: {battery.initial_energy_kwh}",
        f"  minimum_energy_kwh: {battery.minimum_energy_kwh}",
        f"  max_charge_kwh_per_hour: {battery.max_charge_kwh_per_hour}",
        f"  max_discharge_kwh_per_hour: {battery.max_discharge_kwh_per_hour}",
        "",
        "operator_notes:",
    ]
    for index, note in enumerate(notes):
        lines.append(f"[{index}] {note}")

    if feedback:
        lines.extend(
            [
                "",
                "Your previous JSON response was rejected by validation for these reasons:",
                *[f"- {item}" for item in feedback],
                "Return a corrected JSON object that fixes every listed problem.",
            ]
        )

    lines.extend(
        [
            "",
            f"Return one JSON object with exactly {len(notes)} directive_interpretation entries.",
        ]
    )
    return "\n".join(lines)


def build_messages(
    scenario_id: str,
    notes: Sequence[str],
    battery: Any,
    *,
    feedback: Sequence[str] | None = None,
) -> list[dict[str, str]]:
    """Chat messages for the interpretation call."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": build_user_prompt(scenario_id, notes, battery, feedback=feedback),
        },
    ]


def describe_feedback(problems: Sequence[str], limit: int = 6) -> list[str]:
    """Trim a problem list into a compact repair instruction."""
    return list(problems[:limit])


def dumps(value: Any) -> str:  # pragma: no cover - debugging helper
    return json.dumps(value, separators=(",", ":"))
