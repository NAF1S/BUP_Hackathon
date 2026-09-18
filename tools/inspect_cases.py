"""Developer utility: summarize the public sample case pack.

Usage:
    python tools/inspect_cases.py [path-to-cases.json]

Prints, for every public case, the operator notes, the expected
machine-checkable directive interpretation, and reference cost figures so the
solver can be cross-checked locally. This script is NOT part of the judged
service path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DEFAULT_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    Path(__file__).resolve().parents[1] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
]


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def summarize(case: dict) -> None:
    inp = case["input"]
    out = case.get("expected_output", {})
    bat = inp["battery"]
    hours = inp["hours"]

    print("=" * 100)
    print(f"{case['id']}  |  {case.get('label', '')}")
    print("-" * 100)
    print(f"  battery: capacity={bat['capacity_kwh']} init={bat['initial_energy_kwh']} "
          f"min={bat['minimum_energy_kwh']} max_chg={bat['max_charge_kwh_per_hour']} "
          f"max_dis={bat['max_discharge_kwh_per_hour']}")
    tariff = [h["tariff_bdt_per_kwh"] for h in hours]
    demand = [h["demand_kwh"] for h in hours]
    solar = [h["solar_kwh"] for h in hours]
    print(f"  demand : min={min(demand)} max={max(demand)} sum={sum(demand)}")
    print(f"  solar  : max={max(solar)} sum={sum(solar)}")
    print(f"  tariff : min={min(tariff)} max={max(tariff)}  peak_hours="
          f"{[h['hour'] for h in hours if h['tariff_bdt_per_kwh'] == max(tariff)]}")

    print("  notes:")
    for i, note in enumerate(inp["operator_notes"]):
        print(f"    [{i}] {note}")

    print("  expected directive_interpretation:")
    for entry in out.get("directive_interpretation", []):
        print(f"    [{entry['note_index']}] applies={entry['applies']!s:<5} "
              f"type={entry['directive_type']:<26} adj={json.dumps(entry['structured_adjustment'])}")

    print("  reference metrics:")
    for key in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        if key in out:
            print(f"    {key} = {out[key]}")
    actions = [h["battery_action"] for h in out.get("hourly_plan", [])]
    if actions:
        print(f"    battery actions = {actions}")
    if case.get("rationale"):
        print(f"  rationale: {case['rationale']}")


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        path = Path(argv[1])
    else:
        path = next((p for p in DEFAULT_CANDIDATES if p.exists()), None)
        if path is None:
            print("Could not locate the sample case pack; pass a path explicitly.", file=sys.stderr)
            return 2

    pack = load(path)
    cases = pack["cases"]
    print(f"Pack: {pack['_meta']['title']}  (v{pack['_meta']['version']}, "
          f"{len(cases)} cases) from {path}")

    for case in cases:
        summarize(case)

    # Cross-case directive frequency table.
    counts: dict[str, int] = {}
    for case in cases:
        for entry in case["expected_output"].get("directive_interpretation", []):
            counts[entry["directive_type"]] = counts.get(entry["directive_type"], 0) + 1
    print("=" * 100)
    print("directive_type frequency across public cases:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<28} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
