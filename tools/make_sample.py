"""Developer tool: write one public case request body to a JSON file.

Useful for manual `curl` testing of a running service.

Usage (from the repository root):

    python tools/make_sample.py 9 sample_s10.json
    curl -X POST http://127.0.0.1:8000/optimize-energy \
      -H "Content-Type: application/json" --data @sample_s10.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CASE_PACK_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    Path(__file__).resolve().parents[1] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
]


def main(argv: list[str]) -> int:
    index = int(argv[1]) if len(argv) > 1 else 0
    destination = Path(argv[2]) if len(argv) > 2 else Path(f"sample_case_{index}.json")

    path = next((p for p in CASE_PACK_CANDIDATES if p.exists()), None)
    if path is None:
        print("Could not locate the sample case pack.", file=sys.stderr)
        return 2

    with path.open(encoding="utf-8") as handle:
        cases = json.load(handle)["cases"]

    if not 0 <= index < len(cases):
        print(f"index must be between 0 and {len(cases) - 1}", file=sys.stderr)
        return 2

    case = cases[index]
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(case["input"], handle, indent=2)
    print(f"wrote {destination} from {case['id']} ({case.get('label', '')})")
    print(f"notes: {case['input']['operator_notes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
