"""Developer tool: smoke-test the live LLM interpretation path.

Usage (from the repository root, with LLM_API_KEY set):

    python tools/check_llm.py            # all 10 public cases
    python tools/check_llm.py 3          # a single case index

For each case it calls the configured provider once, runs the deterministic
guardrails, and compares the parsed directives against the published ground
truth. This validates the prompt and the provider wiring without needing the
full service to be running.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.guardrails.normalize import extract_json_payload, normalize_interpretation  # noqa: E402
from app.llm.openai_compat import build_llm_client  # noqa: E402
from app.llm.prompts import build_messages  # noqa: E402
from app.schemas import ScenarioRequest  # noqa: E402

CASE_PACK_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    Path(__file__).resolve().parents[1] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
]


def load_cases() -> list[dict]:
    path = next((p for p in CASE_PACK_CANDIDATES if p.exists()), None)
    if path is None:
        raise SystemExit("public sample case pack not found")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)["cases"]


def comparable(entry) -> tuple:
    adjustment = getattr(entry, "structured_adjustment", None)
    if adjustment is not None:
        adjustment = {
            key: (
                list(value)
                if key == "hours"
                else (round(float(value), 3) if isinstance(value, (int, float)) else value)
            )
            for key, value in sorted(adjustment.items())
        }
    kind = getattr(entry.directive_type, "value", entry.directive_type)
    return (entry.note_index, bool(entry.applies), kind, json.dumps(adjustment, sort_keys=True))


async def check(cases: list[dict]) -> int:
    settings = get_settings()
    client = build_llm_client(settings)
    if client is None:
        print("LLM_API_KEY is not set; nothing to test.")
        return 2

    print(f"provider={settings.llm_base_url} model={settings.llm_model} cases={len(cases)}\n")
    mismatches = 0
    try:
        for case in cases:
            request = ScenarioRequest(**case["input"])
            messages = build_messages(
                request.scenario_id, request.operator_notes, request.battery
            )
            raw = await client.complete_json(messages[0]["content"], messages[1]["content"])

            slots, problems = normalize_interpretation(
                extract_json_payload(raw), request.operator_notes, request.battery
            )
            expected = case["expected_output"]["directive_interpretation"]

            ok = True
            detail: list[str] = []
            for index, want in enumerate(expected):
                got = slots[index]
                if got is None:
                    ok = False
                    detail.append(f"note {index}: no usable entry")
                    continue
                got_key = comparable(got)
                want_key = comparable(type("E", (), {
                    "note_index": want["note_index"],
                    "applies": want["applies"],
                    "directive_type": want["directive_type"],
                    "structured_adjustment": want["structured_adjustment"],
                })())
                if got_key != want_key:
                    ok = False
                    detail.append(f"note {index}: {got_key} != {want_key}")

            if not ok:
                mismatches += 1
            status = "match" if ok else "MISMATCH"
            print(f"{case['id']:<12} {status}" + (f"  {'; '.join(detail)}" if detail else ""))
            if problems:
                print(f"{'':<13}guardrail notes: {problems[:3]}")
    finally:
        await client.aclose()

    print(f"\nmismatches: {mismatches}/{len(cases)}")
    return 1 if mismatches else 0


def main(argv: list[str]) -> int:
    cases = load_cases()
    if len(argv) > 1:
        cases = [cases[int(argv[1])]]
    return asyncio.run(check(cases))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
