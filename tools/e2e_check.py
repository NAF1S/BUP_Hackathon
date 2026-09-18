"""Developer tool: end-to-end check against a running service over real HTTP.

Exercises the complete judged path - HTTP -> Pydantic -> LLM -> guardrails ->
optimizer -> replay -> JSON - for every public sample case, and reports latency.

Usage:
    python tools/e2e_check.py [base_url]

Defaults to http://127.0.0.1:8000. Exits non-zero if any case fails.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import httpx

TOLERANCE = 0.01
REQUIRED_KEYS = {
    "scenario_id",
    "directive_interpretation",
    "hourly_plan",
    "total_grid_kwh",
    "total_cost_bdt",
    "peak_grid_kwh",
    "plan_summary",
}
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


def directive_matches(actual: list[dict], expected: list[dict]) -> bool:
    if len(actual) != len(expected):
        return False
    for got, want in zip(actual, expected):
        if got["note_index"] != want["note_index"]:
            return False
        if bool(got["applies"]) != bool(want["applies"]):
            return False
        if got["directive_type"] != want["directive_type"]:
            return False
        got_adjustment = got["structured_adjustment"] or {}
        want_adjustment = want["structured_adjustment"] or {}
        if set(got_adjustment) != set(want_adjustment):
            return False
        for key, want_value in want_adjustment.items():
            got_value = got_adjustment[key]
            if key == "hours":
                if list(got_value) != list(want_value):
                    return False
            elif abs(float(got_value) - float(want_value)) > 1e-3:
                return False
    return True


def main(argv: list[str]) -> int:
    base_url = (argv[1] if len(argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
    cases = load_cases()

    with httpx.Client(base_url=base_url, timeout=40.0) as client:
        try:
            health = client.get("/health")
        except httpx.HTTPError as exc:
            print(f"cannot reach {base_url}/health: {exc}")
            return 2
        print(f"GET /health -> {health.status_code} {health.text.strip()}")
        if health.status_code != 200 or health.json().get("status") != "ok":
            print("health check failed")
            return 2

        header = (
            f"\n{'case':<12}{'status':>7}{'cost':>12}{'reference':>12}{'delta':>9}"
            f"{'ms':>8}  directives"
        )
        print(header)
        print("-" * len(header))

        latencies: list[float] = []
        failures = 0

        for case in cases:
            started = time.perf_counter()
            try:
                response = client.post("/optimize-energy", json=case["input"])
            except httpx.HTTPError as exc:
                print(f"{case['id']:<12}{'ERR':>7}  {exc}")
                failures += 1
                continue
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies.append(elapsed_ms)

            if response.status_code != 200:
                print(f"{case['id']:<12}{response.status_code:>7}  {response.text[:120]}")
                failures += 1
                continue

            body = response.json()
            reference = float(case["expected_output"]["total_cost_bdt"])
            cost = float(body["total_cost_bdt"])
            delta = cost - reference

            problems: list[str] = []
            if not REQUIRED_KEYS.issubset(body):
                problems.append("missing schema keys")
            if body["scenario_id"] != case["input"]["scenario_id"]:
                problems.append("scenario_id mismatch")
            if len(body["hourly_plan"]) != 24:
                problems.append("hourly_plan length")
            if cost > reference + TOLERANCE:
                problems.append(f"cost above reference by {delta:.2f}")
            if not directive_matches(
                body["directive_interpretation"],
                case["expected_output"]["directive_interpretation"],
            ):
                problems.append("directive mismatch")

            if problems:
                failures += 1
            verdict = "ok" if not problems else "FAIL"
            print(
                f"{case['id']:<12}{response.status_code:>7}{cost:>12.2f}{reference:>12.2f}"
                f"{delta:>9.2f}{elapsed_ms:>8.0f}  "
                + (verdict if not problems else f"{verdict}: {'; '.join(problems)}")
            )

    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        print("-" * len(header))
        print(
            f"latency: min={min(latencies):.0f}ms  median={statistics.median(latencies):.0f}ms  "
            f"p95={p95:.0f}ms  max={max(latencies):.0f}ms"
        )
        print("p95 target: <=5000ms scores 3/3 on the latency rubric")
    print(f"failures: {failures}/{len(cases)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
