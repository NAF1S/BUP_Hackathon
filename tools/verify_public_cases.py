"""Developer tool: solve every public sample case with the ground-truth
directives and compare our optimum against the organizer's reference cost.

Usage (from the repository root):

    python tools/verify_public_cases.py [path/to/cases.json]

This exercises the real optimizer and the real replay verifier without needing a
provider key, so it is the primary local correctness check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.optimization.constraints import compile_constraints  # noqa: E402
from app.optimization.replay import verify_plan  # noqa: E402
from app.optimization.solver import optimize  # noqa: E402
from app.schemas import DirectiveInterpretation, ScenarioRequest  # noqa: E402

DEFAULT_CANDIDATES = [
    Path(__file__).resolve().parents[2] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    Path(__file__).resolve().parents[1] / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
]


def load_pack(path: Path | None) -> dict:
    if path is None:
        path = next((p for p in DEFAULT_CANDIDATES if p.exists()), None)
        if path is None:
            raise SystemExit("Could not locate the sample case pack; pass a path explicitly.")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def ground_truth(case: dict) -> list[DirectiveInterpretation]:
    return [
        DirectiveInterpretation(**entry)
        for entry in case["expected_output"]["directive_interpretation"]
    ]


def main(argv: list[str]) -> int:
    pack = load_pack(Path(argv[1]) if len(argv) > 1 else None)
    tolerance = 0.01

    header = (
        f"{'case':<12}{'entries':>8}{'our cost':>12}{'ref cost':>12}"
        f"{'delta':>10}{'grid':>11}{'peak':>9}  replay"
    )
    print(header)
    print("-" * len(header))

    failures = 0
    total_reference = 0.0
    total_ours = 0.0

    for case in pack["cases"]:
        request = ScenarioRequest(**case["input"])
        interpretation = ground_truth(case)
        constraints = compile_constraints(interpretation)

        result = optimize(request, constraints, tolerance=tolerance)
        plan = _plan(request, result)
        report = verify_plan(request, interpretation, plan, tolerance=tolerance)

        reference = float(case["expected_output"]["total_cost_bdt"])
        delta = result.objective - reference
        total_reference += reference
        total_ours += result.objective

        status = "OK" if report.ok else "VIOLATIONS: " + ",".join(sorted(set(report.codes())))
        verdict = "ok" if (report.ok and delta <= tolerance) else "FAIL"
        if verdict == "FAIL":
            failures += 1

        print(
            f"{case['id']:<12}{len(interpretation):>8}{result.objective:>12.2f}"
            f"{reference:>12.2f}{delta:>10.2f}{sum(result.grid):>11.2f}"
            f"{max(result.grid):>9.2f}  {status} [{verdict}]"
        )

    print("-" * len(header))
    print(
        f"total: ours={total_ours:.2f} BDT  reference={total_reference:.2f} BDT  "
        f"savings={total_reference - total_ours:.2f} BDT"
    )
    print(f"failures: {failures}/{len(pack['cases'])}")
    return 1 if failures else 0


def _plan(request: ScenarioRequest, result) -> list:
    """Build response-schema plan entries so the replay verifier can be reused."""
    from app.pipeline import _plan_entries

    return _plan_entries(result)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
