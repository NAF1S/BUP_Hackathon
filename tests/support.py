"""Test doubles and shared helpers."""

from __future__ import annotations

import json
from typing import Any, Sequence


class FakeLLMClient:
    """Deterministic stand-in for a provider client.

    Records how many times it was called and what user prompt it received, so
    tests can assert the recovery ladder behaved as designed.
    """

    name = "fake"

    def __init__(self, responses: str | Sequence[str]) -> None:
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls = 0
        self.last_user_prompt = ""

    async def complete_json(self, system: str, user: str, *, deadline: float | None = None) -> str:
        self.last_user_prompt = user
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[index]

    async def aclose(self) -> None:
        return None


def interpretation_payload(entries: Sequence[dict[str, Any]]) -> str:
    """Render directive entries the way a well-behaved model would."""
    return json.dumps({"directive_interpretation": list(entries)})


def signature(entry: Any) -> tuple:
    """Comparable form of an interpretation entry with rounded numerics."""
    adjustment = getattr(entry, "structured_adjustment", None)
    if adjustment is not None:
        adjustment = {
            key: (
                list(value)
                if key == "hours"
                else (round(float(value), 4) if isinstance(value, (int, float)) else value)
            )
            for key, value in sorted(adjustment.items())
        }
    kind = getattr(entry.directive_type, "value", entry.directive_type)
    return (entry.note_index, bool(entry.applies), kind, json.dumps(adjustment, sort_keys=True))
