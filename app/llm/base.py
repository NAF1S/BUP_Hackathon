"""Provider-agnostic client contract.

Any adapter that returns raw model text works, as long as it is a *language
model* producing the structured operator-note interpretation that the optimizer
consumes. Using a model only for the free-text ``plan_summary`` would not satisfy
the challenge requirement.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface the pipeline depends on."""

    name: str

    async def complete_json(
        self, system: str, user: str, *, deadline: float | None = None
    ) -> str:
        """Return the raw model response, which must contain a JSON object."""
        ...

    async def aclose(self) -> None:
        """Release network resources."""
        ...
