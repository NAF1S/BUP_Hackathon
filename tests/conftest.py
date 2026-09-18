"""Shared fixtures for the GridWise service test suite."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings, get_settings  # noqa: E402

_CASE_PACK_CANDIDATES = [
    ROOT.parent / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    ROOT / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
]


@pytest.fixture(autouse=True)
def hermetic_environment(monkeypatch: pytest.MonkeyPatch):
    """Keep the suite offline, fast, and deterministic.

    Without this, a developer's local ``.env`` would be picked up by
    ``get_settings()`` and the API tests would start making real, billed
    provider calls.
    """
    monkeypatch.setenv("DISABLE_DOTENV", "1")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def pack() -> dict[str, Any]:
    path = next((p for p in _CASE_PACK_CANDIDATES if p.exists()), None)
    if path is None:
        pytest.skip("public sample case pack not found")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def cases(pack: dict[str, Any]) -> list[dict[str, Any]]:
    return pack["cases"]


@pytest.fixture
def settings() -> Settings:
    """Deterministic, offline configuration: no provider, no cache."""
    return Settings(
        llm_api_key=None,
        cache_size=0,
        allow_deterministic_fallback=True,
        numeric_tolerance=0.01,
    )
