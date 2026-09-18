"""HTTP contract tests for the FastAPI surface."""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient

REQUIRED_RESPONSE_KEYS = {
    "scenario_id",
    "directive_interpretation",
    "hourly_plan",
    "total_grid_kwh",
    "total_cost_bdt",
    "peak_grid_kwh",
    "plan_summary",
}


@pytest.fixture
def client(monkeypatch):
    """App instance with no provider key, caching disabled.

    The autouse ``hermetic_environment`` fixture in ``conftest`` has already
    disabled ``.env`` loading, so this stays offline even when a real
    ``.env`` exists on the developer's machine.
    """
    monkeypatch.setenv("CACHE_SIZE", "0")
    monkeypatch.setenv("ALLOW_DETERMINISTIC_FALLBACK", "true")

    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_suite_stays_offline_when_a_dotenv_exists(monkeypatch) -> None:
    """A local `.env` with a real key must not leak into the test process."""
    from app.config import get_settings

    get_settings.cache_clear()
    assert get_settings().llm_configured is False


def sample_input(cases: Any, index: int = 0) -> dict[str, Any]:
    """Deep-copied request body for one public case.

    Accepts either the full case list (with ``index``) or a single case dict, so
    individual tests can read either ``sample_input(cases, 4)`` or
    ``sample_input(cases[4])`` without extra unpacking.
    """
    case = cases[index] if isinstance(cases, list) else cases
    return json.loads(json.dumps(case["input"]))


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# /optimize-energy - success
# ---------------------------------------------------------------------------


def test_optimize_energy_returns_full_schema(client: TestClient, cases) -> None:
    payload = sample_input(cases)
    response = client.post("/optimize-energy", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert REQUIRED_RESPONSE_KEYS.issubset(body)
    assert body["scenario_id"] == payload["scenario_id"]
    assert len(body["hourly_plan"]) == 24
    assert len(body["directive_interpretation"]) == len(payload["operator_notes"])

    for entry in body["hourly_plan"]:
        assert set(entry) == {
            "hour",
            "grid_kwh",
            "solar_used_kwh",
            "battery_action",
            "battery_kwh",
            "battery_energy_after_kwh",
        }
        assert entry["battery_action"] in {"charge", "discharge", "idle"}
        assert entry["grid_kwh"] >= 0
        if entry["battery_action"] == "idle":
            assert entry["battery_kwh"] == 0

    for index, entry in enumerate(body["directive_interpretation"]):
        assert entry["note_index"] == index
        assert entry["directive_type"] in {
            "solar_reduction",
            "minimum_battery_reserve",
            "no_charge_window",
            "no_discharge_window",
            "max_grid_window",
            "no_op",
        }
        if entry["directive_type"] == "no_op":
            assert entry["applies"] is False
            assert entry["structured_adjustment"] is None
        else:
            assert entry["applies"] is True


def test_totals_are_self_consistent(client: TestClient, cases) -> None:
    payload = sample_input(cases[4])
    body = client.post("/optimize-energy", json=payload).json()

    tariff = {hour["hour"]: hour["tariff_bdt_per_kwh"] for hour in payload["hours"]}
    plan = body["hourly_plan"]

    assert body["total_grid_kwh"] == pytest.approx(sum(e["grid_kwh"] for e in plan), abs=0.01)
    assert body["total_cost_bdt"] == pytest.approx(
        sum(e["grid_kwh"] * tariff[e["hour"]] for e in plan), abs=0.01
    )
    assert body["peak_grid_kwh"] == pytest.approx(max(e["grid_kwh"] for e in plan), abs=0.01)
    assert body["plan_summary"]


def test_hours_may_arrive_out_of_order(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    payload["hours"] = list(reversed(payload["hours"]))
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 200
    assert [entry["hour"] for entry in response.json()["hourly_plan"]] == list(range(24))


@pytest.mark.parametrize("index", [0, 5, 9])
def test_every_public_case_is_served(client: TestClient, cases, index: int) -> None:
    response = client.post("/optimize-energy", json=sample_input(cases[index]))
    assert response.status_code == 200, response.text
    assert response.json()["hourly_plan"]


# ---------------------------------------------------------------------------
# /optimize-energy - rejection
# ---------------------------------------------------------------------------


def test_malformed_json_returns_400(client: TestClient) -> None:
    response = client.post(
        "/optimize-energy",
        content="{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_missing_battery_returns_400(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    del payload["battery"]
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400


def test_wrong_hour_count_returns_400(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    payload["hours"] = payload["hours"][:23]
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400


def test_duplicate_hours_return_400(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    payload["hours"][1]["hour"] = payload["hours"][0]["hour"]
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400


def test_too_many_notes_return_400(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    payload["operator_notes"] = ["a", "b", "c", "d"]
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400


def test_blank_note_returns_400(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    payload["operator_notes"] = ["   "]
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400


def test_negative_demand_returns_400(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    payload["hours"][0]["demand_kwh"] = -1
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 400


def test_semantically_invalid_battery_returns_422(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    payload["battery"]["initial_energy_kwh"] = payload["battery"]["capacity_kwh"] + 50
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "semantically_invalid_request"


def test_extra_request_fields_are_ignored(client: TestClient, cases) -> None:
    payload = sample_input(cases[0])
    payload["trace_id"] = "abc-123"
    response = client.post("/optimize-energy", json=payload)
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# safety
# ---------------------------------------------------------------------------


def test_error_bodies_leak_nothing_sensitive(client: TestClient) -> None:
    response = client.post(
        "/optimize-energy",
        content="{broken",
        headers={"Content-Type": "application/json"},
    )
    text = response.text.lower()
    for forbidden in ("traceback", "api_key", "authorization", "bearer", "file \""):
        assert forbidden not in text
