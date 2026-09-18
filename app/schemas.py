"""Request and response models for the exact Problem Statement contract.

Field names, types, and enum values are fixed by the canonical specification.
Extra keys in a request are ignored rather than rejected so a judge harness can
add trace fields without triggering a 400.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_LENIENT = ConfigDict(extra="ignore", allow_inf_nan=False)
_STRICT = ConfigDict(extra="forbid", allow_inf_nan=False)


class DirectiveType(str, Enum):
    """The six accepted directive types. No other value is legal."""

    SOLAR_REDUCTION = "solar_reduction"
    MINIMUM_BATTERY_RESERVE = "minimum_battery_reserve"
    NO_CHARGE_WINDOW = "no_charge_window"
    NO_DISCHARGE_WINDOW = "no_discharge_window"
    MAX_GRID_WINDOW = "max_grid_window"
    NO_OP = "no_op"


class BatteryAction(str, Enum):
    """Exactly one action is reported per hour."""

    CHARGE = "charge"
    DISCHARGE = "discharge"
    IDLE = "idle"


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class HourEntry(BaseModel):
    """One hourly scenario row."""

    model_config = _LENIENT

    hour: int = Field(..., ge=0, le=23, description="Unique integer 0-23.")
    demand_kwh: float = Field(..., ge=0, description="Must be supplied this hour.")
    solar_kwh: float = Field(..., ge=0, description="Base solar before directives.")
    tariff_bdt_per_kwh: float = Field(..., ge=0, description="Grid price.")


class BatterySpec(BaseModel):
    """Battery capability and operating limits."""

    model_config = _LENIENT

    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)


class ScenarioRequest(BaseModel):
    """`POST /optimize-energy` request body."""

    model_config = _LENIENT

    scenario_id: str
    operator_notes: list[str] = Field(..., min_length=1, max_length=3)
    hours: list[HourEntry] = Field(..., min_length=24, max_length=24)
    battery: BatterySpec

    @field_validator("operator_notes")
    @classmethod
    def _notes_non_empty(cls, value: list[str]) -> list[str]:
        cleaned = [note.strip() for note in value]
        if any(not note for note in cleaned):
            raise ValueError("operator_notes must contain non-empty strings")
        return cleaned

    @model_validator(mode="after")
    def _hours_complete_and_unique(self) -> "ScenarioRequest":
        seen = {entry.hour for entry in self.hours}
        if len(seen) != len(self.hours):
            raise ValueError("hours must not repeat an hour value")
        missing = sorted(set(range(24)) - seen)
        if missing:
            raise ValueError(f"hours must cover 0-23 exactly; missing {missing}")
        return self

    def ordered_hours(self) -> list[HourEntry]:
        """Scenario rows sorted by hour; the internal canonical order."""
        return sorted(self.hours, key=lambda entry: entry.hour)

    def demand(self) -> list[float]:
        return [entry.demand_kwh for entry in self.ordered_hours()]

    def solar(self) -> list[float]:
        return [entry.solar_kwh for entry in self.ordered_hours()]

    def tariff(self) -> list[float]:
        return [entry.tariff_bdt_per_kwh for entry in self.ordered_hours()]


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class DirectiveInterpretation(BaseModel):
    """One machine-checkable interpretation entry per operator note."""

    model_config = _STRICT

    note_index: int = Field(..., ge=0)
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: dict[str, Any] | None
    explanation: str


class HourlyPlanEntry(BaseModel):
    """One hour of the final operating plan."""

    model_config = _STRICT

    hour: int = Field(..., ge=0, le=23)
    grid_kwh: float = Field(..., ge=0)
    solar_used_kwh: float = Field(..., ge=0)
    battery_action: BatteryAction
    battery_kwh: float = Field(..., ge=0)
    battery_energy_after_kwh: float = Field(..., ge=0)


class OptimizeResponse(BaseModel):
    """`POST /optimize-energy` success body."""

    model_config = _STRICT

    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


class HealthResponse(BaseModel):
    """`GET /health` body."""

    model_config = _LENIENT

    status: str = "ok"


def semantic_problems(request: ScenarioRequest) -> list[str]:
    """Cross-field problems that are well-formed but semantically unusable.

    These map to HTTP 422 in the Problem Statement's status table.
    """
    problems: list[str] = []
    battery = request.battery
    if battery.initial_energy_kwh > battery.capacity_kwh + 1e-9:
        problems.append("battery.initial_energy_kwh must not exceed capacity_kwh")
    if battery.minimum_energy_kwh > battery.capacity_kwh + 1e-9:
        problems.append("battery.minimum_energy_kwh must not exceed capacity_kwh")
    return problems
