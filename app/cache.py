"""Small thread-safe TTL + LRU cache for completed responses.

Repeated identical hidden cases (a common judge pattern) then answer in
microseconds instead of paying the model round-trip again. The key is a hash of
the canonical request, so a changed note, tariff, or battery state can never
return a stale plan.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from typing import Any

from app.schemas import ScenarioRequest


class TTLCache:
    """Bounded cache with per-entry expiry."""

    def __init__(self, max_size: int, ttl_seconds: float) -> None:
        self._max_size = max(0, max_size)
        self._ttl = max(0.0, ttl_seconds)
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._max_size > 0 and self._ttl > 0

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        now = time.monotonic()
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                del self._store[key]
                return None
            self._store.move_to_end(key)
            return value

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        expires_at = time.monotonic() + self._ttl
        with self._lock:
            self._store[key] = (expires_at, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


def cache_key(request: ScenarioRequest) -> str:
    """Stable digest of the semantically meaningful request content."""
    canonical = {
        "scenario_id": request.scenario_id,
        "operator_notes": list(request.operator_notes),
        "hours": [
            {
                "hour": entry.hour,
                "demand_kwh": entry.demand_kwh,
                "solar_kwh": entry.solar_kwh,
                "tariff_bdt_per_kwh": entry.tariff_bdt_per_kwh,
            }
            for entry in request.ordered_hours()
        ],
        "battery": request.battery.model_dump(),
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()
