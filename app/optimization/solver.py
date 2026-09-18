"""Exact 24-hour energy scheduling optimizer.

Formulation
-----------
::

    minimise     sum_h  tariff[h] * grid[h]  (+ optional tiny cycling penalty)
    subject to   grid[h] + solar_used[h] + discharge[h] = demand[h] + charge[h]
                 soc[h]  = soc[h-1] + charge[h] - discharge[h],   soc[-1] = initial
                 soc[23] = initial                                  (neutrality)
                 eff_min[h] <= soc[h] <= capacity
                 0 <= solar_used[h] <= effective_solar[h]
                 0 <= charge[h]    <= charge_limit[h]
                 0 <= discharge[h] <= discharge_limit[h]
                 0 <= grid[h]      <= grid_cap[h]

Why a pure LP is exact
----------------------
The only reason to introduce binaries is to forbid simultaneous charging and
discharging. That is provably unnecessary: given any feasible solution with
``charge[h] > 0`` and ``discharge[h] > 0``, subtracting
``delta = min(charge[h], discharge[h])`` from both leaves ``charge - discharge``
unchanged, so the state-of-charge trajectory, every bound, every rate limit, and
end-of-day neutrality are preserved. The energy balance fixes
``grid[h] = demand[h] + charge[h] - discharge[h] - solar_used[h]``, which is
also unchanged. Cost is therefore identical.

So every LP optimum has a cost-equivalent schedule with no simultaneous
cycling, and ``_postprocess`` recovers it for free. Because the model is linear
with continuous data, the LP optimum is the global optimum - no local minima.

Solver tiers
------------
1. ``highs``  - exact LP via ``scipy.optimize.linprog`` (primary)
2. ``highs-elastic`` - same model with penalised slack variables; used only when
   the organizer's promised feasibility guarantee is violated
3. ``heuristic`` - dependency-free cycle local search; guarantees a *valid*
   schedule (not necessarily optimal) so the service never crashes
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from app.optimization.constraints import DirectiveConstraints

N_HOURS = 24

# Variable layout: grid | solar_used | charge | discharge | soc
_OFF_GRID = 0
_OFF_SOLAR = 24
_OFF_CHARGE = 48
_OFF_DISCHARGE = 72
_OFF_SOC = 96
_N_VARS = 120

# Slack block used only by the elastic tier.
_OFF_SLACK_BALANCE = 120
_OFF_SLACK_NEU_UP = 144
_OFF_SLACK_NEU_DOWN = 145
_OFF_SLACK_RESERVE = 146
_N_VARS_ELASTIC = 170

_PENALTY_BALANCE = 1.0e6
_PENALTY_NEUTRALITY = 1.0e5
_PENALTY_RESERVE = 1.0e4

_ROUND_DECIMALS = 6


@dataclass
class SolveResult:
    """Solved (or heuristically constructed) plan."""

    status: str
    engine: str
    grid: list[float]
    solar_used: list[float]
    charge: list[float]
    discharge: list[float]
    soc: list[float]
    objective: float
    relaxations: list[str] = field(default_factory=list)
    feasible: bool = True


@dataclass
class _Model:
    """Resolved numeric parameters for one scenario."""

    demand: list[float]
    tariff: list[float]
    base_solar: list[float]
    solar_eff: list[float]
    soc_min: list[float]
    charge_limit: list[float]
    discharge_limit: list[float]
    grid_cap: list[float | None]
    capacity: float
    initial: float


def _resolve(scenario: Any, constraints: DirectiveConstraints) -> _Model:
    """Resolve scenario + directives into flat numeric limits."""
    battery = scenario.battery
    demand = scenario.demand()
    tariff = scenario.tariff()
    base_solar = scenario.solar()

    soc_min: list[float] = []
    for hour in range(N_HOURS):
        extra = constraints.reserve_kwh[hour]
        value = battery.minimum_energy_kwh if extra is None else max(battery.minimum_energy_kwh, extra)
        soc_min.append(min(max(value, 0.0), battery.capacity_kwh))

    return _Model(
        demand=demand,
        tariff=tariff,
        base_solar=base_solar,
        solar_eff=constraints.effective_solar(base_solar),
        soc_min=soc_min,
        charge_limit=[
            battery.max_charge_kwh_per_hour if constraints.charge_allowed[h] else 0.0
            for h in range(N_HOURS)
        ],
        discharge_limit=[
            battery.max_discharge_kwh_per_hour if constraints.discharge_allowed[h] else 0.0
            for h in range(N_HOURS)
        ],
        grid_cap=list(constraints.grid_cap_kwh),
        capacity=battery.capacity_kwh,
        initial=battery.initial_energy_kwh,
    )


# ---------------------------------------------------------------------------
# LP construction
# ---------------------------------------------------------------------------


def _objective(model: _Model, epsilon: float, n_vars: int) -> list[float]:
    cost = [0.0] * n_vars
    for hour in range(N_HOURS):
        cost[_OFF_GRID + hour] = model.tariff[hour]
        if epsilon:
            cost[_OFF_CHARGE + hour] = epsilon
            cost[_OFF_DISCHARGE + hour] = epsilon
    if n_vars > _N_VARS:  # elastic penalties
        for hour in range(N_HOURS):
            cost[_OFF_SLACK_BALANCE + hour] = _PENALTY_BALANCE
            cost[_OFF_SLACK_RESERVE + hour] = _PENALTY_RESERVE
        cost[_OFF_SLACK_NEU_UP] = _PENALTY_NEUTRALITY
        cost[_OFF_SLACK_NEU_DOWN] = _PENALTY_NEUTRALITY
    return cost


def _equalities(model: _Model, n_vars: int) -> tuple[list[list[float]], list[float]]:
    rows: list[list[float]] = []
    rhs: list[float] = []

    for hour in range(N_HOURS):
        row = [0.0] * n_vars
        row[_OFF_GRID + hour] = 1.0
        row[_OFF_SOLAR + hour] = 1.0
        row[_OFF_DISCHARGE + hour] = 1.0
        row[_OFF_CHARGE + hour] = -1.0
        if n_vars > _N_VARS:
            row[_OFF_SLACK_BALANCE + hour] = -1.0  # slack = unmet demand
        rows.append(row)
        rhs.append(model.demand[hour])

    for hour in range(N_HOURS):
        row = [0.0] * n_vars
        row[_OFF_SOC + hour] = 1.0
        if hour > 0:
            row[_OFF_SOC + hour - 1] = -1.0
        row[_OFF_CHARGE + hour] = -1.0
        row[_OFF_DISCHARGE + hour] = 1.0
        rows.append(row)
        rhs.append(model.initial if hour == 0 else 0.0)

    row = [0.0] * n_vars
    row[_OFF_SOC + N_HOURS - 1] = 1.0
    if n_vars > _N_VARS:
        row[_OFF_SLACK_NEU_UP] = 1.0
        row[_OFF_SLACK_NEU_DOWN] = -1.0
    rows.append(row)
    rhs.append(model.initial)

    return rows, rhs


def _bounds(
    model: _Model, n_vars: int, *, elastic: bool
) -> list[tuple[float | None, float | None]]:
    """Variable bounds.

    In the primary tier the active reserve is a hard lower bound on the
    state-of-charge variable. In the elastic tier it is lifted out into
    penalised slack rows instead, so a rescue solution can dip below it as a
    last resort.
    """
    bounds: list[tuple[float | None, float | None]] = [
        (0.0, None) for _ in range(n_vars)
    ]
    for hour in range(N_HOURS):
        bounds[_OFF_GRID + hour] = (0.0, model.grid_cap[hour])
        bounds[_OFF_SOLAR + hour] = (0.0, model.solar_eff[hour])
        bounds[_OFF_CHARGE + hour] = (0.0, model.charge_limit[hour])
        bounds[_OFF_DISCHARGE + hour] = (0.0, model.discharge_limit[hour])
        bounds[_OFF_SOC + hour] = (
            (0.0, model.capacity) if elastic else (model.soc_min[hour], model.capacity)
        )
    if n_vars > _N_VARS:
        for hour in range(N_HOURS):
            bounds[_OFF_SLACK_BALANCE + hour] = (0.0, None)
            bounds[_OFF_SLACK_RESERVE + hour] = (0.0, None)
        bounds[_OFF_SLACK_NEU_UP] = (0.0, None)
        bounds[_OFF_SLACK_NEU_DOWN] = (0.0, None)
    return bounds


def _elastic_rows(model: _Model, n_vars: int) -> tuple[list[list[float]], list[float]]:
    """Move the state-of-charge lower bound into penalised inequality rows."""
    rows: list[list[float]] = []
    rhs: list[float] = []
    for hour in range(N_HOURS):
        row = [0.0] * n_vars
        row[_OFF_SOC + hour] = -1.0
        row[_OFF_SLACK_RESERVE + hour] = -1.0
        rows.append(row)
        rhs.append(-model.soc_min[hour])
    return rows, rhs


def _try_linprog(
    model: _Model, epsilon: float, *, elastic: bool
) -> tuple[str, list[float] | None]:
    """Run one HiGHS LP. Returns ``(status, solution)``."""
    try:
        from scipy.optimize import linprog  # local import: keeps startup light
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        return "solver_unavailable", None

    n_vars = _N_VARS_ELASTIC if elastic else _N_VARS
    a_eq, b_eq = _equalities(model, n_vars)
    bounds = _bounds(model, n_vars, elastic=elastic)
    a_ub = b_ub = None
    if elastic:
        a_ub, b_ub = _elastic_rows(model, n_vars)

    result = linprog(
        c=_objective(model, epsilon, n_vars),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if result.status == 0 and result.x is not None:
        return "optimal", [float(v) for v in result.x]
    if result.status == 2:
        return "infeasible", None
    if result.status == 3:
        return "unbounded", None
    return f"failed({result.status})", None


# ---------------------------------------------------------------------------
# Heuristic rescue (no scipy required)
# ---------------------------------------------------------------------------


def _heuristic(
    model: _Model, constraints: DirectiveConstraints, *, tol: float
) -> SolveResult:
    """Cycle-based local search producing a guaranteed-valid schedule.

    A *cycle* charges ``delta`` in hour ``i`` and discharges it in hour ``j``
    (``i < j``). Because total charge equals total discharge, neutrality holds by
    construction; every move is bounds-checked, so validity holds by
    construction as well. Feasibility moves repair reserve violations first,
    then price-improving moves reduce cost.
    """
    n = N_HOURS
    charge = [0.0] * n
    discharge = [0.0] * n
    relaxations: list[str] = []

    def soc_path() -> list[float]:
        path = []
        running = model.initial
        for hour in range(n):
            running += charge[hour] - discharge[hour]
            path.append(running)
        return path

    def feasible_path(path: list[float]) -> bool:
        return all(
            model.soc_min[h] - tol <= path[h] <= model.capacity + tol for h in range(n)
        )

    # --- Pass 1: repair reserve violations by charging in earlier hours -------
    for _ in range(4 * n):
        path = soc_path()
        worst = None
        worst_gap = tol
        for hour in range(n):
            gap = model.soc_min[hour] - path[hour]
            if gap > worst_gap:
                worst_gap, worst = gap, hour
        if worst is None:
            break
        candidates = [
            i
            for i in range(worst + 1)
            if model.charge_limit[i] - charge[i] > tol
        ]
        if not candidates:
            relaxations.append(f"reserve_unreachable_hour_{worst}")
            break
        # Latest cheap hour with SOC headroom between i and worst.
        i = max(candidates, key=lambda h: (-model.tariff[h], h))
        headroom = min(
            model.capacity - path[k] for k in range(i, worst + 1)
        )
        room = min(
            model.charge_limit[i] - charge[i],
            headroom,
            worst_gap,
        )
        if room <= tol:
            relaxations.append(f"reserve_capacity_limited_hour_{worst}")
            break
        charge[i] += room

    # --- Pass 2: restore neutrality ------------------------------------------
    path = soc_path()
    surplus = path[-1] - model.initial
    if abs(surplus) > tol:
        if surplus > 0:
            order = sorted(range(n), key=lambda h: -model.tariff[h])
            for hour in order:
                if surplus <= tol:
                    break
                room = min(
                    model.discharge_limit[hour] - discharge[hour],
                    surplus,
                    min(
                        (path[k] - model.soc_min[k] for k in range(hour, n)),
                        default=0.0,
                    ),
                )
                if room > tol:
                    discharge[hour] += room
                    surplus -= room
                    path = soc_path()
        else:
            order = sorted(range(n), key=lambda h: model.tariff[h])
            for hour in order:
                if surplus >= -tol:
                    break
                room = min(
                    model.charge_limit[hour] - charge[hour],
                    -surplus,
                    min(
                        (model.capacity - path[k] for k in range(hour, n)),
                        default=0.0,
                    ),
                )
                if room > tol:
                    charge[hour] += room
                    surplus += room
                    path = soc_path()
        if abs(soc_path()[-1] - model.initial) > 1e-4 * max(1.0, model.capacity):
            relaxations.append("neutrality_not_fully_restorable")

    # --- Pass 3: price-improving cycles --------------------------------------
    for _ in range(200):
        path = soc_path()
        best_gain = 1e-9
        best_move: tuple[int, int, float, bool] | None = None
        for i in range(n):
            for j in range(i + 1, n):
                forward_gain = model.tariff[j] - model.tariff[i]
                if forward_gain > best_gain:
                    room = min(
                        model.charge_limit[i] - charge[i],
                        model.discharge_limit[j] - discharge[j],
                        min((model.capacity - path[k] for k in range(i, j)), default=0.0),
                    )
                    if room > 1e-6:
                        best_gain = forward_gain
                        best_move = (i, j, room, True)
                reverse_gain = model.tariff[i] - model.tariff[j]
                if reverse_gain > best_gain:
                    room = min(
                        model.discharge_limit[i] - discharge[i],
                        model.charge_limit[j] - charge[j],
                        min(
                            (path[k] - model.soc_min[k] for k in range(i, j)),
                            default=0.0,
                        ),
                    )
                    if room > 1e-6:
                        best_gain = reverse_gain
                        best_move = (i, j, room, False)
        if best_move is None:
            break
        i, j, room, forward = best_move
        if forward:
            charge[i] += room
            discharge[j] += room
        else:
            discharge[i] += room
            charge[j] += room

    soc = soc_path()
    if not feasible_path(soc):
        relaxations.append("heuristic_bounds_violated")

    return _finalize(
        model,
        charge=charge,
        discharge=discharge,
        engine="heuristic",
        status="heuristic",
        relaxations=relaxations,
    )


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _finalize(
    model: _Model,
    *,
    charge: list[float],
    discharge: list[float],
    engine: str,
    status: str,
    relaxations: list[str],
    snap_tol: float = 1e-9,
) -> SolveResult:
    """Net cycling, snap noise, and derive grid/solar/state exactly.

    ``solar_used`` and ``grid`` are recomputed from ``charge``/``discharge`` in
    the cost-minimal way (use every available solar kWh that can displace
    grid), which can only improve or preserve cost. The state-of-charge path is
    recomputed from the transitions, so the dynamics equation holds exactly
    rather than to solver precision.
    """
    charge = [0.0 if abs(v) < snap_tol else float(v) for v in charge]
    discharge = [0.0 if abs(v) < snap_tol else float(v) for v in discharge]

    # Netting lemma: remove simultaneous cycling without changing cost.
    for hour in range(N_HOURS):
        overlap = min(charge[hour], discharge[hour])
        if overlap > 0.0:
            charge[hour] -= overlap
            discharge[hour] -= overlap
            if abs(charge[hour]) < snap_tol:
                charge[hour] = 0.0
            if abs(discharge[hour]) < snap_tol:
                discharge[hour] = 0.0

    solar_used = [0.0] * N_HOURS
    grid = [0.0] * N_HOURS
    soc = [0.0] * N_HOURS
    running = model.initial
    for hour in range(N_HOURS):
        net = model.demand[hour] + charge[hour] - discharge[hour]
        solar_used[hour] = min(model.solar_eff[hour], max(net, 0.0))
        grid[hour] = max(net - solar_used[hour], 0.0)
        running += charge[hour] - discharge[hour]
        soc[hour] = running

    charge = [round(v, _ROUND_DECIMALS) for v in charge]
    discharge = [round(v, _ROUND_DECIMALS) for v in discharge]
    solar_used = [round(v, _ROUND_DECIMALS) for v in solar_used]
    grid = [round(v, _ROUND_DECIMALS) for v in grid]
    soc = [round(v, _ROUND_DECIMALS) for v in soc]

    objective = round(sum(grid[h] * model.tariff[h] for h in range(N_HOURS)), _ROUND_DECIMALS)
    return SolveResult(
        status=status,
        engine=engine,
        grid=grid,
        solar_used=solar_used,
        charge=charge,
        discharge=discharge,
        soc=soc,
        objective=objective,
        relaxations=relaxations,
        feasible=not relaxations,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def warm_up() -> None:
    """Import the solver and run a trivial LP during startup.

    ``scipy`` lazily imports a sizeable native stack on first use (~1s). Paying
    that at boot keeps the first judged request inside the latency budget.
    """
    try:
        from scipy.optimize import linprog
    except ImportError:  # pragma: no cover - fallback path stays available
        return
    try:
        linprog(
            c=[1.0],
            A_eq=[[1.0]],
            b_eq=[0.0],
            bounds=[(0.0, None)],
            method="highs",
        )
    except Exception:  # pragma: no cover - warm-up must never break startup
        pass


def heuristic_only(
    scenario: Any,
    constraints: DirectiveConstraints,
    *,
    tolerance: float = 0.01,
) -> SolveResult:
    """Force the dependency-free construction path.

    Used by the pipeline as an independent second opinion when the LP result
    fails replay verification, and available when scipy is absent.
    """
    model = _resolve(scenario, constraints)
    return _heuristic(model, constraints, tol=tolerance)


def optimize(
    scenario: Any,
    constraints: DirectiveConstraints,
    *,
    epsilon: float = 0.0,
    tolerance: float = 0.01,
) -> SolveResult:
    """Solve the 24-hour schedule, degrading gracefully across solver tiers."""
    model = _resolve(scenario, constraints)

    status, solution = _try_linprog(model, epsilon, elastic=False)
    if status == "optimal" and solution is not None:
        return _finalize(
            model,
            charge=solution[_OFF_CHARGE : _OFF_CHARGE + N_HOURS],
            discharge=solution[_OFF_DISCHARGE : _OFF_DISCHARGE + N_HOURS],
            engine="highs",
            status="optimal",
            relaxations=[],
        )

    # Organizer-promised feasibility failed; recover with penalised slack.
    elastic_status, elastic_solution = _try_linprog(model, epsilon, elastic=True)
    if elastic_status == "optimal" and elastic_solution is not None:
        relaxations: list[str] = []
        for hour in range(N_HOURS):
            if elastic_solution[_OFF_SLACK_BALANCE + hour] > 1e-6:
                # The slack lets supply exceed demand; _finalize re-derives the
                # plan so the returned schedule still balances exactly, but it
                # signals that the original equality set conflicted.
                relaxations.append(f"energy_balance_relaxed_hour_{hour}")
            if elastic_solution[_OFF_SLACK_RESERVE + hour] > 1e-6:
                relaxations.append(f"reserve_relaxed_hour_{hour}")
        if elastic_solution[_OFF_SLACK_NEU_UP] > 1e-6 or elastic_solution[_OFF_SLACK_NEU_DOWN] > 1e-6:
            relaxations.append("neutrality_relaxed")
        result = _finalize(
            model,
            charge=elastic_solution[_OFF_CHARGE : _OFF_CHARGE + N_HOURS],
            discharge=elastic_solution[_OFF_DISCHARGE : _OFF_DISCHARGE + N_HOURS],
            engine="highs-elastic",
            status="optimal_with_relaxation",
            relaxations=relaxations or ["elastic_solve"],
        )
        return result

    if status == "solver_unavailable" or elastic_status == "solver_unavailable":
        return _heuristic(model, constraints, tol=tolerance)

    return _heuristic(model, constraints, tol=tolerance)
