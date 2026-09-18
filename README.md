# GridWise — LLM-Assisted Operator Directive Interpretation & 24-Hour Energy Optimization

FastAPI service for the **BUP CSE Fest 2026 · Smart Campus Energy Optimization Challenge**.

It receives a 24-hour campus energy scenario plus 1–3 free-text operator notes, interprets those
notes with a language model into a fixed set of structured directives, validates the interpretation
through deterministic guardrails, applies the directives as hard constraints to an exact linear
program, and returns a **provably cost-optimal, rule-valid** hourly operating plan.

---

## Table of contents

1. [Quickstart](#1-quickstart)
2. [Architecture](#2-architecture)
3. [The LLM's role](#3-the-llms-role)
4. [Guardrails](#4-guardrails)
5. [Optimizer](#5-optimizer)
6. [API reference](#6-api-reference)
7. [Configuration](#7-configuration)
8. [Local validation and tests](#8-local-validation-and-tests)
9. [Project layout](#9-project-layout)
10. [Design decisions](#10-design-decisions)
11. [Known limitations](#11-known-limitations)
12. [Security and secret handling](#12-security-and-secret-handling)
13. [Deployment status](#13-deployment-status)

---

## 1. Quickstart

Copy-paste from a clean environment. Requires **Python 3.10+**.

```bash
# 1. clone / pull the repository, then:
cd Bup_soln

# 2. create an isolated environment and install dependencies
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt

# 3. configure environment variables (names only - see section 7)
cp .env.example .env          # Windows: copy .env.example .env
# edit .env and set LLM_API_KEY

# 4. start the service
python -m app.main
# or, equivalently:
# uvicorn app.main:app --host 0.0.0.0 --port 8000

# 5. verify readiness
curl http://127.0.0.1:8000/health
# -> {"status":"ok"}
```

Run one public sample case end to end:

```bash
# write any public case's request body to a file
python tools/make_sample.py 9 sample_s10.json

curl -X POST http://127.0.0.1:8000/optimize-energy \
  -H "Content-Type: application/json" \
  --data @sample_s10.json
```

> **No API key?** The service still starts and answers correctly using the offline deterministic
> interpreter (section 3). The live model is required for the judged interpretation path, but the
> optimizer, guardrails, and verifier are all exercisable without one.

---

## 2. Architecture

```
                    POST /optimize-energy
                             │
                  ┌──────────▼───────────┐
                  │  Pydantic validation │  L0  structural (400 / 422)
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  LLM interpretation  │  free text -> structured directives
                  └──────────┬───────────┘
                             │  raw model text (untrusted)
                  ┌──────────▼───────────┐
                  │   Deterministic      │  L1 parse  L2 shape
                  │   guardrails         │  L3 normalize  L4 policy
                  └──────────┬───────────┘
                             │  retry once with error feedback, else rule fallback
                             │
                  ┌──────────▼───────────┐
                  │ compile directives   │  -> per-hour factor/reserve/limit tables
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  LP solve (HiGHS)    │  120 variables, 49 rows, ~5 ms
                  └──────────┬───────────┘
                             │
                  ┌──────────▼───────────┐
                  │  L6 replay verifier  │  independent hour-by-hour re-check
                  └──────────┬───────────┘
                             │  fail -> heuristic second opinion
                  ┌──────────▼───────────┐
                  │  L7 recompute totals │  totals always match hourly_plan
                  └──────────┬───────────┘
                             ▼
                       Structured JSON
```

The central design rule: **the optimizer never reads English, and the verifier never trusts the
optimizer.** Those are two separate code paths on purpose — if they shared an implementation bug,
the check would be worthless.

> **Full architecture:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) has six Mermaid diagrams
> (pipeline, request lifecycle, recovery ladder, English→math transformation, module map, deployment
> topology), the design invariants, and a **3-minute video narration script** with timings.
> Open [`docs/architecture.html`](docs/architecture.html) in any browser for a self-contained,
> CDN-free diagram — no network required, so it cannot fail mid-recording.

---

## 3. The LLM's role

The operator notes are interpreted by a **language-capable generative model that produces the
structured directives the optimizer consumes**. This satisfies the mandatory requirement that the
model be part of the interpretation path — using AI only for `plan_summary` or documentation would
not.

| Item | Value |
|---|---|
| Provider | Any OpenAI-compatible `/chat/completions` endpoint |
| Default | **DeepSeek** — `https://api.deepseek.com/v1`, model `deepseek-chat` |
| Also works with | OpenAI, Groq, Together, OpenRouter, vLLM, Ollama, LM Studio |
| Output mode | JSON object (`response_format: {"type": "json_object"}`), temperature `0` |
| Prompt location | `app/llm/prompts.py` |

### What the prompt teaches

* the **six** legal directive types and their exact `structured_adjustment` shapes;
* **start-inclusive / end-exclusive** whole-hour windows
  (`1 PM to 3 PM` → `[13, 14]`, `noon until 2 PM` → `[12, 13]`, `6 PM until 9 PM` → `[18, 19, 20]`);
* the two **opposite** percentage phrasings:
  `"25% of the forecast"` → `factor = 0.25`, but `"an 80% reduction"` → `factor = 0.2`;
* relative reserves resolved against the request's `capacity_kwh`
  (`"50% of battery capacity"` on a 200 kWh battery → `100`);
* distractor handling — administrative, menu, booking, sports, library, and seminar notes are `no_op`;
* one entry per note, `note_index` order, no gaps or duplicates.

The few-shot examples deliberately use **wording that does not appear in the public sample pack**,
so the model learns the semantics rather than memorising public phrases. Hidden cases paraphrase,
and hard-coding surface strings would not generalise.

### Prompt injection resistance

Model output is *never* executed and *never* trusted. It is parsed as JSON, coerced, range-checked,
and enum-checked before it can influence the optimization model, so a note that tries to instruct
the model to emit a rogue directive type is simply rejected by the guardrails.

### Recovery ladder

```
LLM call ──▶ guardrails ──▶ (fail) retry once, quoting the validation errors back
                              │
                              └▶ (still failing) deterministic rule-based interpreter per note
                                    │
                                    └▶ (no window found) no_op — never an invented rule
```

---

## 4. Guardrails

Model output is untrusted structured data until every layer passes.

| Layer | Responsibility | On failure |
|---|---|---|
| **L0** Transport | Pydantic: 24 unique hours 0–23, 1–3 non-empty notes, finite non-negative numerics | `400` structural, `422` semantic |
| **L1** Parse | Extract JSON from fenced/prose-wrapped output; string-aware brace matching | retry with feedback |
| **L2** Structure | `directive_type` ∈ enum; `structured_adjustment` shape per type | entry invalidated → fallback |
| **L3** Normalize | sort/dedupe hours; clamp `factor` → [0,1]; clamp reserve → [0, capacity]; drop negatives | silent repair, logged |
| **L4** Policy | `no_op` ⇒ `applies=false` + `null`; everything else ⇒ `applies=true` + object; exactly one entry per note in index order | enforced |
| **L5** Apply | Compile directives into per-hour numeric tables | — |
| **L6** Replay | Re-derive every rule from `(request, interpretation, plan)` and re-check the final plan hour by hour | re-solve, then heuristic |
| **L7** Consistency | Recompute `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` from `hourly_plan` | always |

Repair is deliberately split: *sloppy* values are fixed (unsorted hours, `factor = 1.5`, a reserve
above capacity, `"0.2"`-as-string), while *structurally wrong* values are rejected outright.
Silently guessing a constraint is worse than admitting the entry is unusable.

Repeated directive types inside one scenario are merged deterministically:
`solar_reduction` factors multiply, reserves take the maximum, grid caps take the minimum, and
charge/discharge outage hours union.

### Replay (L6) checks

`hourly_plan` length and hour coverage · finite non-negative values · action consistency
(`idle` ⇒ `battery_kwh == 0`) · charge and discharge rate limits including directive-forced zeros ·
energy balance · effective-solar cap after `solar_reduction` · state-of-charge transitions ·
SOC bounds including directive-raised reserve · grid cap · end-of-day neutrality · every
directive window · recomputed totals.

---

## 5. Optimizer

### Formulation

For `h = 0 … 23`, with demand `D`, base solar `S`, tariff `T`, capacity `Ecap`, initial energy
`E0`, base reserve `Emin`, and charge/discharge rates `R+` / `R−`:

Directives become parameters:

```
S_eff[h] = S[h] * factor[h]                       (solar_reduction)
Emin_eff[h] = max(Emin, directive_reserve[h])     (minimum_battery_reserve)
R+[h] = 0                                         (no_charge_window)
R−[h] = 0                                         (no_discharge_window)
Gcap[h] = directive_cap                           (max_grid_window)
```

The linear program:

```
minimise    Σ_h  T[h] * grid[h]

subject to  grid[h] + solar_used[h] + discharge[h] = demand[h] + charge[h]
            soc[h]  = soc[h-1] + charge[h] - discharge[h],   soc[-1] = E0
            soc[23] = E0                                     (end-of-day neutrality)
            Emin_eff[h] <= soc[h] <= Ecap
            0 <= solar_used[h] <= S_eff[h]
            0 <= charge[h]    <= R+[h]
            0 <= discharge[h] <= R-[h]
            0 <= grid[h]      <= Gcap[h]
            all variables continuous and >= 0
```

**120 variables, 25 equality rows.** Solved by HiGHS in single-digit milliseconds.

### Why a pure LP is exact (no binaries)

The only reason to add binaries is to forbid simultaneous charging and discharging. That is
provably unnecessary:

> **Netting lemma.** If a feasible solution has `charge[h] > 0` and `discharge[h] > 0`, subtract
> `δ = min(charge[h], discharge[h])` from both. Then `charge[h] - discharge[h]` is unchanged, so the
> SOC trajectory, every bound, every rate limit, and end-of-day neutrality are preserved. The energy
> balance fixes `grid[h] = demand[h] + charge[h] - discharge[h] - solar_used[h]`, which is also
> unchanged — so cost is identical.

Every LP optimum therefore has a cost-equivalent cycling-free schedule, recovered for free in
post-processing. Combined with a linear objective and continuous data, **the LP optimum is the
global optimum** — no local minima, no heuristic uncertainty, and we never need to beat a MILP's
runtime.

### Solver tiers

| Tier | Engine | Role |
|---|---|---|
| 1 | `highs` | Exact LP via `scipy.optimize.linprog`. The normal path. |
| 2 | `highs-elastic` | Same model with penalised slack variables (balance ≫ neutrality ≫ reserve). Used only if the organizer's feasibility guarantee is violated. |
| 3 | `heuristic` | Dependency-free cycle local search. Guarantees a **valid** schedule (not necessarily optimal) so the service never crashes. |

### Post-processing

1. **Snap** solver noise (`|x| < 1e-9 → 0`) so `battery_action` is unambiguous.
2. **Net** simultaneous charge/discharge without changing cost.
3. **Re-derive** `solar_used` and `grid` from the charge/discharge plan in the cost-minimal way.
4. **Recompute** the SOC path from the transitions so `soc[h] = soc[h-1] + charge - discharge`
   holds *exactly* rather than to solver precision.
5. **Round** to 6 decimals and recompute all reported totals *from the plan*, so the totals can
   never disagree with `hourly_plan` — a listed critical violation.

Steps 3–4 carry a useful guarantee: because the re-derivation uses at least as much free solar as
the LP did, the recomputed grid can only be **lower**, so directive grid caps are preserved and the
reported cost can only improve.

---

## 6. API reference

### `GET /health`

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok"}
```

Returns `200` with `status = "ok"` once the service is ready.

### `POST /optimize-energy`

```bash
curl -X POST http://127.0.0.1:8000/optimize-energy \
  -H "Content-Type: application/json" \
  -d '{
    "scenario_id": "GRID-101",
    "operator_notes": [
      "Solar output will drop to about 20% from 1 PM to 3 PM.",
      "Do not charge the battery between 2 PM and 4 PM.",
      "The cafeteria menu changes tomorrow."
    ],
    "hours": [
      {"hour": 0, "demand_kwh": 180, "solar_kwh": 0, "tariff_bdt_per_kwh": 7},
      "... 22 more hourly entries ...",
      {"hour": 23, "demand_kwh": 200, "solar_kwh": 0, "tariff_bdt_per_kwh": 9}
    ],
    "battery": {
      "capacity_kwh": 500,
      "initial_energy_kwh": 200,
      "minimum_energy_kwh": 50,
      "max_charge_kwh_per_hour": 100,
      "max_discharge_kwh_per_hour": 100
    }
  }'
```

**Request fields**

| Field | Type | Notes |
|---|---|---|
| `scenario_id` | string | Echoed back in the response |
| `operator_notes` | array[1..3] of string | Non-empty after trimming |
| `hours` | array[24] | Must cover each hour 0–23 exactly once. **Any input order is accepted** and normalised internally — see the note below |
| `hours[].hour` | int 0–23 | |
| `hours[].demand_kwh` | number ≥ 0 | |
| `hours[].solar_kwh` | number ≥ 0 | Base solar before directives |
| `hours[].tariff_bdt_per_kwh` | number ≥ 0 | |
| `battery.capacity_kwh` | number > 0 | |
| `battery.initial_energy_kwh` | number ≥ 0, ≤ capacity | |
| `battery.minimum_energy_kwh` | number ≥ 0, ≤ capacity | |
| `battery.max_charge_kwh_per_hour` | number ≥ 0 | |
| `battery.max_discharge_kwh_per_hour` | number ≥ 0 | |

Extra top-level keys in the request are ignored, so a harness may attach trace fields safely.

> **Why a shuffled `hours` array is accepted, not rejected.** Section 07 requires the request to
> contain *"exactly 24 entries for hours 0 through 23"* and imposes no ordering. Ascending order is
> demanded only of the **response** — `structured_adjustment.hours` (§5.1) and the hours we return
> (§08). Rejecting a shuffled request would fail a legal harness case, so the service sorts
> internally and always returns `hourly_plan` in ascending order. A regression test asserts that
> reversed, swapped and rotated inputs all yield the identical optimal cost.

**Response fields**

| Field | Type | Notes |
|---|---|---|
| `scenario_id` | string | Must match the request |
| `directive_interpretation` | array | One entry per note, in `note_index` order |
| `directive_interpretation[].note_index` | int | Zero-based |
| `directive_interpretation[].applies` | bool | `false` only for `no_op` |
| `directive_interpretation[].directive_type` | enum | One of the six legal types |
| `directive_interpretation[].structured_adjustment` | object \| null | `null` only for `no_op` |
| `directive_interpretation[].explanation` | string | Free text; not byte-matched by the judge |
| `hourly_plan` | array[24] | One entry per hour |
| `hourly_plan[].hour` | int 0–23 | |
| `hourly_plan[].grid_kwh` | number ≥ 0 | |
| `hourly_plan[].solar_used_kwh` | number ≥ 0 | ≤ effective solar |
| `hourly_plan[].battery_action` | `charge` \| `discharge` \| `idle` | |
| `hourly_plan[].battery_kwh` | number ≥ 0 | Magnitude; `0` when idle |
| `hourly_plan[].battery_energy_after_kwh` | number ≥ 0 | SOC after the hour |
| `total_grid_kwh` | number | Σ `grid_kwh`, recomputed from the plan |
| `total_cost_bdt` | number | Σ `grid_kwh × tariff`, recomputed from the plan |
| `peak_grid_kwh` | number | max `grid_kwh`, recomputed from the plan |
| `plan_summary` | string | Deterministic human-readable strategy |

**Directive types**

| `directive_type` | `structured_adjustment` |
|---|---|
| `solar_reduction` | `{"hours": [...], "factor": number}` — `factor` is the usable fraction **remaining** |
| `minimum_battery_reserve` | `{"hours": [...], "minimum_energy_kwh": number}` |
| `no_charge_window` | `{"hours": [...]}` |
| `no_discharge_window` | `{"hours": [...]}` |
| `max_grid_window` | `{"hours": [...], "max_grid_kwh": number}` |
| `no_op` | `null` |

**Status codes**

| Code | Meaning |
|---|---|
| `200` | Success |
| `400` | Malformed JSON or structurally invalid request (missing field, wrong hour count, duplicate hours, 4+ notes, blank note, negative demand) |
| `422` | Well-formed but **provably infeasible**: `initial_energy_kwh` above `capacity_kwh`, `minimum_energy_kwh` above `capacity_kwh`, or `initial_energy_kwh` below `minimum_energy_kwh` |
| `500` | Controlled internal error — generic message only |

Error bodies are always `{"error": {"code": "...", "message": "..."}}`.

---

## 7. Configuration

All configuration is environment-driven. Variable **names** documented here are part of the
submission contract; secret **values** are never committed.

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `LLM_BASE_URL` | `https://api.deepseek.com/v1` | OpenAI-compatible base URL |
| `LLM_API_KEY` | *(unset)* | **Secret.** Provider credential |
| `LLM_MODEL` | `deepseek-chat` | Model identifier |
| `LLM_TEMPERATURE` | `0` | Sampling temperature |
| `LLM_MAX_TOKENS` | `1600` | Response cap |
| `LLM_TIMEOUT_SECONDS` | `18` | Per-attempt timeout |
| `LLM_MAX_RETRIES` | `2` | Transport retries (429/5xx only) |
| `LLM_JSON_MODE` | `true` | Request a JSON object response |
| `LLM_MAX_CONCURRENCY` | `8` | In-flight model calls |
| `ALLOW_DETERMINISTIC_FALLBACK` | `true` | Enable the offline rule interpreter |
| `CACHE_SIZE` | `512` | Identity-cache entries (`0` disables) |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry lifetime |
| `NUMERIC_TOLERANCE` | `0.01` | Comparison tolerance (kWh / BDT) |
| `BATTERY_CYCLING_EPSILON` | `0` | Optional tie-break against needless cycling |
| `REQUEST_TIMEOUT_SECONDS` | `25` | Hard per-request budget (below the 30 s judge timeout) |
| `DISABLE_DOTENV` | *(unset)* | Set to `1` to ignore `.env` entirely (used by the test suite to stay hermetic) |

---

## 8. Local validation and tests

### Full test suite

```bash
python -m pytest
```

**74 tests, all passing** in ~2.5 s, fully offline.

| File | Covers |
|---|---|
| `tests/test_public_cases.py` | Optimality vs all 10 public references, replay cleanliness, exact energy balance, no simultaneous charge/discharge, heuristic validity, rule-interpreter accuracy, full pipeline with and without a model |
| `tests/test_guardrails.py` | Fenced/prose/bare JSON, brace matching inside strings, hour sorting and dedup, factor and reserve clamping, unknown directive rejection, `applies` semantics, index coverage, policy assertions |
| `tests/test_solver.py` | Energy balance exactness, neutrality, price arbitrage, grid-cap pre-charging, solar reduction, no-charge windows, reserve enforcement, infeasible-scenario degradation, scipy-unavailable fallback, tamper detection by the verifier |
| `tests/test_api.py` | `/health`, full response schema, self-consistent totals, out-of-order hours, every status code, extra-field tolerance, secret-leak check, `.env` hermeticity |

The suite is **hermetic**: `DISABLE_DOTENV=1` is set automatically, so a local `.env`
containing a real API key can never turn the tests into billed, network-dependent provider calls.

### Public sample pack verification

```bash
python tools/verify_public_cases.py
```

Solves all 10 public cases using the published ground-truth directives and compares against the
organizer's reference cost:

```
case         entries    our cost    ref cost     delta       grid     peak  replay
----------------------------------------------------------------------------------
SAMPLE-01          2    38365.00    38365.00      0.00    2692.50   187.50  OK [ok]
SAMPLE-02          1    42885.00    42885.00      0.00    2915.00   180.00  OK [ok]
SAMPLE-03          1    35480.00    35480.00      0.00    2430.00   205.00  OK [ok]
SAMPLE-04          1    40495.00    40495.00      0.00    2645.00   225.00  OK [ok]
SAMPLE-05          1    33950.00    33950.00      0.00    2430.00   175.00  OK [ok]
SAMPLE-06          3    34090.00    34090.00      0.00    2395.00   175.00  OK [ok]
SAMPLE-07          2    38550.00    38550.00      0.00    2560.00   185.00  OK [ok]
SAMPLE-08          2    37665.00    37665.00      0.00    2490.00   210.00  OK [ok]
SAMPLE-09          2    34873.00    34873.00      0.00    2504.00   170.00  OK [ok]
SAMPLE-10          3    41620.00    41620.00      0.00    2715.00   190.00  OK [ok]
----------------------------------------------------------------------------------
failures: 0/10
```

Every public reference schedule is reproduced at **exactly** the reference cost, which independently
confirms both the formulation and the directive semantics. Note that our hourly action sequences
differ from the references in several cases while reaching identical cost — alternate optima, which
the judge explicitly accepts.

### Inspecting the case pack

```bash
python tools/inspect_cases.py        # directive patterns, battery shapes, tariff structure
python tools/make_sample.py 9 out.json   # dump one request body for manual curl testing
```

### Live LLM smoke test

```bash
python tools/check_llm.py            # all 10 cases
python tools/check_llm.py 3          # a single case
```

Calls the configured provider once per case, runs the guardrails, and diffs the parsed directives
against the published ground truth. Requires `LLM_API_KEY`.

### End-to-end check against a running service

```bash
# with the service running on port 8000
python tools/e2e_check.py
python tools/e2e_check.py http://127.0.0.1:8100
```

Exercises the complete judged path over real HTTP — Pydantic → LLM → guardrails → optimizer →
replay → JSON — for all 10 public cases, and reports a latency breakdown. Exits non-zero on any
failure, so it doubles as a pre-submission gate.

### 50-case deployment suite

```bash
python tools/suite_50.py --url https://your-service.example.com
python tools/suite_50.py --category TIME        # one group
python tools/suite_50.py --only SR-PCT-OF,P-RES-A
python tools/suite_50.py --salt=cold1           # defeat the cache, measure true latency
python tools/suite_50.py --dump tools/suite_50.json
```

A broad regression suite for a **deployed** instance. Each case is scored on five independent
dimensions:

1. HTTP status matches the contract (`200` / `400` / `422`)
2. response schema is complete and well-formed
3. `directive_interpretation` matches the case's ground truth
4. the returned `hourly_plan` **replays clean against the ground-truth directives** — the judge's
   actual method, so a correct parse with an unapplied directive still fails
5. reported cost is not worse than the optimum computed locally for the same scenario and ground
   truth, i.e. `min(1, optimal/team)` stays at `1.0`

Reported totals are also recomputed from `hourly_plan` to catch any arithmetic drift.

| Category | Cases | Focus |
|---|---|---|
| `SOLAR` | 4 | percentage-of vs percentage-reduction, fractional wording, "drop to X%" |
| `CHARGE` | 2 | explicit prohibition, indirect "charger offline" |
| `DISCHARGE` | 2 | explicit prohibition, passive-voice availability |
| `RESERVE` | 2 | absolute kWh, percentage of capacity |
| `GRID` | 2 | plain cap, tight equipment cap requiring near-max discharge |
| `DISTRACTOR` | 5 | admin/menu/booking notes, including one that says "hours" |
| `TIME` | 7 | single hour, to-midnight, from-midnight, noon, 6-hour window, pre-dawn, late evening |
| `MULTI` | 10 | 2–3 directives, distractor mixing, duplicate-type merging (factors multiply, reserve max, cap min) |
| `PARAPHRASE` | 6 | the same directive written two ways each, for robustness |
| `STRESS` | 6 | zero solar, surplus/curtailment, flat tariff, extreme peak, tiny battery, zero discharge rate |
| `API` | 4 | `/health`, malformed JSON → 400, too many notes → 400, semantic invalid → 422 |

#### Results against the deployed Render instance

```
target: https://bup-hackathon-00ea.onrender.com
  SOLAR 4/4 · CHARGE 2/2 · DISCHARGE 2/2 · RESERVE 2/2 · GRID 2/2 · DISTRACTOR 5/5
  TIME 7/7 · MULTI 10/10 · PARAPHRASE 6/6 · STRESS 6/6 · API 4/4
  total: 50/50 passed
  latency (cold cache): median 837ms · p95 1187ms · max 1345ms
```

* Every `cost` matched the locally computed optimum to the cent, so the optimization quality ratio
  is `1.0` across the board.
* No interpretation entry carried the `Rule-based fallback` marker, confirming the **live model**
  handled all 46 note-interpretation cases with no silent degradation to the offline interpreter.
* Replaying the same cases a second time measured ~79 ms, which is the response cache working —
  hence `--salt`, which perturbs `scenario_id` so latency measurements are not cache hits.

### Verified against the live provider

Measured with `deepseek-chat` (`tools/e2e_check.py`, all 10 public cases, cache cold):

| Check | Result |
|---|---|
| Directive interpretation vs published ground truth | **10/10 exact matches**, 0/10 mismatches |
| Response cost vs organizer reference | **0.00 delta on all 10** |
| Pipeline source | `source=llm`, `warnings=0` on all 10 — no fallbacks triggered |
| Latency | min 791 ms · median 1019 ms · **p95 1489 ms** · max 1489 ms |

The p95 target for full latency credit is ≤ 5 s, so this sits comfortably inside the top band. Both
percentage traps were resolved correctly by the live model — `"25% of the forecast"` → `0.25` and
`"an 80% reduction"` → `0.2` — as was the relative reserve `"50% of battery capacity"` → `100 kWh`
on a 200 kWh battery.

---

## 9. Project layout

```
Bup_soln/
├── app/
│   ├── main.py                    FastAPI app, routes, exception handlers
│   ├── config.py                  env-driven settings
│   ├── schemas.py                 exact request/response contract
│   ├── errors.py                  HTTP-mapped error types
│   ├── cache.py                   TTL + LRU identity cache
│   ├── pipeline.py                orchestration and recovery ladder
│   ├── llm/
│   │   ├── base.py                LLMClient protocol
│   │   ├── openai_compat.py       provider client (DeepSeek default)
│   │   └── prompts.py             system prompt + few-shot examples
│   ├── guardrails/
│   │   ├── normalize.py           L1-L4 parse, repair, reject
│   │   └── validate.py            L4 policy assertions
│   ├── heuristics/
│   │   └── rule_interpreter.py    offline fallback interpreter
│   └── optimization/
│       ├── constraints.py         directives -> per-hour parameters
│       ├── solver.py              LP builder, solver tiers, post-processing
│       └── replay.py              independent hour-by-hour verifier
├── tests/                         74 tests, hermetic and offline
├── docs/
│   ├── ARCHITECTURE.md            6 Mermaid diagrams + 3-minute narration script
│   └── architecture.html          self-contained diagram for screen recording
├── tools/                         developer utilities (not in the judged path)
│   ├── inspect_cases.py           summarize the public case pack
│   ├── make_sample.py             dump one request body for manual curl
│   ├── verify_public_cases.py     optimum check vs the 10 public references
│   ├── check_llm.py               live interpretation check vs ground truth
│   ├── e2e_check.py               full-path HTTP check against a running service
│   ├── suite_50.py                50-case deployment regression suite
│   └── suite_50.json              the generated case pack
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## 10. Design decisions

**Constraint modification, not objective modification.** Every directive changes bounds, rates, or
RHS values; the objective is always `Σ tariff × grid`. That keeps the optimization model family
constant and makes a single LP builder correct for every scenario.

**LP over MILP.** The netting lemma (section 5) proves binaries are unnecessary, so we avoid MILP
runtime entirely. On a 120-variable model this is a sub-10 ms solve, leaving the whole latency budget
to the model call.

**Two independent code paths.** `constraints.py` builds the model; `replay.py` re-derives every rule
from scratch and checks the finished plan. They intentionally share no code, so a bug in one is
caught by the other.

**No English in the optimizer.** Directives are compiled to numbers before the solver runs, which
means the solver cannot be prompt-injected and cannot "invent" a rule.

**Deterministic `plan_summary`.** Generating the summary from the solved plan costs nothing, is
always consistent with the numbers, and cannot fail at the last moment. The model's job is
interpretation, which is what the challenge actually requires.

**Identity caching.** Repeated identical scenarios (a common harness pattern) answer in ~7 ms
instead of paying another round-trip. The key is a SHA-256 of the canonical request, so a changed
note, tariff, or battery state can never return a stale plan.

**Fail soft, never 5xx on a valid request.** Provider outage, malformed model output, infeasible
model, and replay mismatch each have a defined recovery stage. The service always tries to return a
plan rather than an error page.

**Reject only what is provably infeasible.** Semantic rejections (422) are limited to battery states
where *no* valid plan can exist, each justified by a constraint contradiction rather than a guess —
for example `initial < minimum` is impossible because neutrality forces `soc[23] == initial` while
the SOC bound forces `soc[23] >= minimum`. Conversely, inputs that are merely unusual but legal are
accepted and normalised, because over-strict validation silently fails valid harness cases.

---

## 11. Known limitations

* **The offline rule interpreter is a fallback, not a substitute.** It handles the writing patterns
  in the public pack (verified exactly), but hidden paraphrases are the model's job. It is
  intentionally not the primary interpreter, since the challenge requires a language model in that
  path.
* **Only OpenAI-compatible APIs are wired up natively.** Gemini and Anthropic would need a small
  adapter implementing `LLMClient` (`app/llm/base.py`); the pipeline is provider-agnostic.
* **The elastic tier knowingly relaxes base rules.** It exists to avoid a crash on a scenario the
  organizers promised would be feasible. If it fires, `plan_summary` says so and the plan may not
  satisfy every base constraint.
* **Rounding to 6 decimals** can leave sub-micro-kWh residuals in the balance equation. This is far
  inside the official 0.01 tolerance.
* **No intra-request parallel model calls.** Notes are interpreted in a single call, which is
  cheaper and keeps `note_index` ordering trivial.
* **Latency depends on the provider.** p95 is dominated by the model round-trip; a slow or
  rate-limited provider will cost Performance & Reliability points even though the solver takes
  milliseconds.
* **`plan_summary` wording is templated**, so it is informative but not conversational. It is not
  scored byte-for-byte.

---

## 12. Security and secret handling

* `LLM_API_KEY` is read from the environment only. It is never logged, never echoed in a response,
  and never included in an error message.
* `.env` is git-ignored; only `.env.example` (names, no values) is committed. Verified with
  `git check-ignore -v .env` → `.gitignore:2:.env`, and `git add -A --dry-run` stages only
  `.env.example`.
* `.env` loading can be disabled entirely with `DISABLE_DOTENV=1`, which the test suite sets so a
  developer's real credentials can never reach a test run.
* Error responses are always generic. Internal detail, stack traces, and provider payloads stay in
  server logs — verified by `tests/test_api.py::test_error_bodies_leak_nothing_sensitive`.
* Model output is treated as untrusted input and cannot introduce a directive type that is not in
  the published enum.
* Only synthetic challenge data is used; no live campus, utility, billing, or personal data.

---

## 13. Deployment status

Per instruction, **deployment artifacts are deferred**: no Dockerfile, no registry image, and no
hosted endpoint are included in this pass. Everything needed to run the service locally is present
and verified.

When deployment is picked up, the remaining work is:

1. a `Dockerfile` that binds `0.0.0.0`, exposes the documented port, and bakes in **no** secrets;
2. a pullable registry reference with an exact tag or digest;
3. a hosted public base URL reachable for `GET /health` and `POST /optimize-energy`;
4. the 3-minute architecture video (tie-break only, no base points).
