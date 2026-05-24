# Ground-Truth Benchmark Set — Design

**Date:** 2026-05-24
**Status:** Approved
**Author:** Shivnarain (with Claude)
**Sub-project:** SP2 of the Eval Harness (SP1 = reproduction-rate tracking, shipped; SP3 = LLM-as-judge for analysis quality; SP4 = score calibration — each its own spec→plan→build cycle)

---

## 1. Context

SP1 turned the per-experiment baseline-comparison verdict into a persisted, trended
`reproduction_rate` metric on the generic `EvalMetric` store. But that metric is only as
trustworthy as `baseline_comparator` itself: a verdict is whatever the pipeline *computed*, and it
silently changes if an experiment is re-run. There is no human-confirmed ground truth and no way to
know whether the system's measurements are actually *correct* — only how many it labeled as
reproduced.

SP2 adds a **curated, fixed golden set**: a deliberately-chosen, stable collection of paper claims
whose true numeric outcomes a human has recorded (expected value + tolerance). Scoring the pipeline's
latest measurements against this set yields a `benchmark_accuracy` signal that detects pipeline
regressions over time and establishes the ground truth SP4 will calibrate against.

This is distinct from SP1: SP1 measures the system's **research yield** ("of what we ran, how much
reproduced"); SP2 measures the system's **measurement accuracy** ("on claims whose answer we already
know, how often does the pipeline land in range"). SP2 validates that SP1's numbers mean something.

## 2. Goals

- Let a human curate a stable golden set of paper-claim ground-truth labels (metric + expected value + tolerance) via a UI.
- Score the pipeline's latest measurements against the golden set, producing a `benchmark_accuracy` metric.
- Persist a full audit trail: every scoring run and every per-item outcome (measured vs expected, pass/fail).
- Trend `benchmark_accuracy` on the existing `EvalMetric` substrate (one query path, dashboard reuse).
- Auto-score at the end of every cycle (cheap, no compute) and on demand from the UI.

## 3. Non-goals (deferred)

- Re-executing golden experiments as part of a benchmark run (scoring reads the latest stored
  `ExperimentResult`; re-running stays the existing manual `experiment run` step).
- LLM-as-judge scoring of analysis quality (SP3).
- Calibrating novelty/relevance/verdict scores against the golden set (SP4 — SP2 only *collects* ground truth).
- A CLI for the benchmark (curation, running, and viewing are all in Streamlit per the brainstorm).
- Per-metric *direction* semantics ("higher is better") — a symmetric tolerance band is sufficient.
- Multi-labeler agreement / provenance beyond a free-text `note` (single-user system).

## 4. Approaches considered

**A — One table + reuse SP1.** A single `BenchmarkItem` table; per-item outcomes computed on demand;
only aggregate `benchmark_accuracy` persisted (in `EvalMetric`). Minimal schema. Rejected: no
per-item history ("which item regressed when"), which is the point of a regression benchmark.

**B — Full audit trail (selected).** Three tables: `BenchmarkItem` (golden labels), `BenchmarkRun`
(one scoring pass), `BenchmarkItemResult` (per-item outcome within a run). Aggregate
`benchmark_accuracy` still written to `EvalMetric` for the trend/dashboard; the run + item-result
tables hold the forensic history. More schema/CRUD, but gives full regression forensics and a stable
immutable record of each run. **Selected.**

**C — No new tables.** Expected-value columns bolted onto `Experiment`/`ExperimentResult`, accuracy
computed on read. Smallest schema change, but conflates golden labels with run data, makes "the active
golden set" a fuzzy flag on operational rows, and has no run history. Wrong shape for a benchmark.

**Selected: B.** EvalMetric remains the harness-wide trend substrate (SP1's stated design intent);
the two new run tables are the SP2-specific audit trail.

## 5. Data model (locked)

All three tables are created by `init_db()` via `SQLModel.metadata.create_all` — no ALTER migration
(mirrors SP1's `EvalMetric`). Appended to `core/models.py` after `EvalMetric`.

### 5.1 `BenchmarkItem` — the golden set

```python
class BenchmarkItem(SQLModel, table=True):
    __tablename__ = "benchmark_item"
    id: str = Field(primary_key=True)                 # uuid4
    experiment_id: str = Field(index=True)            # experiment this claim is scored against
    metric_name: str                                  # key to read from ExperimentResult.metrics JSON
    expected_value: float                             # the paper's known/true value
    tolerance: float                                  # band half-width
    tolerance_type: str = "relative"                  # "relative" (% of expected) | "absolute"
    unit: Optional[str] = None                        # display only
    note: str = ""                                    # provenance / why this is ground truth
    active: bool = True                               # deactivate instead of hard-delete
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
```

### 5.2 `BenchmarkRun` — one scoring pass

```python
class BenchmarkRun(SQLModel, table=True):
    __tablename__ = "benchmark_run"
    id: str = Field(primary_key=True)                 # uuid4
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    cycle_id: Optional[str] = Field(default=None, index=True)  # pipeline cycle, or None for manual
    trigger: str = "manual"                            # "cycle" | "manual"
    n_items: int = 0
    n_pass: int = 0
    n_fail: int = 0
    n_unscorable: int = 0
    accuracy: Optional[float] = None                   # n_pass / (n_pass + n_fail); None if 0 scorable
```

### 5.3 `BenchmarkItemResult` — per-item outcome within a run

```python
class BenchmarkItemResult(SQLModel, table=True):
    __tablename__ = "benchmark_item_result"
    id: str = Field(primary_key=True)                 # uuid4
    run_id: str = Field(index=True)
    item_id: str = Field(index=True)
    experiment_id: str
    metric_name: str
    expected_value: float
    tolerance: float
    tolerance_type: str
    measured_value: Optional[float] = None            # None when unscorable
    passed: Optional[bool] = None                     # None when unscorable
    status: str                                        # pass | fail | no_result | missing_metric | non_numeric
```

## 6. Components & boundaries

### 6.1 `knowledge/benchmark_store.py` — CRUD (sole DB interface)

Mirrors existing `*_store.py` patterns; all write sessions use `expire_on_commit=False`.

| Function | Purpose |
|---|---|
| `save_item(item) -> None` | Insert/update a `BenchmarkItem`. |
| `get_items(active_only: bool = True) -> list[BenchmarkItem]` | The golden set. |
| `get_item(item_id) -> BenchmarkItem \| None` | Single item. |
| `deactivate_item(item_id) -> None` | Soft-delete (sets `active=False`). |
| `save_run(run, item_results) -> None` | Persist a `BenchmarkRun` + its `BenchmarkItemResult`s in one session. |
| `get_runs(limit=30) -> list[BenchmarkRun]` | Run history (most recent first). |
| `get_latest_run() -> BenchmarkRun \| None` | For the UI's current-state table. |
| `get_item_results(run_id) -> list[BenchmarkItemResult]` | Per-item outcomes for a run. |

### 6.2 `analysis/benchmark_scorer.py` — compute + orchestration

Pure-core / thin-orchestrator split, like SP1's `reproduction_metrics.py`.

| Function | Purity | Purpose |
|---|---|---|
| `Measurement` dataclass | — | `(item, measured: Optional[float], status: str, difficulty: str, source: str)`. |
| `ItemOutcome` dataclass | — | scored result: item id/experiment/metric/expected/tolerance + measured/passed/status. |
| `gather_measurements(items) -> list[Measurement]` | DB read | For each item, load the experiment's latest `ExperimentResult`, parse metrics JSON, extract `metric_name` (numeric list → mean, per `statistical_analyzer` convention). Joins `Experiment`/`Paper`/`PaperAnalysis` for `difficulty`/`source` dimensions, selecting only the columns needed (avoids the `paper_analysis` corruption landmine — see SP1 hardening). Missing result/key/non-numeric → status set, `measured=None`. |
| `score(measurements) -> list[ItemOutcome]` | **Pure** | Per item: scorable → relative `|m−e| ≤ tol·|e|` (or absolute `|m−e| ≤ tol`; `expected==0` + relative falls back to absolute); unscorable → carry the status through, `passed=None`. |
| `record_benchmark_run(cycle_id=None, trigger="manual") -> BenchmarkRun` | side effects | gather → score → build `BenchmarkRun` + `BenchmarkItemResult`s → `save_run` → write aggregate `benchmark_accuracy` to the SP1 `EvalMetric` store (`overall` + `difficulty:*` + `source:*`, `numerator=n_pass`, `denominator=n_scorable`). Returns the run. |

### 6.3 `ui/views/benchmark.py` — Streamlit page

Registered in `ui/app.py` nav after Review Queue. Three zones: **Curate** (browse completed
experiments + paper + measured metrics + `baseline_comparison`; add/edit/deactivate items via a form
whose metric dropdown is populated from the chosen result's metrics JSON), **Run** (button →
`record_benchmark_run(trigger="manual")`, shows the run summary), **View** (`benchmark_accuracy` trend
from `EvalMetric` via the dashboard's `line_chart` pattern + latest run's per-item table with
fail/unscorable rows highlighted). Reads cached on the existing fragment cadence; only the curate
forms and Run button write.

## 7. Data flow

### 7.1 Manual run (UI)
Run button → `record_benchmark_run(trigger="manual")` → gather latest results → score → persist run +
item-results → write `benchmark_accuracy` to `EvalMetric` (`cycle_id` = `benchmark-<YYYYMMDD-HHMMSS>`).

### 7.2 Auto run (per cycle)
At the end of `analysis_pipeline.run`, **after** SP1's `record_cycle_snapshot`, in its own try/except:
if `get_items(active_only=True)` is non-empty, call `record_benchmark_run(state.cycle_id,
trigger="cycle")`. A failure is logged, never aborts the cycle (same graceful-degradation pattern as
SP1 and `contradiction_detector`). The accuracy trend self-populates per cycle alongside
`reproduction_rate`.

## 8. Scoring semantics (locked)

- **Scorable** = the bound experiment has a latest `ExperimentResult`, the metrics JSON contains
  `metric_name`, and the value is numeric (or a numeric list → mean). Otherwise **unscorable** with an
  explicit status (`no_result` | `missing_metric` | `non_numeric`).
- **pass** (scorable only): relative → `|measured − expected| ≤ tolerance · |expected|`; absolute →
  `|measured − expected| ≤ tolerance`. `expected == 0` with relative type → absolute comparison
  (avoids a zero-width band).
- **accuracy** = `n_pass / (n_pass + n_fail)`; unscorable items are **excluded from the denominator**
  (never counted as fail — the same honesty rule as SP1's "comparable"). Zero scorable ⇒ `accuracy =
  None`, rendered as `—`.
- **Unit of measurement** is one `BenchmarkItem` (one claim), aggregated per run.

## 9. Error handling & edge cases

- Malformed `ExperimentResult.metrics` JSON → that item is `non_numeric`/`missing_metric`, logged at
  WARN, never aborts the run.
- `record_benchmark_run` wrapped in try/except inside `analysis_pipeline.run`; failure logged, cycle
  continues.
- Empty golden set → auto-run is skipped entirely; a manual run produces a `BenchmarkRun` with
  `n_items=0, accuracy=None`.
- Experiment referenced by an item is deleted → item still loads; scoring yields `no_result` (no hard
  FK cascade, consistent with the codebase).
- Re-running an experiment changes its latest result → reflected in the *next* benchmark run; each
  historical `BenchmarkRun` is immutable.
- `paper_analysis` corruption (SP1 finding) → `gather_measurements` selects only needed columns and
  normalizes unknown difficulty to `unknown`, never loading the corrupt DATETIME.

## 10. Surfaces summary

- **Streamlit `Benchmark` page** — curate, run, view (sole human surface).
- **`EvalMetric` store** — `benchmark_accuracy` trend (dashboard/report reuse for free).
- **No CLI** in SP2.

## 11. Testing

`tests/test_benchmark_store.py` — CRUD across the three tables: save/get-active/deactivate items;
`save_run` persists run + item-results; `get_runs`/`get_latest_run`/`get_item_results`.

`tests/test_benchmark_scorer.py` — pure `score()`: relative pass/fail at band edges, absolute mode,
`expected==0` fallback, each unscorable status (`no_result`/`missing_metric`/`non_numeric`), numeric-list
mean handling, `accuracy = n_pass / n_scorable`, zero-scorable → `None`; integration
`record_benchmark_run` on a seeded in-memory DB asserts `BenchmarkRun` + `BenchmarkItemResult` rows
**and** the `benchmark_accuracy` `EvalMetric` row (overall + dimensions) appear; auto-run hook in
`analysis_pipeline` records a run when the golden set is non-empty and is skipped/charmless when empty;
a snapshot failure does not abort the pipeline.

All tests use real in-memory SQLite via the existing `conftest.py` fixture. SP2 is LLM-free — no Claude
mocking required.

## 12. Migrations & compatibility

- Three tables created via `create_all` on `init_db()`; no data migration, no ALTER.
- Reuses SP1's `eval_metric_store` unchanged (new `metric="benchmark_accuracy"` key).
- `analysis_pipeline.run` gains one more try/except-wrapped end-of-stage call; no public-API breakage.
- No new environment variables, no new external services, no new dependencies.

## 13. Risks

- **Curation effort:** the golden set is only as good as the labels a human records. Mitigated by the
  point-and-click UI surfacing the paper's measured metrics next to the entry form. Out of scope to
  enforce label quality.
- **Stale measurements:** scoring reads the latest stored result, which may predate recent code
  changes. This is intentional (cheap, deterministic); active regression detection via re-execution is
  a deferred non-goal. Documented in the scorer docstring.
- **Dimension explosion:** `benchmark_accuracy` reuses SP1's dimension keys; the dashboard tile already
  guards against unknown dimensions by filtering to known prefixes.

## 14. Open questions

None at the time of writing — all clarifications resolved in brainstorming
(golden-set nature, ground-truth label = expected metric value + tolerance, score-latest semantics,
Streamlit curation surface, full-audit-trail data model).
