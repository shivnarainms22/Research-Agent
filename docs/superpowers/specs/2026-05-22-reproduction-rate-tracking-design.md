# Reproduction-Rate Tracking — Design

**Date:** 2026-05-22
**Status:** Approved
**Author:** Shivnarain (with Claude)
**Sub-project:** SP1 of the Eval Harness (SP2 = ground-truth benchmark set; SP3 = LLM-as-judge for analysis quality; SP4 = score calibration — each its own spec→plan→build cycle)

---

## 1. Context

The system already computes a per-experiment baseline-comparison verdict
(`fully_reproduced` | `partially_reproduced` | `not_reproduced` | `no_baselines` | …) inside
`analysis/baseline_comparator.py`, and `reporting/report_generator.py` reduces these to a single
`reproduction_rate` percentage *inside one report only*. That number is recomputed each cycle and
never persisted. There is no time series, no delta-vs-last-cycle, no dimensional slicing, no
historical trend. The autonomous research loop's north-star metric is therefore invisible the
moment a report is closed.

The goal of this sub-project is to turn the ephemeral per-report number into a rigorous, persisted,
trended, dimensionally-sliced metric — and to do so on a substrate that the rest of the Eval
Harness (SP2–SP4) will write into without redesign.

## 2. Goals

- Define reproduction rate rigorously and unambiguously.
- Persist a time series of the metric, sliced by relevant dimensions.
- Compute and write a snapshot automatically at the end of every pipeline cycle.
- One-time backfill seeds the trend with real historical data.
- Surface the metric in the weekly report (with delta vs prior cycle) and the Streamlit dashboard.
- Provide a substrate (the generic `EvalMetric` store) that SP2–SP4 will reuse for future eval metrics.

## 3. Non-goals (deferred)

- LLM-as-judge scoring of analysis quality (SP3).
- A curated ground-truth benchmark set / labeling workflow (SP2).
- Calibration of novelty/relevance scores against human labels (SP4).
- Replacing Streamlit / building a FastAPI service / killing the CLI (separate platformization brainstorm).
- A dedicated `eval` CLI command (deferred until the frontend-architecture decision).

## 4. Approaches considered

**A — Generic eval-metric time series (`EvalMetric` table).** One table keyed by `(metric,
dimension, cycle_id)` storing numerator/denominator/value/context. Reproduction rate is
`metric="reproduction_rate"`; future SP3/SP4 metrics use the same table with different `metric`
keys. **Recommended.** Substrate for the whole harness; one query path; SP2–4 reuse it untouched.
Cost: `dimension` is a stringly-typed key (e.g. `"difficulty:easy"`).

**B — Reproduction-specific table (`ReproductionSnapshot`).** Explicit columns for each count and
rate. Ergonomic for this metric, but forces two more bespoke tables when SP3 and SP4 land. Wrong
shape for the foundation.

**C — Compute-on-read (no new table).** Derive the metric from `ExperimentResult` on each query.
No persistence — but verdicts can change if experiments are re-run (historical points become
unstable), the report's delta calculation gets fragile, and SP3/SP4 fundamentally need
persistence (LLM-judge scores are not derivable). Disqualifying.

**Selected: A.**

## 5. Metric definition (locked)

Per `(metric, dimension)` snapshot:

- **comparable** = an `ExperimentResult` whose `baseline_comparison.overall` ∈
  `{fully_reproduced, partially_reproduced, not_reproduced}`. The `no_*` early-exit statuses
  (`no_analysis`, `no_baselines`, `no_metrics`, `no_experiments`) are **not** comparable and are
  excluded from the denominator.
- **`reproduction_rate`** = `count(overall == fully_reproduced) / count(comparable)`
- **`partial_rate`** = `count(overall == partially_reproduced) / count(comparable)`
- **Denominator 0 ⇒ `value = None`** (undefined). Surfaces render as `"—"` / "no comparable
  experiments". Never as 0% — that would lie.
- **Ablations excluded** (`Experiment.parent_experiment_id IS NOT NULL`). They vary a working
  setup; they do not reproduce a paper's claim.
- **Unit of measurement is one `ExperimentResult` row** (the overall verdict), not one
  per-metric comparison. This is what users think in and what existing surfaces display.

Why store **numerator + denominator**, not just `value`: cumulative rate must be computed as
`Σ numerator / Σ denominator`. Averaging percentages is mathematically wrong and gives
silently misleading trends. This is the single most important data-model decision in the spec.

## 6. Dimensions

Every snapshot writes the `overall` dimension. The following per-bucket dimensions are also
written when at least one comparable experiment falls into them:

| Dimension key family | Values | Source field |
|---|---|---|
| `overall` | (sole value) | — |
| `difficulty:{easy\|medium\|hard}` | three | `PaperAnalysis.reproducibility_difficulty` |
| `target:{local\|cloud_modal}` | two | `Experiment.execution_target` |
| `source:{arxiv\|semantic_scholar\|substack}` | three | `Paper.source` |

A typical cycle writes ~9–14 rows (2 metrics × `overall` + ~3–6 active dimension buckets).
Empty buckets are silently skipped.

## 7. Components & boundaries

Three small, isolated units. Each can be understood and tested independently.

### 7.1 `core/models.py` — `EvalMetric` schema

```python
class EvalMetric(SQLModel, table=True):
    __tablename__ = "eval_metric"
    id: str = Field(primary_key=True)               # uuid4
    metric: str = Field(index=True)                 # "reproduction_rate" | "partial_rate" | (future)
    dimension: str = Field(index=True, default="overall")
    value: Optional[float] = None                   # fraction 0–1; None if denominator==0
    numerator: int = 0
    denominator: int = 0
    cycle_id: str = Field(index=True)               # real cycle id, or "backfill-<YYYY-Www>"
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    context: str = "{}"                             # JSON: {"fully":3,"partial":1,"not":2}
```

Created automatically by `init_db()` via `SQLModel.metadata.create_all`. No ALTER migration needed.

### 7.2 `knowledge/eval_metric_store.py` — CRUD

Public functions (sole interface; mirrors existing `*_store.py` patterns and uses
`expire_on_commit=False`):

| Function | Purpose |
|---|---|
| `save_metrics(points: list[MetricPoint], cycle_id: str) -> None` | Persist a snapshot's rows in one session. |
| `get_latest(metric: str, dimension: str = "overall") -> EvalMetric \| None` | Most recent row for that (metric, dimension). |
| `get_previous(metric: str, dimension: str, before_cycle_id: str) -> EvalMetric \| None` | For delta-vs-prior-cycle in the report. |
| `get_trend(metric: str, dimension: str = "overall", limit: int = 30) -> list[EvalMetric]` | Ordered most-recent-last for sparklines. |
| `count_rows() -> int` | Used to gate the lazy backfill. |

`MetricPoint` is a lightweight dataclass `(metric, dimension, value, numerator, denominator, context)` — the in-memory shape `tally()` returns and `save_metrics` consumes.

### 7.3 `analysis/reproduction_metrics.py` — compute + orchestration

| Function | Purity | Purpose |
|---|---|---|
| `gather_verdicts(experiment_ids: Iterable[str] \| None = None) -> list[VerdictRow]` | DB read | Join `ExperimentResult` × `Experiment` × `PaperAnalysis` × `Paper`. Filter to completed, non-ablation, baseline-comparison-present. `None` = all history. |
| `tally(rows: list[VerdictRow]) -> list[MetricPoint]` | **Pure** | Bucket by every dimension; compute numerator/denominator/value per bucket. The core unit; the heart of the tests. |
| `record_cycle_snapshot(state: RunState) -> None` | side effects | End-of-stage hook: gather → lazy-backfill-if-empty → tally → save. |
| `backfill_from_history() -> int` | side effects | Bucket all historical comparable results by ISO week of `recorded_at`; write one snapshot per non-empty week with `cycle_id="backfill-<YYYY-Www>"`. Idempotent. Returns row count written. |

`VerdictRow` dataclass: `(overall: str, difficulty: str, target: str, source: str, recorded_at: datetime)`.

## 8. Data flow

### 8.1 Per-cycle snapshot (forward)

1. `analysis_pipeline.run(state)` finishes its existing work (analyzing experiments, generating ablations, etc.).
2. New final step (wrapped in try/except): `reproduction_metrics.record_cycle_snapshot(state)`.
3. Inside:
   - If `eval_metric_store.count_rows() == 0` → run `backfill_from_history()` first.
   - `gather_verdicts(state.experiment_ids_this_cycle)` → list of `VerdictRow`.
   - `tally(verdicts)` → list of `MetricPoint`.
   - `save_metrics(points, cycle_id=state.cycle_id)`.

If `experiment_ids_this_cycle` is empty (the documented "no new experiments this cycle" path),
`gather_verdicts` returns `[]`, `tally` returns rows with `denominator=0, value=None` for the
`overall` dimension, and the absence is recorded honestly — the trend will show a gap rather
than silently skipping.

### 8.2 Backfill (one-time, lazy)

- Triggered automatically on the first `record_cycle_snapshot` call when the table is empty.
- Pulls **all** completed `ExperimentResult` rows with non-null `baseline_comparison` and a
  non-ablation parent experiment.
- Buckets by ISO week (`recorded_at.isocalendar()` → `f"backfill-{year}-W{week:02d}"`).
- Per week: `tally(rows_in_week)` → `save_metrics(points, cycle_id=<weekly tag>)`.
- Idempotency: before writing a weekly bucket, check whether any row already exists with that
  `cycle_id`; if so, skip the bucket. (So re-invoking after a partial failure is safe.)
- Weeks with `denominator == 0` across all dimensions are not written (no signal to record).

## 9. Surfaces

### 9.1 Weekly report (`reporting/report_generator.py`)

- Remove the in-place `reproduction_rate = round(reproduced / max(len(exp_sections), 1) * 100, 1)`
  calculation.
- Replace with: `latest = get_latest("reproduction_rate")`, `previous =
  get_previous("reproduction_rate", "overall", state.cycle_id)`. Compute delta in
  percentage-points.
- Template change: render `"{rate}% (n={denominator}), Δ vs last cycle: {±x.x} pp"`, with `"—"`
  if `value is None`. The existing `weekly_report.md.j2` "System Stats" section is the home.

### 9.2 Streamlit dashboard (`ui/views/dashboard.py`)

A new fragment-cached tile **"Reproduction Rate"**:
- Big number: latest `overall` rate (or `—`).
- Subtitle: `n=<denominator>, partial: <x>%`.
- Sparkline / line chart: last 12 snapshots from `get_trend("reproduction_rate", "overall", 12)`.
- Optional `st.selectbox` to slice by dimension (overall / difficulty / target / source).
- All reads cached with the same fragment cadence as the existing dashboard tiles. No write paths
  in the UI.

### 9.3 Headless core is the API

No CLI command in SP1. The same functions are trivially exposable as a Typer subcommand or a
FastAPI route later — and *will* be, once the frontend-architecture decision lands. The point is
that SP1's logic doesn't depend on the surface choice.

## 10. Error handling & edge cases

- Malformed `baseline_comparison` JSON on a result → skip that row, log at WARN, continue. Do not
  fail the snapshot.
- `record_cycle_snapshot` is wrapped in try/except inside `analysis_pipeline.run` and a failure
  is logged but never aborts the cycle. (Mirrors the documented graceful-degradation pattern used
  by `contradiction_detector` and `token_log_store`.)
- `experiment_ids_this_cycle == []` is treated as a valid input — emits `value=None,
  denominator=0` rows so the time series honestly records the gap.
- Denominator-zero rows are written (per above) but rendered as `"—"` everywhere; arithmetic
  consumers (cumulative rates, deltas) must handle `value is None`.
- Backfill is idempotent: re-running after a crash or after upgrading is safe; weeks already
  represented are skipped.
- Cumulative-rate consumers must use `Σ numerator / Σ denominator`, never `mean(value)`.

## 11. Testing

New file `tests/test_reproduction_metrics.py` plus an extension to `tests/test_eval_metric_store.py`.

| Test | Asserts |
|---|---|
| `tally` happy path | Mix of full/partial/not + non-comparable → correct numerators, denominators, values across `overall` and each dimension. |
| `tally` all-non-comparable | Denominator 0 → `value is None`, not 0.0. |
| `tally` excludes ablations | Rows with `is_ablation=True` filtered out before bucketing. |
| `tally` dimensional bucketing | Multiple difficulties/targets/sources → correct per-bucket counts. |
| `gather_verdicts` integration | Seeded in-memory DB returns expected joined rows; non-completed/no-baseline rows excluded. |
| `record_cycle_snapshot` end-to-end | Seeded DB, fake `RunState` → expected rows in `eval_metric` afterward. |
| `record_cycle_snapshot` lazy backfill | Empty `eval_metric` table → first call triggers backfill, populates historical rows AND the cycle snapshot. |
| `backfill_from_history` bucketing | Results spanning two ISO weeks → exactly two weekly snapshots written. |
| `backfill_from_history` idempotency | Second call writes 0 rows. |
| `eval_metric_store` CRUD | `save_metrics`, `get_latest`, `get_previous`, `get_trend`, `count_rows` all behave as specified. |
| Report integration | `report_generator` uses store values and renders `"—"` when `value is None`. |

Tests follow existing patterns in `tests/conftest.py`: real in-memory SQLite via the existing
fixture, no Claude mocking required for SP1 (this whole sub-project is LLM-free).

## 12. Migrations & compatibility

- `EvalMetric` is created via `SQLModel.metadata.create_all` on `init_db()`. No data migration.
- `report_generator.py`'s existing reproduction-rate prose is *replaced*, not duplicated — the
  Jinja template's `{{ reproduction_rate }}` variable continues to exist, but is now sourced
  from the store and supplemented with `n`, `partial_rate`, and `delta_pp`.
- No public-API breakage for any other module.
- No new environment variables, no new external services.

## 13. Risks

- **Verdict mutability:** if an experiment is re-run, its `baseline_comparison` changes, which
  retroactively changes the count for the cycle it was attributed to. Acceptable for SP1; the
  snapshot at time T reflects what was true at time T's compute. Document this in the function
  docstring. SP2's ground-truth labels will make this irrelevant later.
- **Cycle attribution for backfill:** historical results were never tagged with a cycle, so
  backfill uses ISO-week buckets. The trend is "rate by week" until forward snapshots take over.
  Acceptable and explicit.
- **Dimension explosion:** the schema permits arbitrary `dimension` strings; future SP3/SP4 metrics
  could add new dimensions. This is a feature, not a risk — but the dashboard tile must guard
  against unknown dimensions gracefully (filter to known prefixes in the selector).

## 14. Open questions

None at the time of writing — all clarifications resolved in brainstorming.

---

## Appendix A — Reduced "comparable" decision table

| `baseline_comparison` JSON | overall | Comparable? | Contributes to repro num? | Contributes to partial num? |
|---|---|---|---|---|
| `{"overall": "fully_reproduced", …}` | fully_reproduced | ✓ | ✓ | — |
| `{"overall": "partially_reproduced", …}` | partially_reproduced | ✓ | — | ✓ |
| `{"overall": "not_reproduced", …}` | not_reproduced | ✓ | — | — |
| `{"overall": "no_experiments", …}` | no_experiments | ✗ | — | — |
| `{"status": "no_analysis"}` | (absent) | ✗ | — | — |
| `{"status": "no_baselines"}` | (absent) | ✗ | — | — |
| `{"status": "no_metrics"}` | (absent) | ✗ | — | — |
| `None` / unparseable | — | ✗ (logged WARN, skipped) | — | — |
