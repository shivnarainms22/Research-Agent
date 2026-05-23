# Reproduction-Rate Tracking Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the ephemeral per-report reproduction-rate number into a persisted, dimensionally-sliced, trended metric on a generic eval-metric store that SP2–SP4 of the Eval Harness will reuse.

**Architecture:** A new `EvalMetric` SQLModel table (substrate for all eval harness metrics). A pure compute layer (`tally()`) over `VerdictRow` rows joined from existing `ExperimentResult`/`Experiment`/`PaperAnalysis`/`Paper`. An orchestrator (`record_cycle_snapshot`) called at the end of `analysis_pipeline.run()` that also lazily backfills history bucketed by ISO week on first call. Surfaces: store-backed lookup in `report_generator.py` (replacing ad-hoc math) and a fragment-cached Streamlit dashboard tile.

**Tech Stack:** Python 3.11, SQLModel/SQLite (WAL), pytest, Streamlit, Jinja2. No new dependencies, no Claude calls in this sub-project.

**Spec:** `docs/superpowers/specs/2026-05-22-reproduction-rate-tracking-design.md`

---

## File Structure

**New files**
- `knowledge/eval_metric_store.py` — CRUD for `EvalMetric` rows. Sole interface to the table.
- `analysis/reproduction_metrics.py` — `VerdictRow`/`MetricPoint` dataclasses, pure `tally()`, DB-side `gather_verdicts()`, orchestrator `record_cycle_snapshot()`, one-shot `backfill_from_history()`.
- `tests/test_eval_metric_store.py` — store CRUD tests.
- `tests/test_reproduction_metrics.py` — compute + integration tests.

**Modified files**
- `core/models.py` — append `EvalMetric` SQLModel class (after `TokenUsageLog`).
- `core/database.py` — add `EvalMetric` to the `init_db` import line so `create_all` picks it up.
- `analysis/analysis_pipeline.py` — at end of `run()`, call `reproduction_metrics.record_cycle_snapshot(state)` inside try/except.
- `reporting/report_generator.py` — replace lines 309–313 (ad-hoc reproduction-rate math) with store-backed lookup; thread new template vars (`repro_n`, `partial_rate`, `delta_pp`).
- `reporting/templates/weekly_report.md.j2` — rewrite "System Stats" block (lines ~152–162) to render the rigorous metric with `n`, partial-rate, delta, and an honest "—" for `None`.
- `ui/views/dashboard.py` — add new fragment-cached `_reproduction_rate_tile()` render below the existing Experiment Status block.

**No changes to** `core/state.py`, `scheduler/pipeline_runner.py`, the experiment pipeline, the ingestion pipeline, or `main.py`.

---

## Pre-flight

- [ ] **Step 0a: Create feature branch** (current branch is `main`; never implement on `main`).

```bash
git checkout -b feat/eval-harness-sp1-reproduction-rate
```

- [ ] **Step 0b: Confirm baseline state — full test suite green before any changes.**

Run: `uv run pytest tests/ -v --tb=short`
Expected: 39 passed, exit 0.

If any test fails on `main`, stop and surface to the user — do not start implementation on a red baseline.

---

## Chunk 1: Data layer (model + store)

### Task 1: Add `EvalMetric` SQLModel

**Files:**
- Modify: `core/models.py` (append after `TokenUsageLog`, before the section divider on line ~143)
- Modify: `core/database.py:52` (import line inside `init_db`)
- Test: `tests/test_eval_metric_store.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_eval_metric_store.py`:

```python
"""Tests for knowledge/eval_metric_store.py and the EvalMetric model."""
from __future__ import annotations

from sqlmodel import select


def test_eval_metric_table_is_created(in_memory_engine):
    from sqlmodel import Session
    from core.models import EvalMetric

    # Table is created by the fixture's metadata.create_all. Inserting a row exercises the schema.
    row = EvalMetric(
        id="m1",
        metric="reproduction_rate",
        dimension="overall",
        value=0.75,
        numerator=3,
        denominator=4,
        cycle_id="cycle_x",
        context="{}",
    )
    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(row)
        session.commit()
    with Session(in_memory_engine) as session:
        got = list(session.exec(select(EvalMetric)).all())
    assert len(got) == 1
    assert got[0].metric == "reproduction_rate"
    assert got[0].value == 0.75
    assert got[0].numerator == 3
    assert got[0].denominator == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_metric_store.py::test_eval_metric_table_is_created -v`
Expected: FAIL with `ImportError: cannot import name 'EvalMetric' from 'core.models'`.

- [ ] **Step 3: Add the model**

Append to `core/models.py` immediately after the `TokenUsageLog` class (before the `# ---` separator on line ~143):

```python
class EvalMetric(SQLModel, table=True):
    __tablename__ = "eval_metric"

    id: str = Field(primary_key=True)            # uuid4
    metric: str = Field(index=True)              # "reproduction_rate" | "partial_rate" | (future)
    dimension: str = Field(index=True, default="overall")  # "overall" | "difficulty:easy" | "target:local" | ...
    value: Optional[float] = None                # fraction 0-1; None if denominator == 0
    numerator: int = 0
    denominator: int = 0
    cycle_id: str = Field(index=True)            # real cycle_id, or "backfill-<YYYY-Www>"
    recorded_at: datetime = Field(default_factory=datetime.utcnow)
    context: str = "{}"                          # JSON: {"fully":3,"partial":1,"not":2}
```

Also update `core/database.py:52` so `init_db()` registers the new model with `SQLModel.metadata.create_all`:

```python
from core.models import (
    Paper, PaperAnalysis, Experiment, ExperimentResult, ResearchReport,
    Contradiction, ResearchGap, ThemeCluster, TokenUsageLog, EvalMetric,
)  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_eval_metric_store.py::test_eval_metric_table_is_created -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/models.py core/database.py tests/test_eval_metric_store.py
git commit -m "feat(eval): add EvalMetric SQLModel for time-series eval metrics"
```

---

### Task 2: `MetricPoint` dataclass + `save_metrics` + `count_rows`

**Files:**
- Create: `knowledge/eval_metric_store.py`
- Test: `tests/test_eval_metric_store.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_metric_store.py`:

```python
def test_save_metrics_persists_rows(in_memory_engine):
    from knowledge.eval_metric_store import MetricPoint, save_metrics, count_rows

    assert count_rows() == 0

    points = [
        MetricPoint(metric="reproduction_rate", dimension="overall",
                    value=0.5, numerator=1, denominator=2, context='{"fully":1,"partial":0,"not":1}'),
        MetricPoint(metric="partial_rate", dimension="overall",
                    value=0.0, numerator=0, denominator=2, context='{"fully":1,"partial":0,"not":1}'),
    ]
    save_metrics(points, cycle_id="cycle_a")

    assert count_rows() == 2


def test_save_metrics_none_value_persists(in_memory_engine):
    from knowledge.eval_metric_store import MetricPoint, save_metrics, count_rows

    save_metrics(
        [MetricPoint(metric="reproduction_rate", dimension="overall",
                     value=None, numerator=0, denominator=0, context='{"fully":0,"partial":0,"not":0}')],
        cycle_id="cycle_empty",
    )
    assert count_rows() == 1


def test_save_metrics_empty_list_is_noop(in_memory_engine):
    from knowledge.eval_metric_store import save_metrics, count_rows
    save_metrics([], cycle_id="cycle_a")
    assert count_rows() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_metric_store.py -v`
Expected: 3 new tests FAIL with `ModuleNotFoundError: knowledge.eval_metric_store`.

- [ ] **Step 3: Create the store module**

Create `knowledge/eval_metric_store.py`:

```python
"""CRUD for EvalMetric — generic time-series store for eval-harness metrics (SP1+)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlmodel import Session, select, func

from core.database import get_engine
from core.models import EvalMetric


@dataclass
class MetricPoint:
    """In-memory shape of one metric row; produced by tally(), consumed by save_metrics()."""
    metric: str
    dimension: str
    value: Optional[float]
    numerator: int
    denominator: int
    context: str = "{}"


def save_metrics(points: list[MetricPoint], cycle_id: str) -> None:
    """Persist a snapshot's rows in one session. No-op on empty input."""
    if not points:
        return
    now = datetime.utcnow()
    rows = [
        EvalMetric(
            id=str(uuid.uuid4()),
            metric=p.metric,
            dimension=p.dimension,
            value=p.value,
            numerator=p.numerator,
            denominator=p.denominator,
            cycle_id=cycle_id,
            recorded_at=now,
            context=p.context,
        )
        for p in points
    ]
    with Session(get_engine(), expire_on_commit=False) as session:
        for r in rows:
            session.add(r)
        session.commit()


def count_rows() -> int:
    """Total rows in eval_metric. Used to gate the lazy backfill."""
    with Session(get_engine()) as session:
        return session.exec(select(func.count()).select_from(EvalMetric)).one()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_metric_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add knowledge/eval_metric_store.py tests/test_eval_metric_store.py
git commit -m "feat(eval): MetricPoint dataclass + save_metrics + count_rows"
```

---

### Task 3: `get_latest`, `get_previous`, `get_trend`

**Files:**
- Modify: `knowledge/eval_metric_store.py`
- Test: `tests/test_eval_metric_store.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_eval_metric_store.py`:

```python
def _seed_three_cycles(engine):
    """Three cycles, increasing repro rate; for trend/latest/previous tests."""
    from knowledge.eval_metric_store import MetricPoint, save_metrics
    for cid, num, den in [("c1", 1, 4), ("c2", 2, 4), ("c3", 3, 4)]:
        save_metrics([
            MetricPoint(metric="reproduction_rate", dimension="overall",
                        value=num/den, numerator=num, denominator=den, context="{}"),
        ], cycle_id=cid)


def test_get_latest_returns_most_recent(in_memory_engine):
    from knowledge.eval_metric_store import get_latest
    _seed_three_cycles(in_memory_engine)
    latest = get_latest("reproduction_rate", "overall")
    assert latest is not None
    assert latest.cycle_id == "c3"
    assert latest.numerator == 3


def test_get_latest_returns_none_for_unknown(in_memory_engine):
    from knowledge.eval_metric_store import get_latest
    assert get_latest("reproduction_rate") is None


def test_get_previous_skips_target_cycle(in_memory_engine):
    from knowledge.eval_metric_store import get_previous
    _seed_three_cycles(in_memory_engine)
    prev = get_previous("reproduction_rate", "overall", before_cycle_id="c3")
    assert prev is not None
    assert prev.cycle_id == "c2"


def test_get_previous_handles_first_cycle(in_memory_engine):
    from knowledge.eval_metric_store import get_previous, MetricPoint, save_metrics
    save_metrics(
        [MetricPoint(metric="reproduction_rate", dimension="overall",
                     value=0.5, numerator=1, denominator=2)],
        cycle_id="only_one",
    )
    assert get_previous("reproduction_rate", "overall", before_cycle_id="only_one") is None


def test_get_trend_returns_oldest_first(in_memory_engine):
    from knowledge.eval_metric_store import get_trend
    _seed_three_cycles(in_memory_engine)
    trend = get_trend("reproduction_rate", "overall", limit=10)
    assert [r.cycle_id for r in trend] == ["c1", "c2", "c3"]


def test_get_trend_respects_limit(in_memory_engine):
    from knowledge.eval_metric_store import get_trend
    _seed_three_cycles(in_memory_engine)
    trend = get_trend("reproduction_rate", "overall", limit=2)
    # Limit applies to the most-recent N, returned oldest-first.
    assert [r.cycle_id for r in trend] == ["c2", "c3"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_eval_metric_store.py -v`
Expected: 6 new tests FAIL with `ImportError`.

- [ ] **Step 3: Implement the query functions**

Append to `knowledge/eval_metric_store.py`:

```python
def get_latest(metric: str, dimension: str = "overall") -> Optional[EvalMetric]:
    """Most recent row for (metric, dimension). None if no rows."""
    with Session(get_engine()) as session:
        return session.exec(
            select(EvalMetric)
            .where(EvalMetric.metric == metric)
            .where(EvalMetric.dimension == dimension)
            .order_by(EvalMetric.recorded_at.desc())
            .limit(1)
        ).first()


def get_previous(
    metric: str, dimension: str, before_cycle_id: str
) -> Optional[EvalMetric]:
    """Most recent row for (metric, dimension) excluding the given cycle_id."""
    with Session(get_engine()) as session:
        return session.exec(
            select(EvalMetric)
            .where(EvalMetric.metric == metric)
            .where(EvalMetric.dimension == dimension)
            .where(EvalMetric.cycle_id != before_cycle_id)
            .order_by(EvalMetric.recorded_at.desc())
            .limit(1)
        ).first()


def get_trend(
    metric: str, dimension: str = "overall", limit: int = 30
) -> list[EvalMetric]:
    """Most recent N snapshots, returned oldest-first (for charts/sparklines)."""
    with Session(get_engine()) as session:
        rows = list(session.exec(
            select(EvalMetric)
            .where(EvalMetric.metric == metric)
            .where(EvalMetric.dimension == dimension)
            .order_by(EvalMetric.recorded_at.desc())
            .limit(limit)
        ).all())
    return list(reversed(rows))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_eval_metric_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add knowledge/eval_metric_store.py tests/test_eval_metric_store.py
git commit -m "feat(eval): get_latest, get_previous, get_trend queries"
```

---

## Chunk 2: Compute layer

### Task 4: `VerdictRow` + pure `tally()`

**Files:**
- Create: `analysis/reproduction_metrics.py`
- Test: `tests/test_reproduction_metrics.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_reproduction_metrics.py`:

```python
"""Tests for analysis/reproduction_metrics.py."""
from __future__ import annotations

from datetime import datetime


def _row(overall, difficulty="medium", target="local", source="arxiv"):
    from analysis.reproduction_metrics import VerdictRow
    return VerdictRow(
        overall=overall, difficulty=difficulty, target=target,
        source=source, recorded_at=datetime.utcnow(),
    )


def _by(points, metric, dimension):
    return next((p for p in points if p.metric == metric and p.dimension == dimension), None)


def test_tally_happy_path():
    from analysis.reproduction_metrics import tally
    rows = [
        _row("fully_reproduced"), _row("fully_reproduced"),
        _row("partially_reproduced"),
        _row("not_reproduced"),
    ]
    points = tally(rows)
    overall_repro = _by(points, "reproduction_rate", "overall")
    overall_partial = _by(points, "partial_rate", "overall")
    assert overall_repro.numerator == 2
    assert overall_repro.denominator == 4
    assert abs(overall_repro.value - 0.5) < 1e-9
    assert overall_partial.numerator == 1
    assert overall_partial.denominator == 4
    assert abs(overall_partial.value - 0.25) < 1e-9


def test_tally_empty_rows_emits_overall_with_none_value():
    from analysis.reproduction_metrics import tally
    points = tally([])
    overall = _by(points, "reproduction_rate", "overall")
    assert overall is not None
    assert overall.value is None
    assert overall.numerator == 0
    assert overall.denominator == 0


def test_tally_skips_unknown_dimension_buckets():
    """'unknown'/empty dimension values must NOT create sub-buckets but still count toward overall."""
    from analysis.reproduction_metrics import tally
    rows = [_row("fully_reproduced", difficulty="unknown", source="")]
    points = tally(rows)
    overall = _by(points, "reproduction_rate", "overall")
    assert overall.denominator == 1
    # No bucket for difficulty:unknown
    assert _by(points, "reproduction_rate", "difficulty:unknown") is None
    assert not any(p.dimension.startswith("difficulty:") for p in points)
    assert not any(p.dimension.startswith("source:") for p in points)


def test_tally_dimensional_bucketing():
    from analysis.reproduction_metrics import tally
    rows = [
        _row("fully_reproduced", difficulty="easy", target="local", source="arxiv"),
        _row("not_reproduced", difficulty="easy", target="local", source="arxiv"),
        _row("fully_reproduced", difficulty="hard", target="cloud_modal", source="semantic_scholar"),
    ]
    points = tally(rows)
    easy = _by(points, "reproduction_rate", "difficulty:easy")
    hard = _by(points, "reproduction_rate", "difficulty:hard")
    local = _by(points, "reproduction_rate", "target:local")
    modal = _by(points, "reproduction_rate", "target:cloud_modal")
    arxiv = _by(points, "reproduction_rate", "source:arxiv")
    assert easy.numerator == 1 and easy.denominator == 2
    assert hard.numerator == 1 and hard.denominator == 1
    assert local.numerator == 1 and local.denominator == 2
    assert modal.numerator == 1 and modal.denominator == 1
    assert arxiv.numerator == 1 and arxiv.denominator == 2


def test_tally_context_json_records_breakdown():
    import json
    from analysis.reproduction_metrics import tally
    rows = [
        _row("fully_reproduced"),
        _row("fully_reproduced"),
        _row("partially_reproduced"),
        _row("not_reproduced"),
    ]
    points = tally(rows)
    overall = _by(points, "reproduction_rate", "overall")
    ctx = json.loads(overall.context)
    assert ctx == {"fully": 2, "partial": 1, "not": 1}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reproduction_metrics.py -v`
Expected: 5 tests FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `VerdictRow` and `tally`**

Create `analysis/reproduction_metrics.py`:

```python
"""Reproduction-rate computation + cycle-snapshot orchestration (SP1 of Eval Harness).

See docs/superpowers/specs/2026-05-22-reproduction-rate-tracking-design.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import structlog

from knowledge.eval_metric_store import MetricPoint

log = structlog.get_logger()

_COMPARABLE = {"fully_reproduced", "partially_reproduced", "not_reproduced"}
_UNKNOWN = {"", "unknown", None}


@dataclass
class VerdictRow:
    """One experiment-result verdict, denormalized for tallying."""
    overall: str
    difficulty: str
    target: str
    source: str
    recorded_at: datetime


def tally(rows: list[VerdictRow]) -> list[MetricPoint]:
    """Pure: bucket rows by every dimension; compute reproduction_rate + partial_rate per bucket.

    Always emits the "overall" point pair, even on empty input (denominator=0, value=None).
    Sub-dimension buckets are skipped when the key is unknown/empty.
    """
    points: list[MetricPoint] = []
    points.extend(_compute_for_bucket(rows, dimension="overall"))

    for dim_name, getter in (
        ("difficulty", lambda r: r.difficulty),
        ("target", lambda r: r.target),
        ("source", lambda r: r.source),
    ):
        groups: dict[str, list[VerdictRow]] = {}
        for r in rows:
            key = getter(r)
            if key in _UNKNOWN:
                continue
            groups.setdefault(key, []).append(r)
        for key, group_rows in groups.items():
            points.extend(_compute_for_bucket(group_rows, dimension=f"{dim_name}:{key}"))

    return points


def _compute_for_bucket(rows: list[VerdictRow], dimension: str) -> list[MetricPoint]:
    comparable = [r for r in rows if r.overall in _COMPARABLE]
    n = len(comparable)
    fully = sum(1 for r in comparable if r.overall == "fully_reproduced")
    partial = sum(1 for r in comparable if r.overall == "partially_reproduced")
    not_repro = n - fully - partial

    repro_value: Optional[float] = (fully / n) if n else None
    partial_value: Optional[float] = (partial / n) if n else None
    context = json.dumps({"fully": fully, "partial": partial, "not": not_repro})

    return [
        MetricPoint(
            metric="reproduction_rate", dimension=dimension,
            value=repro_value, numerator=fully, denominator=n, context=context,
        ),
        MetricPoint(
            metric="partial_rate", dimension=dimension,
            value=partial_value, numerator=partial, denominator=n, context=context,
        ),
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reproduction_metrics.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/reproduction_metrics.py tests/test_reproduction_metrics.py
git commit -m "feat(eval): VerdictRow + pure tally() for repro-rate computation"
```

---

### Task 5: `gather_verdicts` — DB-side denormalization

**Files:**
- Modify: `analysis/reproduction_metrics.py`
- Test: `tests/test_reproduction_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reproduction_metrics.py`:

```python
def _seed_paper_experiment_result(
    engine, *, paper_id="p1", source="arxiv", difficulty="easy",
    experiment_id="e1", target="local", parent_id=None,
    overall="fully_reproduced", with_baseline_comparison=True,
):
    """Seed one Paper + one PaperAnalysis + one Experiment + one ExperimentResult."""
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import Paper, PaperAnalysis, Experiment, ExperimentResult

    with Session(engine, expire_on_commit=False) as session:
        if not session.get(Paper, paper_id):
            session.add(Paper(
                id=paper_id, title=f"Paper {paper_id}", abstract="",
                source=source, source_id=paper_id, url="http://x", pdf_url=None,
                published_date=date(2025, 1, 1), full_text=None,
            ))
            session.add(PaperAnalysis(
                id=f"analysis_{paper_id}", paper_id=paper_id,
                reproducibility_difficulty=difficulty,
            ))
        session.add(Experiment(
            id=experiment_id, paper_id=paper_id, title=f"Exp {experiment_id}",
            hypothesis="h", execution_target=target, status="completed",
            parent_experiment_id=parent_id,
        ))
        bc = json.dumps({"overall": overall, "comparisons": []}) if with_baseline_comparison else None
        session.add(ExperimentResult(
            id=f"result_{experiment_id}", experiment_id=experiment_id,
            exit_code=0, metrics="{}", baseline_comparison=bc,
            recorded_at=datetime.utcnow(),
        ))
        session.commit()


def test_gather_verdicts_returns_rows_for_completed_with_baseline(in_memory_engine):
    from analysis.reproduction_metrics import gather_verdicts

    _seed_paper_experiment_result(in_memory_engine, experiment_id="e_full",
                                  overall="fully_reproduced", difficulty="easy",
                                  target="local", source="arxiv")
    _seed_paper_experiment_result(in_memory_engine, paper_id="p2",
                                  experiment_id="e_not", overall="not_reproduced",
                                  difficulty="hard", target="cloud_modal", source="semantic_scholar")

    rows = gather_verdicts()
    assert {r.overall for r in rows} == {"fully_reproduced", "not_reproduced"}
    assert {r.difficulty for r in rows} == {"easy", "hard"}
    assert {r.target for r in rows} == {"local", "cloud_modal"}
    assert {r.source for r in rows} == {"arxiv", "semantic_scholar"}


def test_gather_verdicts_excludes_ablations(in_memory_engine):
    from analysis.reproduction_metrics import gather_verdicts
    _seed_paper_experiment_result(in_memory_engine, experiment_id="parent",
                                  overall="fully_reproduced")
    _seed_paper_experiment_result(in_memory_engine, experiment_id="abl",
                                  overall="fully_reproduced", parent_id="parent")
    rows = gather_verdicts()
    assert len(rows) == 1  # only the parent


def test_gather_verdicts_excludes_no_baseline_comparison(in_memory_engine):
    from analysis.reproduction_metrics import gather_verdicts
    _seed_paper_experiment_result(in_memory_engine, experiment_id="no_bc",
                                  with_baseline_comparison=False)
    assert gather_verdicts() == []


def test_gather_verdicts_excludes_non_comparable_verdicts(in_memory_engine):
    """Results with overall='no_experiments' (or {status: ...}) must not appear."""
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import Paper, PaperAnalysis, Experiment, ExperimentResult
    from analysis.reproduction_metrics import gather_verdicts

    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(Paper(id="p", title="t", abstract="", source="arxiv",
                          source_id="p", url="x", published_date=date(2025,1,1)))
        session.add(PaperAnalysis(id="a", paper_id="p"))
        session.add(Experiment(id="e", paper_id="p", title="t", hypothesis="h",
                               execution_target="local", status="completed"))
        session.add(ExperimentResult(
            id="r", experiment_id="e", exit_code=0, metrics="{}",
            baseline_comparison=json.dumps({"status": "no_analysis"}),
            recorded_at=datetime.utcnow(),
        ))
        session.commit()
    assert gather_verdicts() == []


def test_gather_verdicts_filters_by_experiment_ids(in_memory_engine):
    from analysis.reproduction_metrics import gather_verdicts
    _seed_paper_experiment_result(in_memory_engine, experiment_id="keep")
    _seed_paper_experiment_result(in_memory_engine, paper_id="p2",
                                  experiment_id="drop", overall="not_reproduced")
    rows = gather_verdicts(experiment_ids=["keep"])
    assert len(rows) == 1
    assert rows[0].overall == "fully_reproduced"


def test_gather_verdicts_tolerates_missing_analysis(in_memory_engine):
    """An experiment whose paper has no PaperAnalysis row still produces a verdict, with difficulty='unknown'."""
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import Paper, Experiment, ExperimentResult
    from analysis.reproduction_metrics import gather_verdicts

    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(Paper(id="p", title="t", abstract="", source="arxiv",
                          source_id="p", url="x", published_date=date(2025,1,1)))
        session.add(Experiment(id="e", paper_id="p", title="t", hypothesis="h",
                               execution_target="local", status="completed"))
        session.add(ExperimentResult(
            id="r", experiment_id="e", exit_code=0, metrics="{}",
            baseline_comparison=json.dumps({"overall": "fully_reproduced", "comparisons": []}),
            recorded_at=datetime.utcnow(),
        ))
        session.commit()

    rows = gather_verdicts()
    assert len(rows) == 1
    assert rows[0].difficulty == "unknown"
    assert rows[0].source == "arxiv"


def test_gather_verdicts_warns_on_malformed_baseline(in_memory_engine, caplog):
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import Paper, Experiment, ExperimentResult
    from analysis.reproduction_metrics import gather_verdicts

    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(Paper(id="p", title="t", abstract="", source="arxiv",
                          source_id="p", url="x", published_date=date(2025,1,1)))
        session.add(Experiment(id="e", paper_id="p", title="t", hypothesis="h",
                               execution_target="local", status="completed"))
        session.add(ExperimentResult(
            id="r", experiment_id="e", exit_code=0, metrics="{}",
            baseline_comparison="not json{{{",
            recorded_at=datetime.utcnow(),
        ))
        session.commit()
    assert gather_verdicts() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reproduction_metrics.py -v -k gather_verdicts`
Expected: 7 FAIL with `ImportError: cannot import name 'gather_verdicts'`.

- [ ] **Step 3: Implement `gather_verdicts`**

Append to `analysis/reproduction_metrics.py`:

```python
def gather_verdicts(experiment_ids: Optional[Iterable[str]] = None) -> list[VerdictRow]:
    """Join ExperimentResult x Experiment x PaperAnalysis x Paper into VerdictRows.

    Filters:
      * non-ablation experiments only (parent_experiment_id IS NULL)
      * results with a baseline_comparison whose `overall` is one of the 3 real verdicts
      * if `experiment_ids` is given, restrict to that set

    Missing PaperAnalysis → difficulty='unknown'. Missing Paper → source='unknown'.
    Malformed baseline_comparison JSON → logged WARN and skipped.
    """
    from sqlmodel import Session, select
    from core.database import get_engine
    from core.models import Experiment, ExperimentResult, Paper, PaperAnalysis

    with Session(get_engine()) as session:
        exp_stmt = select(Experiment).where(Experiment.parent_experiment_id == None)  # noqa: E711
        if experiment_ids is not None:
            ids = list(experiment_ids)
            if not ids:
                return []
            exp_stmt = exp_stmt.where(Experiment.id.in_(ids))
        experiments = list(session.exec(exp_stmt).all())
        if not experiments:
            return []

        exp_by_id = {e.id: e for e in experiments}
        paper_ids = {e.paper_id for e in experiments}

        results = list(session.exec(
            select(ExperimentResult).where(ExperimentResult.experiment_id.in_(list(exp_by_id)))
        ).all())

        papers = {p.id: p for p in session.exec(
            select(Paper).where(Paper.id.in_(list(paper_ids)))
        ).all()}
        analyses = {a.paper_id: a for a in session.exec(
            select(PaperAnalysis).where(PaperAnalysis.paper_id.in_(list(paper_ids)))
        ).all()}

    out: list[VerdictRow] = []
    for r in results:
        if not r.baseline_comparison:
            continue
        try:
            bc = json.loads(r.baseline_comparison)
        except (json.JSONDecodeError, TypeError):
            log.warning("reproduction_metrics.malformed_baseline_comparison", result_id=r.id)
            continue
        overall = bc.get("overall")
        if overall not in _COMPARABLE:
            continue
        exp = exp_by_id.get(r.experiment_id)
        if not exp:
            continue
        paper = papers.get(exp.paper_id)
        analysis = analyses.get(exp.paper_id)
        out.append(VerdictRow(
            overall=overall,
            difficulty=(analysis.reproducibility_difficulty if analysis else "unknown"),
            target=exp.execution_target,
            source=(paper.source if paper else "unknown"),
            recorded_at=r.recorded_at,
        ))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reproduction_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/reproduction_metrics.py tests/test_reproduction_metrics.py
git commit -m "feat(eval): gather_verdicts joins ExperimentResult x Experiment x Paper(Analysis)"
```

---

### Task 6: `backfill_from_history` — ISO-week bucketing + idempotency

**Files:**
- Modify: `analysis/reproduction_metrics.py`
- Test: `tests/test_reproduction_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reproduction_metrics.py`:

```python
def _override_recorded_at(engine, result_id: str, when):
    from sqlmodel import Session
    from core.models import ExperimentResult
    with Session(engine, expire_on_commit=False) as session:
        r = session.get(ExperimentResult, result_id)
        r.recorded_at = when
        session.add(r)
        session.commit()


def test_backfill_buckets_by_iso_week(in_memory_engine):
    from datetime import datetime
    from analysis.reproduction_metrics import backfill_from_history
    from knowledge.eval_metric_store import get_trend

    # Week 2025-W10 vs 2025-W20 (well-separated weeks).
    _seed_paper_experiment_result(in_memory_engine, experiment_id="e_a", overall="fully_reproduced")
    _override_recorded_at(in_memory_engine, "result_e_a", datetime(2025, 3, 5))   # W10
    _seed_paper_experiment_result(in_memory_engine, paper_id="p2", experiment_id="e_b",
                                  overall="not_reproduced")
    _override_recorded_at(in_memory_engine, "result_e_b", datetime(2025, 5, 14))  # W20

    written = backfill_from_history()
    assert written > 0
    trend = get_trend("reproduction_rate", "overall", limit=10)
    cycle_ids = {row.cycle_id for row in trend}
    assert "backfill-2025-W10" in cycle_ids
    assert "backfill-2025-W20" in cycle_ids


def test_backfill_is_idempotent(in_memory_engine):
    from datetime import datetime
    from analysis.reproduction_metrics import backfill_from_history
    from knowledge.eval_metric_store import count_rows

    _seed_paper_experiment_result(in_memory_engine, experiment_id="e1", overall="fully_reproduced")
    _override_recorded_at(in_memory_engine, "result_e1", datetime(2025, 3, 5))

    backfill_from_history()
    n = count_rows()
    backfill_from_history()
    assert count_rows() == n  # second invocation writes nothing


def test_backfill_skips_empty_weeks(in_memory_engine):
    """If no comparable rows exist, no backfill rows are written."""
    from analysis.reproduction_metrics import backfill_from_history
    from knowledge.eval_metric_store import count_rows
    written = backfill_from_history()
    assert written == 0
    assert count_rows() == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reproduction_metrics.py -v -k backfill`
Expected: 3 FAIL with `ImportError: cannot import name 'backfill_from_history'`.

- [ ] **Step 3: Implement backfill**

Append to `analysis/reproduction_metrics.py`:

```python
def backfill_from_history() -> int:
    """One-shot: bucket all historical comparable results by ISO week of recorded_at.

    Idempotent: existing backfill-* cycle_ids are skipped. Returns the count of rows written.
    """
    from sqlmodel import Session, select
    from core.database import get_engine
    from core.models import EvalMetric
    from knowledge.eval_metric_store import save_metrics

    rows = gather_verdicts(experiment_ids=None)
    if not rows:
        return 0

    # Bucket by ISO week
    buckets: dict[str, list[VerdictRow]] = {}
    for r in rows:
        iso_year, iso_week, _ = r.recorded_at.isocalendar()
        cycle_id = f"backfill-{iso_year}-W{iso_week:02d}"
        buckets.setdefault(cycle_id, []).append(r)

    # Idempotency: skip cycle_ids already present.
    # NOTE: SQLModel single-column `select(EvalMetric.cycle_id)` returns SCALARS, not tuples
    # (see CLAUDE.md "Bugs Fixed" #2). Do NOT index `row[0]`.
    with Session(get_engine()) as session:
        existing = set(session.exec(
            select(EvalMetric.cycle_id).where(EvalMetric.cycle_id.in_(list(buckets)))
        ).all())

    written = 0
    for cycle_id, week_rows in sorted(buckets.items()):
        if cycle_id in existing:
            continue
        points = tally(week_rows)
        overall_repro = next(
            (p for p in points if p.metric == "reproduction_rate" and p.dimension == "overall"),
            None,
        )
        if overall_repro is None or overall_repro.denominator == 0:
            continue  # nothing comparable this week
        save_metrics(points, cycle_id=cycle_id)
        written += len(points)
    log.info("reproduction_metrics.backfill_done", rows_written=written)
    return written
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reproduction_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/reproduction_metrics.py tests/test_reproduction_metrics.py
git commit -m "feat(eval): backfill_from_history with ISO-week bucketing + idempotency"
```

---

### Task 7: `record_cycle_snapshot` orchestrator with lazy backfill

**Files:**
- Modify: `analysis/reproduction_metrics.py`
- Test: `tests/test_reproduction_metrics.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_reproduction_metrics.py`:

```python
def _make_state(cycle_id="cycle_x", experiment_ids=None):
    from core.models import RunState
    from datetime import datetime
    return RunState(
        cycle_id=cycle_id, started_at=datetime.utcnow(),
        experiment_ids_this_cycle=list(experiment_ids or []),
    )


def test_record_cycle_snapshot_writes_overall_and_dimensions(in_memory_engine):
    from analysis.reproduction_metrics import record_cycle_snapshot
    from knowledge.eval_metric_store import get_latest

    _seed_paper_experiment_result(in_memory_engine, experiment_id="e1",
                                  overall="fully_reproduced", difficulty="easy",
                                  target="local", source="arxiv")
    state = _make_state(cycle_id="cycle_one", experiment_ids=["e1"])
    record_cycle_snapshot(state)

    overall = get_latest("reproduction_rate", "overall")
    by_difficulty = get_latest("reproduction_rate", "difficulty:easy")
    assert overall is not None and overall.numerator == 1 and overall.denominator == 1
    assert by_difficulty is not None and by_difficulty.numerator == 1


def test_record_cycle_snapshot_empty_cycle_writes_none_overall(in_memory_engine):
    """A cycle with no comparable experiments still records the gap honestly."""
    from analysis.reproduction_metrics import record_cycle_snapshot
    from knowledge.eval_metric_store import get_latest

    state = _make_state(cycle_id="empty", experiment_ids=[])
    record_cycle_snapshot(state)

    overall = get_latest("reproduction_rate", "overall")
    assert overall is not None
    assert overall.value is None
    assert overall.denominator == 0


def test_record_cycle_snapshot_triggers_lazy_backfill_when_empty(in_memory_engine):
    from datetime import datetime
    from analysis.reproduction_metrics import record_cycle_snapshot
    from knowledge.eval_metric_store import get_trend

    # Seed historical results from a prior week BEFORE the cycle's experiment.
    _seed_paper_experiment_result(in_memory_engine, paper_id="p_hist", experiment_id="e_hist",
                                  overall="fully_reproduced")
    _override_recorded_at(in_memory_engine, "result_e_hist", datetime(2025, 3, 5))  # W10

    # Cycle's own experiment.
    _seed_paper_experiment_result(in_memory_engine, paper_id="p_now", experiment_id="e_now",
                                  overall="not_reproduced")

    state = _make_state(cycle_id="now_cycle", experiment_ids=["e_now"])
    record_cycle_snapshot(state)

    trend = get_trend("reproduction_rate", "overall", limit=20)
    cycle_ids = [r.cycle_id for r in trend]
    assert "backfill-2025-W10" in cycle_ids
    assert "now_cycle" in cycle_ids


def test_record_cycle_snapshot_does_not_re_backfill(in_memory_engine):
    from analysis.reproduction_metrics import record_cycle_snapshot, backfill_from_history
    from knowledge.eval_metric_store import count_rows
    from datetime import datetime

    _seed_paper_experiment_result(in_memory_engine, paper_id="p_h", experiment_id="e_h",
                                  overall="fully_reproduced")
    _override_recorded_at(in_memory_engine, "result_e_h", datetime(2025, 3, 5))
    backfill_from_history()

    rows_after_backfill = count_rows()
    _seed_paper_experiment_result(in_memory_engine, paper_id="p_n", experiment_id="e_n",
                                  overall="not_reproduced")
    record_cycle_snapshot(_make_state(cycle_id="c", experiment_ids=["e_n"]))

    # Only cycle's own rows added; backfill rows not duplicated.
    assert count_rows() > rows_after_backfill  # cycle rows added
    # And re-invoking does not produce more backfill rows.
    record_cycle_snapshot(_make_state(cycle_id="c2", experiment_ids=[]))
    # cycle c2 adds only the overall=None row(s); backfill rows still present once.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reproduction_metrics.py -v -k record_cycle_snapshot`
Expected: 4 FAIL with `ImportError`.

- [ ] **Step 3: Implement `record_cycle_snapshot`**

Append to `analysis/reproduction_metrics.py`:

```python
def record_cycle_snapshot(state) -> None:
    """End-of-stage hook: lazy-backfill if store empty, then tally this cycle's experiments and save.

    `state` is a RunState. Empty `experiment_ids_this_cycle` still writes overall=None
    so the trend honestly records the gap.
    """
    from knowledge.eval_metric_store import count_rows, save_metrics

    if count_rows() == 0:
        backfill_from_history()

    verdicts = gather_verdicts(experiment_ids=state.experiment_ids_this_cycle)
    points = tally(verdicts)
    save_metrics(points, cycle_id=state.cycle_id)
    log.info(
        "reproduction_metrics.snapshot_recorded",
        cycle_id=state.cycle_id, points_written=len(points),
        verdict_rows=len(verdicts),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reproduction_metrics.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/reproduction_metrics.py tests/test_reproduction_metrics.py
git commit -m "feat(eval): record_cycle_snapshot orchestrator with lazy backfill"
```

---

## Chunk 3: Integration & surfaces

### Task 8: Wire snapshot into `analysis_pipeline.run`

**Files:**
- Modify: `analysis/analysis_pipeline.py:67-68`
- Test: `tests/test_reproduction_metrics.py`

- [ ] **Step 1: Write the failing integration test**

Append to `tests/test_reproduction_metrics.py`:

```python
def test_analysis_pipeline_records_snapshot(in_memory_engine, monkeypatch):
    """After analysis_pipeline.run completes, a snapshot row exists for the cycle."""
    from analysis import analysis_pipeline
    from knowledge.eval_metric_store import get_latest

    # Disable Claude conclusion generation (no API key in tests, no network).
    monkeypatch.setattr(analysis_pipeline, "_generate_conclusion", lambda *a, **k: "")

    _seed_paper_experiment_result(in_memory_engine, experiment_id="e1",
                                  overall="fully_reproduced", difficulty="easy")
    state = _make_state(cycle_id="cycle_ap", experiment_ids=["e1"])
    analysis_pipeline.run(state)

    latest = get_latest("reproduction_rate", "overall")
    assert latest is not None
    assert latest.cycle_id == "cycle_ap"


def test_analysis_pipeline_swallows_snapshot_failure(in_memory_engine, monkeypatch, caplog):
    """A failure inside record_cycle_snapshot must not abort analysis_pipeline.run."""
    from analysis import analysis_pipeline

    monkeypatch.setattr(analysis_pipeline, "_generate_conclusion", lambda *a, **k: "")

    def boom(_state):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(
        "analysis.reproduction_metrics.record_cycle_snapshot", boom
    )
    # Must not raise.
    analysis_pipeline.run(_make_state(cycle_id="c", experiment_ids=[]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_reproduction_metrics.py -v -k analysis_pipeline`
Expected: 2 FAIL (`test_analysis_pipeline_records_snapshot`: `get_latest` returns None; the swallow test passes accidentally because nothing's wired yet but the first one fails).

- [ ] **Step 3: Wire into `analysis_pipeline.run`**

Modify the tail of `analysis/analysis_pipeline.py` `run()`. Replace lines 67–68:

```python
    save_state(state)
    log.info("analysis_pipeline.done")
```

with:

```python
    save_state(state)

    # Eval-harness SP1: persist this cycle's reproduction-rate snapshot.
    # Failure here must not abort the pipeline (mirrors contradiction_detector pattern).
    try:
        from analysis import reproduction_metrics
        reproduction_metrics.record_cycle_snapshot(state)
    except Exception as e:
        log.error("analysis_pipeline.snapshot_error", error=str(e))

    log.info("analysis_pipeline.done")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reproduction_metrics.py -v`
Expected: all PASS.

Run the full suite to make sure nothing else regressed: `uv run pytest tests/ -v --tb=short`
Expected: all PASS (39 prior + new tests).

- [ ] **Step 5: Commit**

```bash
git add analysis/analysis_pipeline.py tests/test_reproduction_metrics.py
git commit -m "feat(eval): wire record_cycle_snapshot into analysis_pipeline.run"
```

---

### Task 9: Store-backed reproduction-rate in `report_generator`

**Files:**
- Modify: `reporting/report_generator.py:307-340`
- Test: covered by template integration test in Task 10.

- [ ] **Step 1: Replace the ad-hoc math with a store lookup + delta**

In `reporting/report_generator.py`, locate lines 307–313:

```python
    # Count stats
    total_papers = len(get_all_papers(limit=10000))
    total_experiments = len(all_completed)
    reproduced = sum(
        1 for s in exp_sections
        if s["baseline_status"] in ("fully_reproduced", "partially_reproduced")
    )
    reproduction_rate = round(reproduced / max(len(exp_sections), 1) * 100, 1)
```

Replace with:

```python
    # Count stats
    total_papers = len(get_all_papers(limit=10000))
    total_experiments = len(all_completed)

    # SP1: pull reproduction rate from the eval-metric store (replaces ad-hoc per-report math).
    from knowledge.eval_metric_store import get_latest, get_previous
    latest = get_latest("reproduction_rate", "overall")
    latest_partial = get_latest("partial_rate", "overall")
    previous = get_previous("reproduction_rate", "overall", before_cycle_id=state.cycle_id)

    if latest is None or latest.value is None:
        reproduction_rate = None
        repro_n = latest.denominator if latest else 0
    else:
        reproduction_rate = round(latest.value * 100, 1)
        repro_n = latest.denominator

    partial_rate = (
        round(latest_partial.value * 100, 1)
        if latest_partial and latest_partial.value is not None else None
    )

    if (latest and latest.value is not None
            and previous and previous.value is not None):
        delta_pp = round((latest.value - previous.value) * 100, 1)
    else:
        delta_pp = None
```

Then update the `template.render(...)` call (lines 321–340) so the new variables flow into the template. Add these three keyword arguments to the `template.render(...)` call:

```python
        repro_n=repro_n,
        partial_rate=partial_rate,
        delta_pp=delta_pp,
```

(Leave `reproduction_rate=reproduction_rate` — its semantics change from "always a number" to "Optional[float]"; the template handles `None`.)

- [ ] **Step 2: Run the existing test suite to confirm nothing broke yet**

Run: `uv run pytest tests/ -v --tb=short`
Expected: all PASS. (No new test in this task; template-level assertion comes in Task 10.)

- [ ] **Step 3: Commit**

```bash
git add reporting/report_generator.py
git commit -m "feat(eval): report_generator reads reproduction rate from store + delta"
```

---

### Task 10: Update report template

**Files:**
- Modify: `reporting/templates/weekly_report.md.j2:152-162`
- Test: `tests/test_reproduction_metrics.py` (new template-integration test)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_reproduction_metrics.py`:

```python
def test_report_generator_renders_store_backed_rate(in_memory_engine, monkeypatch, tmp_path):
    """End-to-end: snapshot then report; the rendered markdown contains the rigorous metric."""
    from reporting import report_generator
    from analysis.reproduction_metrics import record_cycle_snapshot
    from config import settings

    # Disable Claude narrative generation.
    monkeypatch.setattr(report_generator, "_generate_narrative", lambda **kw: {})
    monkeypatch.setattr(settings, "reports_dir", tmp_path)

    _seed_paper_experiment_result(in_memory_engine, experiment_id="e1",
                                  overall="fully_reproduced")
    state = _make_state(cycle_id="cycle_rpt", experiment_ids=["e1"])
    record_cycle_snapshot(state)

    report = report_generator.generate(state)
    md = report.markdown_content
    assert "Reproduction rate:" in md
    assert "100.0%" in md  # one of one comparable result = 100%
    assert "n=1" in md


def test_report_renders_dash_when_no_comparable(in_memory_engine, monkeypatch, tmp_path):
    from reporting import report_generator
    from analysis.reproduction_metrics import record_cycle_snapshot
    from config import settings

    monkeypatch.setattr(report_generator, "_generate_narrative", lambda **kw: {})
    monkeypatch.setattr(settings, "reports_dir", tmp_path)

    state = _make_state(cycle_id="cycle_empty", experiment_ids=[])
    record_cycle_snapshot(state)
    report = report_generator.generate(state)
    assert "Reproduction rate: —" in report.markdown_content
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reproduction_metrics.py -v -k report`
Expected: FAIL — old template renders e.g. "Reproduction rate: 100.0% — at least half…" without an `n=` annotation, and renders `0%` instead of `—` for the empty case.

- [ ] **Step 3: Rewrite the System Stats block**

In `reporting/templates/weekly_report.md.j2`, replace the block at lines 152–162:

```jinja
## System Stats

- The system has tracked **{{ total_papers }}** papers to date.
- **{{ total_experiments }}** experiments have been run and completed.
{% if reproduction_rate == 0 %}
- No experiments have matched their paper's claimed results yet.
{% elif reproduction_rate < 50 %}
- Reproduction rate: {{ reproduction_rate }}% — fewer than half of experiments matched their paper's claimed results.
{% else %}
- Reproduction rate: {{ reproduction_rate }}% — at least half of experiments matched their paper's claimed results.
{% endif %}
```

with:

```jinja
## System Stats

- The system has tracked **{{ total_papers }}** papers to date.
- **{{ total_experiments }}** experiments have been run and completed.
{% if reproduction_rate is none %}
- Reproduction rate: — (no comparable experiments this cycle).
{% else %}
- Reproduction rate: **{{ reproduction_rate }}%** (n={{ repro_n }}{% if partial_rate is not none %}, partial: {{ partial_rate }}%{% endif %}){% if delta_pp is not none %} — Δ vs last cycle: {% if delta_pp > 0 %}+{% endif %}{{ delta_pp }} pp{% endif %}.
{% endif %}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_reproduction_metrics.py -v -k report`
Expected: both PASS.

Run the full suite: `uv run pytest tests/ -v --tb=short`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add reporting/templates/weekly_report.md.j2 tests/test_reproduction_metrics.py
git commit -m "feat(eval): report template renders rigorous repro rate with n + delta"
```

---

### Task 11: Streamlit dashboard tile

**Files:**
- Modify: `ui/views/dashboard.py`

This task is UI; verification is manual (no Streamlit harness in the test suite — mirrors existing pattern).

- [ ] **Step 1: Add the fragment-cached tile helper**

Insert at the top of `ui/views/dashboard.py`, just after the existing `_get_paper_count` function (around line 39, before `_get_recent_cycles`):

```python
@st.cache_data(ttl=8)
def _get_reproduction_metric(dimension: str = "overall") -> dict:
    """Return latest reproduction rate + trend for the dashboard tile."""
    from knowledge.eval_metric_store import get_latest, get_trend
    latest = get_latest("reproduction_rate", dimension)
    latest_partial = get_latest("partial_rate", dimension)
    trend = get_trend("reproduction_rate", dimension, limit=12)
    return {
        "value": latest.value if latest else None,
        "n": latest.denominator if latest else 0,
        "partial": latest_partial.value if latest_partial else None,
        "trend": [{"cycle": r.cycle_id, "value": r.value} for r in trend if r.value is not None],
    }


def _reproduction_rate_tile() -> None:
    """Dashboard tile: big-number repro rate + sparkline + dimension selector."""
    import pandas as pd
    st.subheader("Reproduction Rate")

    dim = st.selectbox(
        "Slice by",
        options=[
            "overall",
            "difficulty:easy", "difficulty:medium", "difficulty:hard",
            "target:local", "target:cloud_modal",
            "source:arxiv", "source:semantic_scholar", "source:substack",
        ],
        index=0,
        key="repro_rate_dim",
    )
    data = _get_reproduction_metric(dim)

    c1, c2 = st.columns([1, 3])
    if data["value"] is None:
        c1.metric("Latest", "—")
    else:
        c1.metric("Latest", f"{data['value'] * 100:.1f}%")
    sub_bits = [f"n = {data['n']}"]
    if data["partial"] is not None:
        sub_bits.append(f"partial: {data['partial'] * 100:.1f}%")
    c1.caption(" • ".join(sub_bits))

    if data["trend"]:
        df = pd.DataFrame(data["trend"])
        df["value_pct"] = df["value"] * 100
        c2.line_chart(df, x="cycle", y="value_pct", height=160)
    else:
        c2.info("No comparable experiments recorded yet — trend will appear after the first cycle with a baseline comparison.")
```

- [ ] **Step 2: Render the tile inside `render()`**

In the same file, locate the existing Experiment Status block (around line 309–315) and add a divider + the new tile before it. Replace:

```python
    st.divider()

    # Experiment status
    st.subheader("Experiment Status")
```

with:

```python
    st.divider()

    # SP1 of the Eval Harness: north-star metric tile.
    _reproduction_rate_tile()

    st.divider()

    # Experiment status
    st.subheader("Experiment Status")
```

- [ ] **Step 3: Manual verification**

Run: `uv run python -m streamlit run ui/app.py --server.headless true --browser.gatherUsageStats false`
Open http://localhost:8501; confirm:
- The "Reproduction Rate" tile renders without error.
- The dimension selector includes all 9 options.
- With an empty store, the tile shows "—" and the "No comparable experiments…" hint.
- After running one full pipeline cycle (or invoking `record_cycle_snapshot` once), the tile shows a real percentage and `n=…`.

- [ ] **Step 4: Commit**

```bash
git add ui/views/dashboard.py
git commit -m "feat(eval): Streamlit dashboard tile for reproduction rate trend"
```

---

## Final verification

- [ ] **Run full test suite**

```bash
uv run pytest tests/ -v --tb=short
```

Expected: all original tests + new tests PASS. Exit code 0.

- [ ] **Smoke-test imports**

```bash
uv run python -c "
from core.models import EvalMetric
from knowledge.eval_metric_store import MetricPoint, save_metrics, get_latest, get_previous, get_trend, count_rows
from analysis.reproduction_metrics import VerdictRow, tally, gather_verdicts, record_cycle_snapshot, backfill_from_history
print('All SP1 imports OK')
"
```

Expected: `All SP1 imports OK`.

- [ ] **Confirm DB migration is automatic**

```bash
uv run python -c "from core.database import init_db; init_db(); print('OK')"
```

Expected: `OK` — the `eval_metric` table is created on the live DB without any ALTER migration.

- [ ] **Confirm the wired pipeline writes a snapshot** (optional, requires existing DB with completed experiments)

```bash
uv run python -c "
from core.models import RunState
from datetime import datetime
from analysis import reproduction_metrics
from knowledge.eval_metric_store import get_latest, count_rows
print('rows before:', count_rows())
state = RunState(cycle_id='smoke_test', started_at=datetime.utcnow(), experiment_ids_this_cycle=[])
reproduction_metrics.record_cycle_snapshot(state)
print('rows after:', count_rows())
latest = get_latest('reproduction_rate', 'overall')
print('latest cycle_id:', latest.cycle_id if latest else None, 'value:', latest.value if latest else None)
"
```

Expected: rows-after > rows-before; if the DB has historical comparable results, backfill rows are also present.

- [ ] **Update CLAUDE.md** (project doc) with a short SP1 entry

Append to the "Key Files Map" table in `D:/Research Agent/CLAUDE.md`:

```markdown
| `analysis/reproduction_metrics.py` | Reproduction-rate compute (SP1 of Eval Harness): tally, gather, snapshot, ISO-week backfill |
| `knowledge/eval_metric_store.py` | Generic EvalMetric time-series store — substrate for SP1-SP4 of the Eval Harness |
```

Also add to the Architecture section: "`analysis_pipeline.run` now calls `reproduction_metrics.record_cycle_snapshot(state)` at end-of-stage (wrapped in try/except)."

Commit:

```bash
git add CLAUDE.md
git commit -m "docs(claude.md): document SP1 eval-harness modules"
```

---

## Out of scope (do NOT add — these are later sub-projects)

- LLM-as-judge for analysis quality (SP3).
- Ground-truth benchmark set / labeling workflow (SP2).
- Calibration of novelty/relevance scores (SP4).
- Replacing Streamlit / FastAPI service / killing the CLI (separate platformization brainstorm).
- A new CLI `eval` command (deferred until the frontend-architecture decision).
- Cumulative-vs-time-window dashboards beyond the basic sparkline (YAGNI for SP1).
- Per-comparison granularity (we measure per-experiment-overall; per-metric-comparison can come later if needed).
