# Ground-Truth Benchmark Set Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a curated, human-labeled golden set of paper-claim ground truths (metric + expected value + tolerance), scored against the pipeline's latest measurements to produce a persisted, trended `benchmark_accuracy` metric with a full per-run audit trail.

**Architecture:** Approach B (full audit trail). Three new SQLModel tables (`BenchmarkItem`, `BenchmarkRun`, `BenchmarkItemResult`). A store (`knowledge/benchmark_store.py`) as the sole DB interface. A pure-core/thin-orchestrator scorer (`analysis/benchmark_scorer.py`) that reads each item's latest `ExperimentResult`, scores measured-vs-expected within tolerance, persists the run + per-item results, and writes aggregate `benchmark_accuracy` into the existing SP1 `EvalMetric` store. A Streamlit page (`ui/views/benchmark.py`) curates/runs/views. An auto-score hook in `analysis_pipeline.run` self-populates the trend each cycle. LLM-free.

**Tech Stack:** Python 3.11, SQLModel/SQLite (WAL), pytest, Streamlit, pandas. No new dependencies, no Claude calls.

**Spec:** `docs/superpowers/specs/2026-05-24-benchmark-set-design.md`

---

## File Structure

**New files**
- `knowledge/benchmark_store.py` — CRUD for the three tables. Sole interface.
- `analysis/benchmark_scorer.py` — `Measurement`/`ItemOutcome` dataclasses, `gather_measurements()` (DB read), pure `score()`, pure `build_metric_points()`, orchestrator `record_benchmark_run()`.
- `ui/views/benchmark.py` — Streamlit Benchmark page (curate / run / view).
- `tests/test_benchmark_store.py` — store CRUD tests.
- `tests/test_benchmark_scorer.py` — scoring + integration tests.

**Modified files**
- `core/models.py` — append `BenchmarkItem`, `BenchmarkRun`, `BenchmarkItemResult` after `EvalMetric`.
- `core/database.py` — add the three models to the `init_db` import line so `create_all` picks them up.
- `analysis/analysis_pipeline.py` — after the SP1 snapshot block (line ~75), add a try/except-wrapped benchmark auto-score call.
- `ui/app.py` — add `"◎  Benchmark"` to `PAGE_LABELS` (after Review Queue) + an `elif` route.

**No changes to** `core/state.py`, `scheduler/pipeline_runner.py`, the experiment/ingestion pipelines, `main.py` (no CLI in SP2), or `knowledge/eval_metric_store.py` (reused unchanged with a new `metric="benchmark_accuracy"` key).

---

## Pre-flight

- [ ] **Step 0a: Create feature branch** (current branch is `main`; never implement on `main`).

```bash
git checkout -b feat/eval-harness-sp2-benchmark-set
```

- [ ] **Step 0b: Confirm baseline — full test suite green before any changes.**

Run: `uv run python -m pytest tests/ -q --tb=short`
Expected: 78 passed, exit 0.

If any test fails on `main`, stop and surface to the user — do not start on a red baseline.

> **Windows/uv note:** use `uv run python -m pytest ...` (not `uv run pytest`), which fails with "Failed to canonicalize script path" on this machine.

---

## Chunk 1: Data layer (models + store)

### Task 1: Add the three SQLModel tables

**Files:**
- Modify: `core/models.py` (append after the `EvalMetric` class)
- Modify: `core/database.py` (the `init_db` import line)
- Test: `tests/test_benchmark_store.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_benchmark_store.py`:

```python
"""Tests for knowledge/benchmark_store.py and the benchmark SQLModels."""
from __future__ import annotations

from sqlmodel import Session, select


def test_benchmark_tables_are_created(in_memory_engine):
    from core.models import BenchmarkItem, BenchmarkRun, BenchmarkItemResult

    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(BenchmarkItem(
            id="i1", experiment_id="e1", metric_name="accuracy",
            expected_value=0.92, tolerance=0.05, tolerance_type="relative",
            unit=None, note="from paper table 2",
        ))
        session.add(BenchmarkRun(
            id="r1", trigger="manual", n_items=1, n_pass=1, n_fail=0,
            n_unscorable=0, accuracy=1.0,
        ))
        session.add(BenchmarkItemResult(
            id="ir1", run_id="r1", item_id="i1", experiment_id="e1",
            metric_name="accuracy", expected_value=0.92, tolerance=0.05,
            tolerance_type="relative", measured_value=0.93, passed=True, status="pass",
        ))
        session.commit()

    with Session(in_memory_engine) as session:
        items = list(session.exec(select(BenchmarkItem)).all())
        runs = list(session.exec(select(BenchmarkRun)).all())
        results = list(session.exec(select(BenchmarkItemResult)).all())
    assert items[0].expected_value == 0.92 and items[0].active is True
    assert runs[0].accuracy == 1.0
    assert results[0].passed is True and results[0].status == "pass"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_benchmark_store.py::test_benchmark_tables_are_created -v`
Expected: FAIL — `ImportError: cannot import name 'BenchmarkItem' from 'core.models'`.

- [ ] **Step 3: Add the models**

Append to `core/models.py` immediately after the `EvalMetric` class:

```python
class BenchmarkItem(SQLModel, table=True):
    __tablename__ = "benchmark_item"

    id: str = Field(primary_key=True)                 # uuid4
    experiment_id: str = Field(index=True)            # experiment this claim is scored against
    metric_name: str                                  # key in ExperimentResult.metrics JSON
    expected_value: float                             # the paper's known/true value
    tolerance: float                                  # band half-width
    tolerance_type: str = "relative"                  # "relative" (% of expected) | "absolute"
    unit: Optional[str] = None                        # display only
    note: str = ""                                    # provenance / why this is ground truth
    active: bool = True                               # deactivate instead of hard-delete
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


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
    accuracy: Optional[float] = None                   # n_pass / scorable; None if 0 scorable


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
    status: str                                        # pass|fail|no_result|missing_metric|non_numeric
```

Update the `init_db` import line in `core/database.py` (currently ends `..., TokenUsageLog, EvalMetric`):

```python
    from core.models import Paper, PaperAnalysis, Experiment, ExperimentResult, ResearchReport, Contradiction, ResearchGap, ThemeCluster, TokenUsageLog, EvalMetric, BenchmarkItem, BenchmarkRun, BenchmarkItemResult  # noqa: F401
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_benchmark_store.py::test_benchmark_tables_are_created -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add core/models.py core/database.py tests/test_benchmark_store.py
git commit -m "feat(bench): add BenchmarkItem/Run/ItemResult SQLModels"
```

---

### Task 2: `benchmark_store` — item CRUD

**Files:**
- Create: `knowledge/benchmark_store.py`
- Test: `tests/test_benchmark_store.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark_store.py`:

```python
def _item(item_id="i1", experiment_id="e1", active=True):
    from core.models import BenchmarkItem
    return BenchmarkItem(
        id=item_id, experiment_id=experiment_id, metric_name="accuracy",
        expected_value=0.9, tolerance=0.05, tolerance_type="relative", active=active,
    )


def test_save_and_get_items_active_only(in_memory_engine):
    from knowledge.benchmark_store import save_item, get_items
    save_item(_item("i1"))
    save_item(_item("i2", active=False))
    active = get_items(active_only=True)
    assert {i.id for i in active} == {"i1"}
    allitems = get_items(active_only=False)
    assert {i.id for i in allitems} == {"i1", "i2"}


def test_get_item_returns_none_for_unknown(in_memory_engine):
    from knowledge.benchmark_store import get_item
    assert get_item("nope") is None


def test_deactivate_item(in_memory_engine):
    from knowledge.benchmark_store import save_item, deactivate_item, get_items, get_item
    save_item(_item("i1"))
    deactivate_item("i1")
    assert get_items(active_only=True) == []
    assert get_item("i1").active is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_benchmark_store.py -v`
Expected: 3 new tests FAIL with `ModuleNotFoundError: knowledge.benchmark_store`.

- [ ] **Step 3: Create the store module (item CRUD)**

Create `knowledge/benchmark_store.py`:

```python
"""CRUD for the ground-truth benchmark set (SP2 of the Eval Harness). Sole DB interface."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Session, select

from core.database import get_engine
from core.models import BenchmarkItem, BenchmarkRun, BenchmarkItemResult


def save_item(item: BenchmarkItem) -> None:
    """Insert or update a golden-set item."""
    item.updated_at = datetime.utcnow()
    with Session(get_engine(), expire_on_commit=False) as session:
        session.merge(item)
        session.commit()


def get_items(active_only: bool = True) -> list[BenchmarkItem]:
    with Session(get_engine()) as session:
        stmt = select(BenchmarkItem)
        if active_only:
            stmt = stmt.where(BenchmarkItem.active == True)  # noqa: E712
        return list(session.exec(stmt).all())


def get_item(item_id: str) -> Optional[BenchmarkItem]:
    with Session(get_engine()) as session:
        return session.get(BenchmarkItem, item_id)


def deactivate_item(item_id: str) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        item = session.get(BenchmarkItem, item_id)
        if item:
            item.active = False
            item.updated_at = datetime.utcnow()
            session.add(item)
            session.commit()
```

> **Note:** `save_item` uses `session.merge` so it serves both insert and edit (the UI's add/edit form). `merge` upserts by primary key.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_benchmark_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add knowledge/benchmark_store.py tests/test_benchmark_store.py
git commit -m "feat(bench): benchmark_store item CRUD (save/get/deactivate)"
```

---

### Task 3: `benchmark_store` — run + item-result persistence

**Files:**
- Modify: `knowledge/benchmark_store.py`
- Test: `tests/test_benchmark_store.py` (extend)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark_store.py`:

```python
def _run(run_id="r1", cycle_id=None, accuracy=1.0):
    from core.models import BenchmarkRun
    return BenchmarkRun(id=run_id, cycle_id=cycle_id, trigger="manual",
                        n_items=1, n_pass=1, n_fail=0, n_unscorable=0, accuracy=accuracy)


def _result(res_id, run_id, item_id="i1", status="pass"):
    from core.models import BenchmarkItemResult
    return BenchmarkItemResult(
        id=res_id, run_id=run_id, item_id=item_id, experiment_id="e1",
        metric_name="accuracy", expected_value=0.9, tolerance=0.05,
        tolerance_type="relative", measured_value=0.91, passed=True, status=status,
    )


def test_save_run_persists_run_and_results(in_memory_engine):
    from knowledge.benchmark_store import save_run, get_runs, get_item_results
    save_run(_run("r1"), [_result("ir1", "r1"), _result("ir2", "r1", item_id="i2")])
    runs = get_runs()
    assert len(runs) == 1 and runs[0].id == "r1"
    assert len(get_item_results("r1")) == 2


def test_get_runs_orders_most_recent_first(in_memory_engine):
    from knowledge.benchmark_store import save_run, get_runs
    import time
    save_run(_run("r1"), [])
    time.sleep(0.01)
    save_run(_run("r2"), [])
    assert [r.id for r in get_runs()] == ["r2", "r1"]


def test_get_latest_run(in_memory_engine):
    from knowledge.benchmark_store import save_run, get_latest_run
    assert get_latest_run() is None
    save_run(_run("r1"), [])
    assert get_latest_run().id == "r1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_benchmark_store.py -v`
Expected: 3 new tests FAIL with `ImportError`.

- [ ] **Step 3: Implement run CRUD**

Append to `knowledge/benchmark_store.py`:

```python
def save_run(run: BenchmarkRun, item_results: list[BenchmarkItemResult]) -> None:
    """Persist a scoring run and all its per-item results in one session."""
    with Session(get_engine(), expire_on_commit=False) as session:
        session.add(run)
        for r in item_results:
            session.add(r)
        session.commit()


def get_runs(limit: int = 30) -> list[BenchmarkRun]:
    """Run history, most recent first."""
    with Session(get_engine()) as session:
        return list(session.exec(
            select(BenchmarkRun).order_by(BenchmarkRun.recorded_at.desc()).limit(limit)
        ).all())


def get_latest_run() -> Optional[BenchmarkRun]:
    with Session(get_engine()) as session:
        return session.exec(
            select(BenchmarkRun).order_by(BenchmarkRun.recorded_at.desc()).limit(1)
        ).first()


def get_item_results(run_id: str) -> list[BenchmarkItemResult]:
    with Session(get_engine()) as session:
        return list(session.exec(
            select(BenchmarkItemResult).where(BenchmarkItemResult.run_id == run_id)
        ).all())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_benchmark_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add knowledge/benchmark_store.py tests/test_benchmark_store.py
git commit -m "feat(bench): benchmark_store run + item-result persistence"
```

---

## Chunk 2: Scoring layer

### Task 4: dataclasses + pure `score()`

**Files:**
- Create: `analysis/benchmark_scorer.py`
- Test: `tests/test_benchmark_scorer.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_benchmark_scorer.py`:

```python
"""Tests for analysis/benchmark_scorer.py."""
from __future__ import annotations


def _measurement(measured, status="ok", expected=0.90, tol=0.05,
                 ttype="relative", difficulty="medium", source="arxiv"):
    from core.models import BenchmarkItem
    from analysis.benchmark_scorer import Measurement
    item = BenchmarkItem(
        id="i1", experiment_id="e1", metric_name="accuracy",
        expected_value=expected, tolerance=tol, tolerance_type=ttype,
    )
    return Measurement(item=item, measured=measured, difficulty=difficulty,
                       source=source, status=status)


def test_score_relative_pass_within_band():
    from analysis.benchmark_scorer import score
    out = score([_measurement(0.93, expected=0.90, tol=0.05)])  # band = ±0.045
    assert out[0].status == "pass" and out[0].passed is True


def test_score_relative_fail_outside_band():
    from analysis.benchmark_scorer import score
    out = score([_measurement(0.80, expected=0.90, tol=0.05)])
    assert out[0].status == "fail" and out[0].passed is False


def test_score_absolute_mode():
    from analysis.benchmark_scorer import score
    out = score([_measurement(0.62, expected=0.60, tol=0.05, ttype="absolute")])
    assert out[0].passed is True  # |0.62-0.60|=0.02 <= 0.05


def test_score_expected_zero_falls_back_to_absolute():
    from analysis.benchmark_scorer import score
    out = score([_measurement(0.03, expected=0.0, tol=0.05, ttype="relative")])
    assert out[0].passed is True  # |0.03-0|=0.03 <= 0.05 (absolute fallback)


def test_score_carries_unscorable_status():
    from analysis.benchmark_scorer import score
    for st in ("no_result", "missing_metric", "non_numeric"):
        out = score([_measurement(None, status=st)])
        assert out[0].status == st and out[0].passed is None and out[0].measured_value is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_benchmark_scorer.py -v`
Expected: 5 FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement dataclasses + `score()`**

Create `analysis/benchmark_scorer.py`:

```python
"""Benchmark scoring + run orchestration (SP2 of the Eval Harness).

Scores the golden set (BenchmarkItem) against each experiment's LATEST stored
ExperimentResult — no re-execution. See
docs/superpowers/specs/2026-05-24-benchmark-set-design.md.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import structlog

from core.models import BenchmarkItem

log = structlog.get_logger()

_UNKNOWN = {"", "unknown", None}
_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class Measurement:
    """An item plus the value pulled from its experiment's latest result."""
    item: BenchmarkItem
    measured: Optional[float]
    difficulty: str
    source: str
    status: str  # "ok" | "no_result" | "missing_metric" | "non_numeric"


@dataclass
class ItemOutcome:
    """A scored item; persisted as a BenchmarkItemResult and aggregated into a run."""
    item_id: str
    experiment_id: str
    metric_name: str
    expected_value: float
    tolerance: float
    tolerance_type: str
    measured_value: Optional[float]
    passed: Optional[bool]
    status: str  # "pass" | "fail" | "no_result" | "missing_metric" | "non_numeric"
    difficulty: str
    source: str


def _within_tolerance(measured: float, expected: float, tol: float, ttype: str) -> bool:
    if ttype == "relative" and expected != 0:
        return abs(measured - expected) <= tol * abs(expected)
    # absolute, or relative with expected == 0 (avoid a zero-width band)
    return abs(measured - expected) <= tol


def score(measurements: list[Measurement]) -> list[ItemOutcome]:
    """Pure: turn measurements into scored outcomes. Unscorable statuses pass through."""
    out: list[ItemOutcome] = []
    for m in measurements:
        it = m.item
        if m.status != "ok" or m.measured is None:
            out.append(ItemOutcome(
                item_id=it.id, experiment_id=it.experiment_id, metric_name=it.metric_name,
                expected_value=it.expected_value, tolerance=it.tolerance,
                tolerance_type=it.tolerance_type, measured_value=None, passed=None,
                status=m.status, difficulty=m.difficulty, source=m.source,
            ))
            continue
        passed = _within_tolerance(m.measured, it.expected_value, it.tolerance, it.tolerance_type)
        out.append(ItemOutcome(
            item_id=it.id, experiment_id=it.experiment_id, metric_name=it.metric_name,
            expected_value=it.expected_value, tolerance=it.tolerance,
            tolerance_type=it.tolerance_type, measured_value=m.measured,
            passed=passed, status=("pass" if passed else "fail"),
            difficulty=m.difficulty, source=m.source,
        ))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_benchmark_scorer.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/benchmark_scorer.py tests/test_benchmark_scorer.py
git commit -m "feat(bench): Measurement/ItemOutcome + pure score()"
```

---

### Task 5: `gather_measurements` — read latest results + dimensions

**Files:**
- Modify: `analysis/benchmark_scorer.py`
- Test: `tests/test_benchmark_scorer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark_scorer.py`:

```python
def _seed_item_and_result(engine, *, item_id="i1", experiment_id="e1", paper_id="p1",
                          metric_name="accuracy", metrics='{"accuracy": 0.91}',
                          source="arxiv", difficulty="easy", with_result=True):
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import (Paper, PaperAnalysis, Experiment, ExperimentResult,
                             BenchmarkItem)
    with Session(engine, expire_on_commit=False) as session:
        if not session.get(Paper, paper_id):
            session.add(Paper(id=paper_id, title="t", abstract="", source=source,
                              source_id=paper_id, url="x", published_date=date(2025, 1, 1)))
            session.add(PaperAnalysis(id=f"a_{paper_id}", paper_id=paper_id,
                                      reproducibility_difficulty=difficulty))
        session.add(Experiment(id=experiment_id, paper_id=paper_id, title="t",
                               hypothesis="h", execution_target="local", status="completed"))
        if with_result:
            session.add(ExperimentResult(id=f"result_{experiment_id}", experiment_id=experiment_id,
                                         exit_code=0, metrics=metrics, recorded_at=datetime.utcnow()))
        session.add(BenchmarkItem(id=item_id, experiment_id=experiment_id,
                                  metric_name=metric_name, expected_value=0.9,
                                  tolerance=0.05, tolerance_type="relative"))
        session.commit()


def test_gather_ok_with_scalar_metric(in_memory_engine):
    from knowledge.benchmark_store import get_items
    from analysis.benchmark_scorer import gather_measurements
    _seed_item_and_result(in_memory_engine, metrics='{"accuracy": 0.91}',
                          source="arxiv", difficulty="easy")
    m = gather_measurements(get_items())[0]
    assert m.status == "ok" and abs(m.measured - 0.91) < 1e-9
    assert m.difficulty == "easy" and m.source == "arxiv"


def test_gather_no_result(in_memory_engine):
    from knowledge.benchmark_store import get_items
    from analysis.benchmark_scorer import gather_measurements
    _seed_item_and_result(in_memory_engine, with_result=False)
    assert gather_measurements(get_items())[0].status == "no_result"


def test_gather_missing_metric_key(in_memory_engine):
    from knowledge.benchmark_store import get_items
    from analysis.benchmark_scorer import gather_measurements
    _seed_item_and_result(in_memory_engine, metric_name="f1", metrics='{"accuracy": 0.9}')
    assert gather_measurements(get_items())[0].status == "missing_metric"


def test_gather_non_numeric_metric(in_memory_engine):
    from knowledge.benchmark_store import get_items
    from analysis.benchmark_scorer import gather_measurements
    _seed_item_and_result(in_memory_engine, metrics='{"accuracy": "high"}')
    assert gather_measurements(get_items())[0].status == "non_numeric"


def test_gather_numeric_list_uses_mean(in_memory_engine):
    from knowledge.benchmark_store import get_items
    from analysis.benchmark_scorer import gather_measurements
    _seed_item_and_result(in_memory_engine, metrics='{"accuracy": [0.90, 0.92]}')
    m = gather_measurements(get_items())[0]
    assert m.status == "ok" and abs(m.measured - 0.91) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_benchmark_scorer.py -v -k gather`
Expected: 5 FAIL with `ImportError: cannot import name 'gather_measurements'`.

- [ ] **Step 3: Implement `gather_measurements`**

Append to `analysis/benchmark_scorer.py`:

```python
def _extract_numeric(metrics: dict, key: str) -> tuple[Optional[float], str]:
    """Return (value, status). status ∈ {ok, missing_metric, non_numeric}."""
    if key not in metrics:
        return None, "missing_metric"
    v = metrics[key]
    if isinstance(v, bool):  # bool is a subclass of int — reject explicitly
        return None, "non_numeric"
    if isinstance(v, (int, float)):
        return float(v), "ok"
    if isinstance(v, list):
        nums = [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if nums:
            return sum(nums) / len(nums), "ok"
    return None, "non_numeric"


def gather_measurements(items: list[BenchmarkItem]) -> list[Measurement]:
    """For each item, read its experiment's latest ExperimentResult and pull metric_name.

    Joins Experiment/Paper/PaperAnalysis for the difficulty/source dimensions, selecting only
    the columns needed (never loads the corrupt paper_analysis.analyzed_at — see SP1 hardening).
    """
    from sqlmodel import Session, select
    from core.database import get_engine
    from core.models import Experiment, ExperimentResult, Paper, PaperAnalysis

    if not items:
        return []

    exp_ids = list({it.experiment_id for it in items})
    with Session(get_engine()) as session:
        experiments = {e.id: e for e in session.exec(
            select(Experiment).where(Experiment.id.in_(exp_ids))
        ).all()}
        paper_ids = list({e.paper_id for e in experiments.values()})
        results = {r.experiment_id: r for r in session.exec(
            select(ExperimentResult).where(ExperimentResult.experiment_id.in_(exp_ids))
        ).all()}
        sources = {pid: src for pid, src in session.exec(
            select(Paper.id, Paper.source).where(Paper.id.in_(paper_ids))
        ).all()} if paper_ids else {}
        difficulties = {pid: diff for pid, diff in session.exec(
            select(PaperAnalysis.paper_id, PaperAnalysis.reproducibility_difficulty)
            .where(PaperAnalysis.paper_id.in_(paper_ids))
        ).all()} if paper_ids else {}

    out: list[Measurement] = []
    for it in items:
        exp = experiments.get(it.experiment_id)
        paper_id = exp.paper_id if exp else None
        raw_diff = difficulties.get(paper_id, "unknown")
        difficulty = raw_diff if raw_diff in _DIFFICULTIES else "unknown"
        source = sources.get(paper_id, "unknown")

        result = results.get(it.experiment_id)
        if result is None or not result.metrics:
            out.append(Measurement(it, None, difficulty, source, "no_result"))
            continue
        try:
            metrics = json.loads(result.metrics)
        except (json.JSONDecodeError, TypeError):
            out.append(Measurement(it, None, difficulty, source, "non_numeric"))
            continue
        value, status = _extract_numeric(metrics, it.metric_name)
        out.append(Measurement(it, value, difficulty, source, status))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_benchmark_scorer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/benchmark_scorer.py tests/test_benchmark_scorer.py
git commit -m "feat(bench): gather_measurements reads latest result + dimensions"
```

---

### Task 6: `build_metric_points` + `record_benchmark_run`

**Files:**
- Modify: `analysis/benchmark_scorer.py`
- Test: `tests/test_benchmark_scorer.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_benchmark_scorer.py`:

```python
def test_build_metric_points_overall_and_dimensions():
    from analysis.benchmark_scorer import ItemOutcome, build_metric_points

    def _o(passed, status, difficulty="easy", source="arxiv"):
        return ItemOutcome(item_id="x", experiment_id="e", metric_name="accuracy",
                           expected_value=0.9, tolerance=0.05, tolerance_type="relative",
                           measured_value=0.9, passed=passed, status=status,
                           difficulty=difficulty, source=source)

    outcomes = [_o(True, "pass", "easy", "arxiv"),
                _o(False, "fail", "easy", "arxiv"),
                _o(None, "no_result", "hard", "substack")]
    points = build_metric_points(outcomes)
    overall = next(p for p in points if p.dimension == "overall")
    assert overall.metric == "benchmark_accuracy"
    assert overall.numerator == 1 and overall.denominator == 2  # unscorable excluded
    easy = next(p for p in points if p.dimension == "difficulty:easy")
    assert easy.numerator == 1 and easy.denominator == 2
    # no scorable hard items -> no difficulty:hard bucket
    assert not any(p.dimension == "difficulty:hard" for p in points)


def test_build_metric_points_zero_scorable_value_none():
    from analysis.benchmark_scorer import ItemOutcome, build_metric_points
    o = ItemOutcome(item_id="x", experiment_id="e", metric_name="accuracy",
                    expected_value=0.9, tolerance=0.05, tolerance_type="relative",
                    measured_value=None, passed=None, status="no_result",
                    difficulty="easy", source="arxiv")
    overall = next(p for p in build_metric_points([o]) if p.dimension == "overall")
    assert overall.value is None and overall.denominator == 0


def test_record_benchmark_run_persists_and_writes_eval_metric(in_memory_engine):
    from analysis.benchmark_scorer import record_benchmark_run
    from knowledge.benchmark_store import get_runs, get_item_results
    from knowledge.eval_metric_store import get_latest

    _seed_item_and_result(in_memory_engine, item_id="i1", experiment_id="e1",
                          metrics='{"accuracy": 0.91}')  # within ±5% of 0.9 -> pass
    run = record_benchmark_run(trigger="manual")
    assert run.n_pass == 1 and run.n_fail == 0 and run.accuracy == 1.0
    assert len(get_runs()) == 1
    assert len(get_item_results(run.id)) == 1
    em = get_latest("benchmark_accuracy", "overall")
    assert em is not None and em.numerator == 1 and em.denominator == 1


def test_record_benchmark_run_empty_golden_set(in_memory_engine):
    from analysis.benchmark_scorer import record_benchmark_run
    run = record_benchmark_run(trigger="manual")
    assert run.n_items == 0 and run.accuracy is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_benchmark_scorer.py -v -k "build_metric_points or record_benchmark_run"`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_metric_points` + `record_benchmark_run`**

Append to `analysis/benchmark_scorer.py`:

```python
def _bucket_point(outcomes: list[ItemOutcome], dimension: str):
    from knowledge.eval_metric_store import MetricPoint
    scorable = [o for o in outcomes if o.status in ("pass", "fail")]
    n = len(scorable)
    n_pass = sum(1 for o in scorable if o.status == "pass")
    n_fail = n - n_pass
    value: Optional[float] = (n_pass / n) if n else None
    context = json.dumps({"pass": n_pass, "fail": n_fail,
                          "unscorable": len(outcomes) - n})
    return MetricPoint(metric="benchmark_accuracy", dimension=dimension,
                       value=value, numerator=n_pass, denominator=n, context=context)


def build_metric_points(outcomes: list[ItemOutcome]) -> list:
    """Pure: aggregate outcomes into benchmark_accuracy MetricPoints (overall + dimensions)."""
    points = [_bucket_point(outcomes, "overall")]
    for dim_name, getter in (("difficulty", lambda o: o.difficulty),
                             ("source", lambda o: o.source)):
        groups: dict[str, list[ItemOutcome]] = {}
        for o in outcomes:
            key = getter(o)
            if key in _UNKNOWN:
                continue
            groups.setdefault(key, []).append(o)
        for key, grp in groups.items():
            # Only emit a bucket if it has at least one scorable item.
            if any(o.status in ("pass", "fail") for o in grp):
                points.append(_bucket_point(grp, f"{dim_name}:{key}"))
    return points


def record_benchmark_run(cycle_id: Optional[str] = None, trigger: str = "manual"):
    """Score the active golden set against latest results; persist run + item-results;
    write aggregate benchmark_accuracy to the SP1 EvalMetric store. Returns the BenchmarkRun."""
    from core.models import BenchmarkRun, BenchmarkItemResult
    from knowledge.benchmark_store import get_items, save_run
    from knowledge.eval_metric_store import save_metrics

    items = get_items(active_only=True)
    outcomes = score(gather_measurements(items))

    n_pass = sum(1 for o in outcomes if o.status == "pass")
    n_fail = sum(1 for o in outcomes if o.status == "fail")
    n_unscorable = len(outcomes) - n_pass - n_fail
    n_scorable = n_pass + n_fail
    accuracy = (n_pass / n_scorable) if n_scorable else None

    run = BenchmarkRun(
        id=str(uuid.uuid4()), recorded_at=datetime.utcnow(), cycle_id=cycle_id,
        trigger=trigger, n_items=len(items), n_pass=n_pass, n_fail=n_fail,
        n_unscorable=n_unscorable, accuracy=accuracy,
    )
    item_results = [
        BenchmarkItemResult(
            id=str(uuid.uuid4()), run_id=run.id, item_id=o.item_id,
            experiment_id=o.experiment_id, metric_name=o.metric_name,
            expected_value=o.expected_value, tolerance=o.tolerance,
            tolerance_type=o.tolerance_type, measured_value=o.measured_value,
            passed=o.passed, status=o.status,
        )
        for o in outcomes
    ]
    save_run(run, item_results)

    points = build_metric_points(outcomes)
    snapshot_cycle = cycle_id or f"benchmark-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    save_metrics(points, cycle_id=snapshot_cycle)

    log.info("benchmark_scorer.run_recorded", run_id=run.id, n_items=len(items),
             n_pass=n_pass, n_fail=n_fail, n_unscorable=n_unscorable, accuracy=accuracy)
    return run
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_benchmark_scorer.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add analysis/benchmark_scorer.py tests/test_benchmark_scorer.py
git commit -m "feat(bench): build_metric_points + record_benchmark_run (writes EvalMetric)"
```

---

## Chunk 3: Integration & surfaces

### Task 7: Auto-score hook in `analysis_pipeline.run`

**Files:**
- Modify: `analysis/analysis_pipeline.py` (after the SP1 snapshot block, ~line 75)
- Test: `tests/test_benchmark_scorer.py`

- [ ] **Step 1: Write the failing integration tests**

Append to `tests/test_benchmark_scorer.py`:

```python
def test_analysis_pipeline_scores_benchmark_when_golden_set_nonempty(in_memory_engine, monkeypatch):
    from analysis import analysis_pipeline
    from core.models import RunState
    from datetime import datetime
    from knowledge.benchmark_store import get_runs

    monkeypatch.setattr(analysis_pipeline, "_generate_conclusion", lambda *a, **k: "")
    _seed_item_and_result(in_memory_engine, item_id="i1", experiment_id="e1",
                          metrics='{"accuracy": 0.91}')
    state = RunState(cycle_id="cyc_bench", started_at=datetime.utcnow(),
                     experiment_ids_this_cycle=["e1"])
    analysis_pipeline.run(state)

    runs = get_runs()
    assert len(runs) == 1 and runs[0].cycle_id == "cyc_bench" and runs[0].trigger == "cycle"


def test_analysis_pipeline_skips_benchmark_when_golden_set_empty(in_memory_engine, monkeypatch):
    from analysis import analysis_pipeline
    from core.models import RunState
    from datetime import datetime
    from knowledge.benchmark_store import get_runs

    monkeypatch.setattr(analysis_pipeline, "_generate_conclusion", lambda *a, **k: "")
    analysis_pipeline.run(RunState(cycle_id="c", started_at=datetime.utcnow(),
                                   experiment_ids_this_cycle=[]))
    assert get_runs() == []  # no golden set -> no run recorded


def test_analysis_pipeline_swallows_benchmark_failure(in_memory_engine, monkeypatch):
    from analysis import analysis_pipeline
    from core.models import RunState
    from datetime import datetime

    monkeypatch.setattr(analysis_pipeline, "_generate_conclusion", lambda *a, **k: "")
    _seed_item_and_result(in_memory_engine, item_id="i1", experiment_id="e1")

    def boom(*a, **k):
        raise RuntimeError("kaboom")

    monkeypatch.setattr("analysis.benchmark_scorer.record_benchmark_run", boom)
    # Must not raise.
    analysis_pipeline.run(RunState(cycle_id="c", started_at=datetime.utcnow(),
                                   experiment_ids_this_cycle=["e1"]))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run python -m pytest tests/test_benchmark_scorer.py -v -k analysis_pipeline`
Expected: `..._scores_benchmark...` FAILS (no run recorded — not wired). The skip/swallow tests may pass accidentally.

- [ ] **Step 3: Wire the hook**

In `analysis/analysis_pipeline.py`, insert **after** the SP1 snapshot try/except block and **before** `log.info("analysis_pipeline.done")`:

```python
    # Eval-harness SP2: score the golden set against latest results (no compute).
    # Skipped when the golden set is empty; failure never aborts the cycle.
    try:
        from knowledge.benchmark_store import get_items
        if get_items(active_only=True):
            from analysis import benchmark_scorer
            benchmark_scorer.record_benchmark_run(cycle_id=state.cycle_id, trigger="cycle")
    except Exception as e:
        log.error("analysis_pipeline.benchmark_error", error=str(e))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_benchmark_scorer.py -v -k analysis_pipeline`
Expected: all PASS.

Run the full suite: `uv run python -m pytest tests/ -q --tb=short`
Expected: all PASS (78 prior + new).

- [ ] **Step 5: Commit**

```bash
git add analysis/analysis_pipeline.py tests/test_benchmark_scorer.py
git commit -m "feat(bench): auto-score golden set at end of analysis_pipeline.run"
```

---

### Task 8: Streamlit Benchmark page

**Files:**
- Create: `ui/views/benchmark.py`
- Modify: `ui/app.py` (`PAGE_LABELS` + route)

UI verification is manual (no Streamlit harness in the suite — mirrors SP1's dashboard-tile task).

- [ ] **Step 1: Create the page**

Create `ui/views/benchmark.py`:

```python
"""Benchmark page — curate the golden set, run scoring, view accuracy trend + per-item results."""
from __future__ import annotations

import json
import uuid

import streamlit as st


@st.cache_data(ttl=8)
def _completed_experiments() -> list[dict]:
    """Completed experiments with their paper title and available metric keys."""
    from knowledge.experiment_store import get_experiments_by_status, get_result
    from knowledge.paper_store import get_paper
    rows = []
    for exp in get_experiments_by_status("completed"):
        result = get_result(exp.id)
        metric_keys, measured = [], {}
        if result and result.metrics:
            try:
                m = json.loads(result.metrics)
                metric_keys = [k for k, v in m.items() if isinstance(v, (int, float, list))]
                measured = m
            except (json.JSONDecodeError, TypeError):
                pass
        paper = get_paper(exp.paper_id)
        rows.append({
            "experiment_id": exp.id, "title": exp.title,
            "paper_title": paper.title if paper else exp.paper_id,
            "metric_keys": metric_keys, "measured": measured,
        })
    return rows


def _add_item_form(experiments: list[dict]) -> None:
    st.subheader("Add a golden-set item")
    if not experiments:
        st.info("No completed experiments yet — run some experiments first.")
        return
    labels = {f"{e['title']} — {e['paper_title']}": e for e in experiments}
    choice = st.selectbox("Experiment", list(labels.keys()), key="bench_exp")
    exp = labels[choice]
    keys = exp["metric_keys"] or ["(no numeric metrics found)"]
    with st.form("bench_add", clear_on_submit=True):
        metric_name = st.selectbox("Metric key", keys)
        if metric_name in exp["measured"]:
            st.caption(f"Latest measured value: {exp['measured'][metric_name]}")
        c1, c2 = st.columns(2)
        expected = c1.number_input("Expected value", value=0.0, format="%.6f")
        tolerance = c2.number_input("Tolerance", value=0.05, format="%.6f", min_value=0.0)
        ttype = st.radio("Tolerance type", ["relative", "absolute"], horizontal=True)
        unit = st.text_input("Unit (optional)")
        note = st.text_area("Note / provenance", placeholder="e.g. paper Table 2, top-1 accuracy")
        if st.form_submit_button("Add to golden set"):
            from core.models import BenchmarkItem
            from knowledge.benchmark_store import save_item
            save_item(BenchmarkItem(
                id=str(uuid.uuid4()), experiment_id=exp["experiment_id"],
                metric_name=metric_name, expected_value=float(expected),
                tolerance=float(tolerance), tolerance_type=ttype,
                unit=unit or None, note=note,
            ))
            st.success("Added.")
            st.cache_data.clear()
            st.rerun()


def _golden_set_table() -> None:
    import pandas as pd
    from knowledge.benchmark_store import get_items, deactivate_item
    items = get_items(active_only=True)
    st.subheader(f"Golden set ({len(items)} active)")
    if not items:
        st.caption("Empty — add items above.")
        return
    st.dataframe(pd.DataFrame([{
        "metric": i.metric_name, "expected": i.expected_value,
        "tol": f"{i.tolerance}{'%' if i.tolerance_type == 'relative' else ''}",
        "type": i.tolerance_type, "unit": i.unit or "", "note": i.note,
        "experiment_id": i.experiment_id,
    } for i in items]), use_container_width=True, hide_index=True)
    to_remove = st.selectbox("Deactivate item", ["—"] + [i.id for i in items], key="bench_rm")
    if to_remove != "—" and st.button("Deactivate"):
        deactivate_item(to_remove)
        st.cache_data.clear()
        st.rerun()


def _run_and_view() -> None:
    import pandas as pd
    from knowledge.benchmark_store import get_latest_run, get_item_results
    from knowledge.eval_metric_store import get_trend

    if st.button("▶ Run benchmark"):
        from analysis import benchmark_scorer
        with st.status("Scoring golden set…", expanded=False):
            run = benchmark_scorer.record_benchmark_run(trigger="manual")
        acc = "—" if run.accuracy is None else f"{run.accuracy * 100:.1f}%"
        st.success(f"Accuracy: {acc}  ·  pass {run.n_pass} / fail {run.n_fail} / unscorable {run.n_unscorable}")
        st.cache_data.clear()

    trend = get_trend("benchmark_accuracy", "overall", limit=12)
    pts = [{"cycle": r.cycle_id, "accuracy_pct": r.value * 100} for r in trend if r.value is not None]
    if pts:
        st.line_chart(pd.DataFrame(pts), x="cycle", y="accuracy_pct", height=200)

    latest = get_latest_run()
    if latest:
        st.caption(f"Latest run: {latest.recorded_at:%Y-%m-%d %H:%M} ({latest.trigger})")
        results = get_item_results(latest.id)
        if results:
            st.dataframe(pd.DataFrame([{
                "metric": r.metric_name, "expected": r.expected_value,
                "measured": r.measured_value, "status": r.status,
            } for r in results]), use_container_width=True, hide_index=True)


def render() -> None:
    st.title("Benchmark")
    st.caption("Curate a ground-truth golden set and track how accurately the pipeline reproduces known results.")
    experiments = _completed_experiments()
    _add_item_form(experiments)
    st.divider()
    _golden_set_table()
    st.divider()
    _run_and_view()
```

- [ ] **Step 2: Register the page in `ui/app.py`**

Add `"◎  Benchmark"` to `PAGE_LABELS` immediately after `"✓  Review Queue"`:

```python
PAGE_LABELS = [
    "⊞  Dashboard",
    "＋  Add Paper",
    "✓  Review Queue",
    "◎  Benchmark",
    "◫  Papers",
    "⚗  Experiments",
    "◉  Living Review",
    "≡  Reports",
    "⚙  Settings",
]
```

And add a route, immediately after the `Review Queue` branch:

```python
elif page == "Benchmark":
    from ui.views.benchmark import render
    render()
```

- [ ] **Step 3: Smoke-check the module compiles**

Run: `uv run python -m py_compile ui/views/benchmark.py`
Expected: exit 0 (no output).

- [ ] **Step 4: Manual verification**

Run: `uv run python -m streamlit run ui/app.py --server.headless true --browser.gatherUsageStats false`
Open http://localhost:8501 → Benchmark page. Confirm:
- The page renders; the experiment dropdown lists completed experiments and the metric-key dropdown populates from the selected experiment's metrics.
- Adding an item shows it in the golden-set table.
- "Run benchmark" produces an accuracy summary + per-item table; the trend chart appears after at least one run with a scorable item.
- Deactivating an item removes it from the active table.

- [ ] **Step 5: Commit**

```bash
git add ui/views/benchmark.py ui/app.py
git commit -m "feat(bench): Streamlit Benchmark page (curate/run/view) + nav route"
```

---

## Final verification

- [ ] **Run the full test suite**

```bash
uv run python -m pytest tests/ -q --tb=short
```

Expected: all prior + new tests PASS, exit 0.

- [ ] **Smoke-test imports**

```bash
uv run python -c "
from core.models import BenchmarkItem, BenchmarkRun, BenchmarkItemResult
from knowledge.benchmark_store import save_item, get_items, get_item, deactivate_item, save_run, get_runs, get_latest_run, get_item_results
from analysis.benchmark_scorer import Measurement, ItemOutcome, gather_measurements, score, build_metric_points, record_benchmark_run
print('All SP2 imports OK')
"
```

Expected: `All SP2 imports OK`.

- [ ] **Confirm DB migration is automatic**

```bash
uv run python -c "from core.database import init_db; init_db(); print('OK')"
```

Expected: `OK` — the three benchmark tables are created on the live DB without any ALTER.

- [ ] **Update CLAUDE.md** (project doc; note: CLAUDE.md is gitignored in this repo — edit on disk, do not commit it)

Append to the "Key Files Map" table:

```markdown
| `analysis/benchmark_scorer.py` | Benchmark scoring (SP2 of Eval Harness): gather latest results, pure score, record_benchmark_run → BenchmarkRun/ItemResult + benchmark_accuracy EvalMetric |
| `knowledge/benchmark_store.py` | CRUD for BenchmarkItem (golden set) + BenchmarkRun + BenchmarkItemResult |
| `ui/views/benchmark.py` | Streamlit Benchmark page — curate golden set, run scoring, view accuracy trend + per-item results |
```

Add to the Architecture pipeline-stages note: "`analysis_pipeline.run` also calls `benchmark_scorer.record_benchmark_run(state.cycle_id, trigger='cycle')` at end-of-stage when the golden set is non-empty (try/except, SP2)."

- [ ] **Finish the branch** — use the `finishing-a-development-branch` skill (or merge `--no-ff` into `main` and delete the branch), matching the SP1 workflow.

---

## Out of scope (do NOT add — later sub-projects)

- Re-executing golden experiments during a benchmark run (scoring reads latest stored results only).
- LLM-as-judge for analysis quality (SP3).
- Calibrating scores against the golden set (SP4).
- A CLI `benchmark` command (UI-only in SP2).
- Per-item regression *alerting* / diffing beyond the per-run table (YAGNI for SP2).
