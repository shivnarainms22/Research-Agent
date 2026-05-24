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
