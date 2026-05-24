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
