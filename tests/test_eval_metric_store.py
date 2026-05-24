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
