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
