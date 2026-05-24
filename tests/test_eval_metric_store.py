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
