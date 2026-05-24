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


def _seed_item_and_result(engine, *, item_id="i1", experiment_id="e1", paper_id="p1",
                          metric_name="accuracy", metrics='{"accuracy": 0.91}',
                          source="arxiv", difficulty="easy", with_result=True):
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
