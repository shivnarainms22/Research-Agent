"""Test the one-time baseline re-score + repro-metric rebuild."""
from __future__ import annotations

import json
from datetime import date, datetime

from sqlmodel import Session, select

from core.models import (
    EvalMetric, Experiment, ExperimentResult, Paper, PaperAnalysis,
)


def _seed(session):
    session.add(Paper(
        id="p1", title="Paper", abstract="a", source="arxiv", source_id="p1",
        url="u", published_date=date(2024, 1, 1),
    ))
    session.add(PaperAnalysis(
        id="a1", paper_id="p1", reproducibility_difficulty="easy",
        reproducible_experiments=json.dumps([{
            "title": "Accuracy test", "expected_metric": "accuracy",
            "baseline_claimed": {"metric_name": "accuracy", "value": 0.9},
        }]),
    ))
    session.add(Experiment(
        id="e1", paper_id="p1", title="Accuracy test", hypothesis="h",
        status="completed", created_at=datetime.utcnow(),
    ))
    # metrics reproduce the claim, but the stored comparison is the broken verdict
    session.add(ExperimentResult(
        id="result_e1", experiment_id="e1", exit_code=0,
        metrics=json.dumps({"accuracy": 0.91}),
        baseline_comparison=json.dumps({"overall": "not_reproduced", "comparisons": []}),
    ))
    # a stale reproduction_rate row that must be replaced
    session.add(EvalMetric(
        id="m1", metric="reproduction_rate", dimension="overall", value=0.0,
        numerator=0, denominator=1, cycle_id="backfill-2024-W01",
    ))
    session.commit()


def test_rescore_fixes_verdict_and_rebuilds_metric(in_memory_engine):
    from analysis.rescore_baselines import rescore_all

    with Session(in_memory_engine) as session:
        _seed(session)

    summary = rescore_all()
    assert summary["rescored"] == 1

    with Session(in_memory_engine) as session:
        result = session.get(ExperimentResult, "result_e1")
        assert json.loads(result.baseline_comparison)["overall"] == "fully_reproduced"

        # stale 0% row gone; a fresh reproduction_rate point reflects the fix
        repro = list(session.exec(
            select(EvalMetric).where(EvalMetric.metric == "reproduction_rate")
        ).all())
        assert repro, "expected a rebuilt reproduction_rate point"
        overall = [r for r in repro if r.dimension == "overall"]
        assert overall and overall[0].value == 1.0
