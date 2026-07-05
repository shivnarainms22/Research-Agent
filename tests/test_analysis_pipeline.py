"""Tests for analysis/analysis_pipeline.analyze_result (shared by pipeline and UI)."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch

from core.models import Experiment, ExperimentResult


def _make_pair() -> tuple[Experiment, ExperimentResult]:
    exp = Experiment(
        id="e1", paper_id="p1", title="Exp", hypothesis="h",
        status="completed", created_at=datetime.utcnow(),
    )
    result = ExperimentResult(
        id="result_e1", experiment_id="e1", exit_code=0,
        metrics=json.dumps({"accuracy": 0.9}),
    )
    return exp, result


def test_analyze_result_attaches_summaries_and_saves():
    from analysis import analysis_pipeline

    exp, result = _make_pair()
    with (
        patch("analysis.analysis_pipeline.statistical_analyzer.analyze",
              return_value={"accuracy": {"mean": 0.9}}),
        patch("analysis.analysis_pipeline.baseline_comparator.compare",
              return_value={"overall": "fully_reproduced"}),
        patch("analysis.analysis_pipeline._generate_conclusion", return_value="ok"),
        patch("analysis.analysis_pipeline.save_result") as mock_save,
    ):
        comparison = analysis_pipeline.analyze_result(exp, result)

    assert comparison == {"overall": "fully_reproduced"}
    assert json.loads(result.statistical_summary) == {"accuracy": {"mean": 0.9}}
    assert json.loads(result.baseline_comparison) == {"overall": "fully_reproduced"}
    assert result.conclusion == "ok"
    mock_save.assert_called_once_with(result)


def test_analyze_result_conclusion_failure_does_not_block_save():
    from analysis import analysis_pipeline

    exp, result = _make_pair()
    with (
        patch("analysis.analysis_pipeline.statistical_analyzer.analyze",
              return_value={"accuracy": {"mean": 0.9}}),
        patch("analysis.analysis_pipeline.baseline_comparator.compare",
              return_value={"overall": "not_reproduced"}),
        patch("analysis.analysis_pipeline._generate_conclusion",
              side_effect=RuntimeError("api down")),
        patch("analysis.analysis_pipeline.save_result") as mock_save,
    ):
        analysis_pipeline.analyze_result(exp, result)

    assert result.conclusion is None
    mock_save.assert_called_once_with(result)
