"""Tests for analysis/baseline_comparator.py"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from core.models import ExperimentResult, PaperAnalysis
from analysis.baseline_comparator import compare


def _make_result(metrics: dict) -> ExperimentResult:
    return ExperimentResult(
        id="result_test",
        experiment_id="exp_test",
        metrics=json.dumps(metrics),
    )


def _make_analysis(reproducible_experiments: list) -> PaperAnalysis:
    return PaperAnalysis(
        id="analysis_test",
        paper_id="paper_test",
        reproducible_experiments=json.dumps(reproducible_experiments),
    )


def test_no_analysis_returns_no_analysis():
    result = _make_result({"accuracy": 0.9})
    with patch("analysis.baseline_comparator.get_analysis", return_value=None):
        out = compare(result, "paper_test")
    assert out["status"] == "no_analysis"


def test_no_baselines_in_analysis():
    analysis = _make_analysis([])
    result = _make_result({"accuracy": 0.9})
    with patch("analysis.baseline_comparator.get_analysis", return_value=analysis):
        out = compare(result, "paper_test")
    assert out["status"] == "no_baselines"


def test_no_metrics_in_result():
    analysis = _make_analysis([{
        "title": "Test", "description": "...", "expected_metric": "accuracy",
        "baseline_claimed": {"metric_name": "accuracy", "value": 0.9, "unit": "%"},
    }])
    result = _make_result({})
    with patch("analysis.baseline_comparator.get_analysis", return_value=analysis):
        out = compare(result, "paper_test")
    assert out["status"] == "no_metrics"


def test_reproduced_within_5_percent():
    analysis = _make_analysis([{
        "title": "Test", "description": "...", "expected_metric": "accuracy",
        "baseline_claimed": {"metric_name": "accuracy", "value": 0.9, "unit": "ratio"},
    }])
    result = _make_result({"accuracy": 0.91})
    with patch("analysis.baseline_comparator.get_analysis", return_value=analysis):
        out = compare(result, "paper_test")
    assert out["overall"] == "fully_reproduced"
    assert out["comparisons"][0]["status"] == "reproduced"


def test_partial_within_15_percent():
    analysis = _make_analysis([{
        "title": "Test", "description": "...", "expected_metric": "accuracy",
        "baseline_claimed": {"metric_name": "accuracy", "value": 0.9, "unit": "ratio"},
    }])
    result = _make_result({"accuracy": 0.80})
    with patch("analysis.baseline_comparator.get_analysis", return_value=analysis):
        out = compare(result, "paper_test")
    assert out["comparisons"][0]["status"] == "partial"


def test_failed_beyond_15_percent():
    analysis = _make_analysis([{
        "title": "Test", "description": "...", "expected_metric": "accuracy",
        "baseline_claimed": {"metric_name": "accuracy", "value": 0.9, "unit": "ratio"},
    }])
    result = _make_result({"accuracy": 0.5})
    with patch("analysis.baseline_comparator.get_analysis", return_value=analysis):
        out = compare(result, "paper_test")
    assert out["comparisons"][0]["status"] == "failed"


def test_string_baseline_legacy_format():
    """Old string format like 'accuracy 92.3%' should parse the first numeric value."""
    analysis = _make_analysis([{
        "title": "Test", "description": "...", "expected_metric": "accuracy",
        "baseline_claimed": "accuracy 92.0",
    }])
    # 92.0 claimed, 93.0 actual => ~1.09% diff => reproduced
    result = _make_result({"accuracy": 93.0})
    with patch("analysis.baseline_comparator.get_analysis", return_value=analysis):
        out = compare(result, "paper_test")
    assert out["comparisons"][0]["status"] == "reproduced"
