"""Tests for experiments/result_collector.py"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.result_collector import collect, parse_metrics_from_stdout


def test_collect_returns_empty_for_missing_workspace(tmp_path):
    with patch("experiments.result_collector.settings") as mock_settings:
        mock_settings.experiments_dir = tmp_path
        result = collect("nonexistent_exp_id")
    assert result == {}


def test_collect_reads_metrics_json(tmp_path):
    """Should read metrics.json from the results subdirectory."""
    exp_id = "test_exp_001"
    workspace = tmp_path / exp_id / "results"
    workspace.mkdir(parents=True)
    metrics = {"accuracy": 0.95, "loss": 0.05}
    (workspace / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    with patch("experiments.result_collector.settings") as mock_settings:
        mock_settings.experiments_dir = tmp_path
        result = collect(exp_id)

    assert result["metrics"] == metrics


def test_collect_lists_artifacts(tmp_path):
    """Should enumerate artifact files."""
    exp_id = "test_exp_002"
    workspace = tmp_path / exp_id / "results"
    workspace.mkdir(parents=True)
    (workspace / "metrics.json").write_text("{}", encoding="utf-8")
    (workspace / "plot.png").write_bytes(b"fake png")

    with patch("experiments.result_collector.settings") as mock_settings:
        mock_settings.experiments_dir = tmp_path
        result = collect(exp_id)

    assert len(result["artifacts"]) == 2


def test_parse_metrics_from_stdout_basic():
    stdout = "accuracy: 0.95\nloss = 0.05\nepochs: 10"
    metrics = parse_metrics_from_stdout(stdout)
    assert metrics["accuracy"] == pytest.approx(0.95)
    assert metrics["loss"] == pytest.approx(0.05)
    assert metrics["epochs"] == pytest.approx(10.0)


def test_parse_metrics_from_stdout_empty():
    metrics = parse_metrics_from_stdout("No metrics here, just prose.")
    assert isinstance(metrics, dict)
