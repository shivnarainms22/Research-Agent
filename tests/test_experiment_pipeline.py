"""Integration tests for experiments/experiment_pipeline.py with mocked runner."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from core.models import Experiment, ExperimentResult, RunState


def _make_experiment(exp_id: str, target: str = "local") -> Experiment:
    return Experiment(
        id=exp_id,
        paper_id="paper_001",
        title=f"Test Experiment {exp_id}",
        hypothesis="Testing hypothesis",
        generated_code="import json\nwith open('/workspace/results/metrics.json','w') as f:\n    json.dump({'accuracy': 0.9}, f)",
        execution_target=target,
        status="pending",
        created_at=datetime.utcnow(),
        retry_count=0,
    )


def _make_result(exp_id: str, exit_code: int = 0, metrics: dict | None = None) -> ExperimentResult:
    return ExperimentResult(
        id=f"result_{exp_id}",
        experiment_id=exp_id,
        stdout="Done",
        exit_code=exit_code,
        metrics=json.dumps(metrics or {"accuracy": 0.9}),
        artifacts="[]",
        runtime_seconds=1.0,
        recorded_at=datetime.utcnow(),
    )


def test_experiment_pipeline_runs_pending(in_memory_engine):
    """Pending experiments should be picked up and their status updated."""
    from experiments import experiment_pipeline

    exp = _make_experiment("exp_001")
    result = _make_result("exp_001")
    state = RunState(
        cycle_id="test_cycle",
        started_at=datetime.utcnow(),
    )

    # Patch at the experiment_pipeline module's imported names
    with (
        patch("experiments.experiment_pipeline.get_experiments_by_status", return_value=[exp]),
        patch("experiments.experiment_pipeline.code_validator.validate_with_retry", return_value=(exp.generated_code, True)),
        patch("experiments.experiment_pipeline.router.decide_target", return_value="local"),
        patch("experiments.experiment_pipeline.local_runner.run", return_value=result),
        patch("experiments.experiment_pipeline.get_result", return_value=None),
        patch("experiments.experiment_pipeline.delete_result"),
        patch("experiments.experiment_pipeline.save_result"),
        patch("experiments.experiment_pipeline.update_experiment_status") as mock_status,
        patch("experiments.experiment_pipeline.increment_retry"),
        patch("experiments.experiment_pipeline.save_state"),
    ):
        experiment_pipeline.run(state)

    # Should have been marked running then completed
    calls = [str(c) for c in mock_status.call_args_list]
    assert any("running" in c for c in calls)
    assert any("completed" in c for c in calls)


def test_experiment_pipeline_handles_runner_failure(in_memory_engine):
    """When the runner returns a non-zero exit code, experiment should be marked failed."""
    from experiments import experiment_pipeline

    exp = _make_experiment("exp_002")
    failed_result = _make_result("exp_002", exit_code=1, metrics={})
    failed_result.metrics = "{}"
    state = RunState(
        cycle_id="test_cycle_fail",
        started_at=datetime.utcnow(),
    )

    with (
        patch("experiments.experiment_pipeline.get_experiments_by_status", return_value=[exp]),
        patch("experiments.experiment_pipeline.code_validator.validate_with_retry", return_value=(exp.generated_code, True)),
        patch("experiments.experiment_pipeline.router.decide_target", return_value="local"),
        patch("experiments.experiment_pipeline.local_runner.run", return_value=failed_result),
        patch("experiments.experiment_pipeline.get_result", return_value=None),
        patch("experiments.experiment_pipeline.delete_result"),
        patch("experiments.experiment_pipeline.save_result"),
        patch("experiments.experiment_pipeline.update_experiment_status") as mock_status,
        patch("experiments.experiment_pipeline.increment_retry"),
        patch("experiments.experiment_pipeline.save_state"),
    ):
        experiment_pipeline.run(state)

    calls = [str(c) for c in mock_status.call_args_list]
    assert any("failed" in c for c in calls)
