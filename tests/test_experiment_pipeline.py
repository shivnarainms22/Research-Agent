"""Integration tests for experiments/experiment_pipeline.py with mocked runner."""
from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import patch


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
    ):
        experiment_pipeline.run(state)

    # Should have been marked running then completed
    calls = [str(c) for c in mock_status.call_args_list]
    assert any("running" in c for c in calls)
    assert any("completed" in c for c in calls)


def test_failed_run_repairs_and_requeues():
    """A failed run with retries left gets repaired code and goes back to pending."""
    from experiments import experiment_pipeline

    exp = _make_experiment("exp_002")
    failed_result = _make_result("exp_002", exit_code=1)
    failed_result.metrics = "{}"
    failed_result.stdout = "Traceback: ImportError: no module named foo"

    state = RunState(cycle_id="t", started_at=datetime.utcnow())
    with (
        patch("experiments.experiment_pipeline.get_experiments_by_status", return_value=[exp]),
        patch("experiments.experiment_pipeline.code_validator.validate_with_retry", return_value=(exp.generated_code, True)),
        patch("experiments.experiment_pipeline.router.decide_target", return_value="local"),
        patch("experiments.experiment_pipeline.experiment_critic.review", return_value=("sound", "")),
        patch("experiments.experiment_pipeline.local_runner.run", return_value=failed_result),
        patch("experiments.experiment_pipeline.get_result", return_value=None),
        patch("experiments.experiment_pipeline.delete_result"),
        patch("experiments.experiment_pipeline.save_result") as mock_save,
        patch("experiments.experiment_pipeline.update_experiment_status") as mock_status,
        patch("experiments.experiment_pipeline.update_experiment_code") as mock_code,
        patch("experiments.experiment_pipeline.increment_retry"),
        patch("experiments.experiment_pipeline.code_repairer.repair",
              return_value=("fixed code", "missing import")) as mock_repair,
    ):
        experiment_pipeline.run(state)

    # stdout tail preserved on the saved failed result (repair signal not destroyed)
    saved = mock_save.call_args[0][0]
    assert "ImportError" in saved.stdout

    mock_repair.assert_called_once()
    mock_code.assert_called_once_with("exp_002", "fixed code")
    statuses = [c.args[1] for c in mock_status.call_args_list]
    assert statuses[-1] == "pending"


def test_failed_run_repair_unavailable_marks_failed():
    """If no repair can be generated, the experiment is marked failed."""
    from experiments import experiment_pipeline

    exp = _make_experiment("exp_003")
    failed_result = _make_result("exp_003", exit_code=1)
    failed_result.metrics = "{}"

    state = RunState(cycle_id="t", started_at=datetime.utcnow())
    with (
        patch("experiments.experiment_pipeline.get_experiments_by_status", return_value=[exp]),
        patch("experiments.experiment_pipeline.code_validator.validate_with_retry", return_value=(exp.generated_code, True)),
        patch("experiments.experiment_pipeline.router.decide_target", return_value="local"),
        patch("experiments.experiment_pipeline.experiment_critic.review", return_value=("sound", "")),
        patch("experiments.experiment_pipeline.local_runner.run", return_value=failed_result),
        patch("experiments.experiment_pipeline.get_result", return_value=None),
        patch("experiments.experiment_pipeline.delete_result"),
        patch("experiments.experiment_pipeline.save_result"),
        patch("experiments.experiment_pipeline.update_experiment_status") as mock_status,
        patch("experiments.experiment_pipeline.update_experiment_code") as mock_code,
        patch("experiments.experiment_pipeline.increment_retry"),
        patch("experiments.experiment_pipeline.code_repairer.repair", return_value=None),
    ):
        experiment_pipeline.run(state)

    statuses = [c.args[1] for c in mock_status.call_args_list]
    assert statuses[-1] == "failed"
    mock_code.assert_not_called()


def test_failed_run_at_retry_cap_skips_repair():
    """At the retry cap, no repair is attempted — experiment is marked failed."""
    from experiments import experiment_pipeline

    exp = _make_experiment("exp_004")
    exp.retry_count = 2  # this failure is the 3rd run
    failed_result = _make_result("exp_004", exit_code=1)
    failed_result.metrics = "{}"

    state = RunState(cycle_id="t", started_at=datetime.utcnow())
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
        patch("experiments.experiment_pipeline.code_repairer.repair") as mock_repair,
    ):
        experiment_pipeline.run(state)

    mock_repair.assert_not_called()
    statuses = [c.args[1] for c in mock_status.call_args_list]
    assert statuses[-1] == "failed"


def test_successful_run_clears_stdout(in_memory_engine):
    """On success, stdout is cleared to save storage."""
    from experiments import experiment_pipeline

    exp = _make_experiment("exp_005")
    result = _make_result("exp_005", exit_code=0)  # has metrics

    state = RunState(cycle_id="t", started_at=datetime.utcnow())
    with (
        patch("experiments.experiment_pipeline.get_experiments_by_status", return_value=[exp]),
        patch("experiments.experiment_pipeline.code_validator.validate_with_retry", return_value=(exp.generated_code, True)),
        patch("experiments.experiment_pipeline.router.decide_target", return_value="local"),
        patch("experiments.experiment_pipeline.experiment_critic.review", return_value=("sound", "")),
        patch("experiments.experiment_pipeline.local_runner.run", return_value=result),
        patch("experiments.experiment_pipeline.get_result", return_value=None),
        patch("experiments.experiment_pipeline.delete_result"),
        patch("experiments.experiment_pipeline.save_result") as mock_save,
        patch("experiments.experiment_pipeline.update_experiment_status") as mock_status,
        patch("experiments.experiment_pipeline.increment_retry"),
    ):
        experiment_pipeline.run(state)

    assert mock_save.call_args[0][0].stdout == ""
    statuses = [c.args[1] for c in mock_status.call_args_list]
    assert statuses[-1] == "completed"


def test_critic_flagged_experiment_repaired_before_run():
    """A flawed critic verdict repairs + requeues without running the experiment."""
    from experiments import experiment_pipeline

    exp = _make_experiment("exp_006")
    state = RunState(cycle_id="t", started_at=datetime.utcnow())
    with (
        patch("experiments.experiment_pipeline.get_experiments_by_status", return_value=[exp]),
        patch("experiments.experiment_pipeline.code_validator.validate_with_retry", return_value=(exp.generated_code, True)),
        patch("experiments.experiment_pipeline.experiment_critic.review", return_value=("flawed", "uses wrong dataset")),
        patch("experiments.experiment_pipeline.code_repairer.repair", return_value=("fixed", "switched dataset")),
        patch("experiments.experiment_pipeline.update_experiment_code") as mock_code,
        patch("experiments.experiment_pipeline.update_experiment_status") as mock_status,
        patch("experiments.experiment_pipeline.increment_retry") as mock_retry,
        patch("experiments.experiment_pipeline.local_runner.run") as mock_run,
    ):
        experiment_pipeline.run(state)

    mock_code.assert_called_once_with("exp_006", "fixed")
    mock_retry.assert_called_once()
    mock_run.assert_not_called()  # requeued, not executed this pass
    statuses = [c.args[1] for c in mock_status.call_args_list]
    assert statuses[-1] == "pending"


def test_critic_flawed_but_no_repair_runs_anyway():
    """If repair is unavailable, a flawed experiment still runs (critic is advisory)."""
    from experiments import experiment_pipeline

    exp = _make_experiment("exp_007")
    result = _make_result("exp_007", exit_code=0)
    state = RunState(cycle_id="t", started_at=datetime.utcnow())
    with (
        patch("experiments.experiment_pipeline.get_experiments_by_status", return_value=[exp]),
        patch("experiments.experiment_pipeline.code_validator.validate_with_retry", return_value=(exp.generated_code, True)),
        patch("experiments.experiment_pipeline.experiment_critic.review", return_value=("flawed", "unsure")),
        patch("experiments.experiment_pipeline.code_repairer.repair", return_value=None),
        patch("experiments.experiment_pipeline.router.decide_target", return_value="local"),
        patch("experiments.experiment_pipeline.local_runner.run", return_value=result) as mock_run,
        patch("experiments.experiment_pipeline.get_result", return_value=None),
        patch("experiments.experiment_pipeline.delete_result"),
        patch("experiments.experiment_pipeline.save_result"),
        patch("experiments.experiment_pipeline.update_experiment_status") as mock_status,
        patch("experiments.experiment_pipeline.increment_retry"),
    ):
        experiment_pipeline.run(state)

    mock_run.assert_called_once()
    statuses = [c.args[1] for c in mock_status.call_args_list]
    assert statuses[-1] == "completed"
