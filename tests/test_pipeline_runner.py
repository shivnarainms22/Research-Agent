"""Tests for scheduler/pipeline_runner.py poll behavior.

Bug guard: experiment polls must never persist state files — run_cycle resumes
the newest incomplete state, so a persisted poll state (current_stage=
"experiments") would make scheduled cycles skip ingestion and synthesis.
"""
from __future__ import annotations

from unittest.mock import patch


def test_experiment_poll_writes_no_state_file(tmp_path):
    # data_dir is redirected to tmp_path by the autouse _isolate_data_dir fixture.
    with patch(
        "experiments.experiment_pipeline.get_experiments_by_status", return_value=[]
    ):
        from scheduler.pipeline_runner import run_experiment_poll
        run_experiment_poll()

    assert list((tmp_path / "state").glob("*.json")) == []
