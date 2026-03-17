"""Full-cycle pipeline orchestrator with crash recovery."""
from __future__ import annotations

import uuid
from datetime import datetime

import structlog

from core import token_tracker
from core.state import (
    RunState,
    acquire_pipeline_lock,
    advance_stage,
    find_incomplete_states,
    log_error,
    mark_complete,
    new_state,
    save_state,
)
from ingestion import ingestion_pipeline
from synthesis import synthesis_pipeline
from experiments import experiment_pipeline
from analysis import analysis_pipeline
from reporting import report_generator

log = structlog.get_logger()

STAGES = ["ingestion", "synthesis", "experiments", "analysis", "reporting"]


def run_cycle(days_back: int = 1) -> RunState:
    """Run a full pipeline cycle (with crash recovery)."""
    lock = acquire_pipeline_lock()
    if lock is None:
        log.warning("pipeline_runner.already_running")
        raise RuntimeError("Pipeline already running (lock held)")

    try:
        # Check for incomplete cycle to resume
        incomplete = find_incomplete_states()
        if incomplete:
            state = sorted(incomplete, key=lambda s: s.started_at)[-1]
            log.info("pipeline_runner.resuming", cycle_id=state.cycle_id, stage=state.current_stage)
        else:
            cycle_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            state = new_state(cycle_id)
            log.info("pipeline_runner.new_cycle", cycle_id=state.cycle_id)

        token_tracker.reset()
        token_tracker.set_cycle_id(state.cycle_id)
        _run_from_stage(state, days_back)
        return state

    finally:
        lock.release()


def _run_from_stage(state: RunState, days_back: int) -> None:
    """Execute pipeline stages sequentially, starting from current_stage."""
    start_idx = STAGES.index(state.current_stage) if state.current_stage in STAGES else 0

    for stage in STAGES[start_idx:]:
        if stage in state.completed_stages:
            continue

        log.info("pipeline_runner.stage_start", stage=stage)
        try:
            if stage == "ingestion":
                ingestion_pipeline.run(state, days_back=days_back)
            elif stage == "synthesis":
                synthesis_pipeline.run(state)
            elif stage == "experiments":
                experiment_pipeline.run(state)
            elif stage == "analysis":
                analysis_pipeline.run(state)
            elif stage == "reporting":
                report_generator.generate(state)

            next_idx = STAGES.index(stage) + 1
            next_stage = STAGES[next_idx] if next_idx < len(STAGES) else "done"
            advance_stage(state, completed=stage, next_stage=next_stage)

        except Exception as e:
            log.error("pipeline_runner.stage_error", stage=stage, error=str(e))
            log_error(state, stage, str(e))
            raise

    mark_complete(state)
    totals = token_tracker.get_totals()
    state.total_input_tokens = totals["input_total"]
    state.total_output_tokens = totals["output_total"]
    save_state(state)
    log.info("pipeline_runner.cycle_complete", cycle_id=state.cycle_id,
             input_tokens=state.total_input_tokens, output_tokens=state.total_output_tokens)


def run_experiment_poll() -> None:
    """Poll and run any pending experiments (called by scheduler interval job)."""
    from experiments import experiment_pipeline as ep
    from core.state import find_incomplete_states

    # Create a minimal state for standalone experiment runs
    incomplete = find_incomplete_states()
    if incomplete:
        state = incomplete[-1]
    else:
        state = RunState(
            cycle_id=f"poll_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            started_at=datetime.utcnow(),
            current_stage="experiments",
        )

    ep.run(state)
    log.info("pipeline_runner.experiment_poll_done")
