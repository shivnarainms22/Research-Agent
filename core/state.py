"""Crash-safe RunState management using atomic JSON writes + filelock."""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import structlog
from filelock import FileLock, Timeout

from config import settings
from core.models import RunState

log = structlog.get_logger()

_LOCK_TIMEOUT = 10  # seconds


def _state_path(cycle_id: str) -> Path:
    return settings.state_dir / f"{cycle_id}.json"


def _lock_path(cycle_id: str) -> Path:
    return settings.state_dir / f"{cycle_id}.lock"


def save_state(state: RunState) -> None:
    """Atomically write state to JSON (os.replace for crash safety)."""
    path = _state_path(state.cycle_id)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.replace(tmp_path, path)
    log.debug("state.saved", cycle_id=state.cycle_id, stage=state.current_stage)


def load_state(cycle_id: str) -> Optional[RunState]:
    path = _state_path(cycle_id)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return RunState(**data)


def find_incomplete_states() -> list[RunState]:
    """Scan state dir for unfinished pipeline runs."""
    states = []
    for p in settings.state_dir.glob("*.json"):
        try:
            state = RunState(**json.loads(p.read_text(encoding="utf-8")))
            if not state.is_complete:
                states.append(state)
        except Exception:
            pass
    return states


def acquire_pipeline_lock() -> Optional[FileLock]:
    """Acquire a global pipeline lock (prevents concurrent runs).
    Returns lock object if acquired, None if already locked."""
    lock_path = settings.state_dir / "pipeline.lock"
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=_LOCK_TIMEOUT)
        return lock
    except Timeout:
        log.warning("pipeline.lock_busy")
        return None


def new_state(cycle_id: str) -> RunState:
    state = RunState(
        cycle_id=cycle_id,
        started_at=datetime.utcnow(),
    )
    save_state(state)
    return state


def advance_stage(state: RunState, completed: str, next_stage: str) -> RunState:
    state.completed_stages.append(completed)
    state.current_stage = next_stage
    save_state(state)
    log.info("pipeline.stage_advance", completed=completed, next=next_stage)
    return state


def mark_complete(state: RunState) -> RunState:
    state.is_complete = True
    save_state(state)
    log.info("pipeline.complete", cycle_id=state.cycle_id)
    return state


def log_error(state: RunState, stage: str, error: str) -> RunState:
    state.error_log.append({"stage": stage, "error": error, "at": datetime.utcnow().isoformat()})
    save_state(state)
    return state
