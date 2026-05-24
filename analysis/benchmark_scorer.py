"""Benchmark scoring + run orchestration (SP2 of the Eval Harness).

Scores the golden set (BenchmarkItem) against each experiment's LATEST stored
ExperimentResult — no re-execution. See
docs/superpowers/specs/2026-05-24-benchmark-set-design.md.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import structlog

from core.models import BenchmarkItem

log = structlog.get_logger()

_UNKNOWN = {"", "unknown", None}
_DIFFICULTIES = {"easy", "medium", "hard"}


@dataclass
class Measurement:
    """An item plus the value pulled from its experiment's latest result."""
    item: BenchmarkItem
    measured: Optional[float]
    difficulty: str
    source: str
    status: str  # "ok" | "no_result" | "missing_metric" | "non_numeric"


@dataclass
class ItemOutcome:
    """A scored item; persisted as a BenchmarkItemResult and aggregated into a run."""
    item_id: str
    experiment_id: str
    metric_name: str
    expected_value: float
    tolerance: float
    tolerance_type: str
    measured_value: Optional[float]
    passed: Optional[bool]
    status: str  # "pass" | "fail" | "no_result" | "missing_metric" | "non_numeric"
    difficulty: str
    source: str


def _within_tolerance(measured: float, expected: float, tol: float, ttype: str) -> bool:
    if ttype == "relative" and expected != 0:
        return abs(measured - expected) <= tol * abs(expected)
    # absolute, or relative with expected == 0 (avoid a zero-width band)
    return abs(measured - expected) <= tol


def score(measurements: list[Measurement]) -> list[ItemOutcome]:
    """Pure: turn measurements into scored outcomes. Unscorable statuses pass through."""
    out: list[ItemOutcome] = []
    for m in measurements:
        it = m.item
        if m.status != "ok" or m.measured is None:
            out.append(ItemOutcome(
                item_id=it.id, experiment_id=it.experiment_id, metric_name=it.metric_name,
                expected_value=it.expected_value, tolerance=it.tolerance,
                tolerance_type=it.tolerance_type, measured_value=None, passed=None,
                status=m.status, difficulty=m.difficulty, source=m.source,
            ))
            continue
        passed = _within_tolerance(m.measured, it.expected_value, it.tolerance, it.tolerance_type)
        out.append(ItemOutcome(
            item_id=it.id, experiment_id=it.experiment_id, metric_name=it.metric_name,
            expected_value=it.expected_value, tolerance=it.tolerance,
            tolerance_type=it.tolerance_type, measured_value=m.measured,
            passed=passed, status=("pass" if passed else "fail"),
            difficulty=m.difficulty, source=m.source,
        ))
    return out
