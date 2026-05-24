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


def _extract_numeric(metrics: dict, key: str) -> tuple[Optional[float], str]:
    """Return (value, status). status ∈ {ok, missing_metric, non_numeric}."""
    if key not in metrics:
        return None, "missing_metric"
    v = metrics[key]
    if isinstance(v, bool):  # bool is a subclass of int — reject explicitly
        return None, "non_numeric"
    if isinstance(v, (int, float)):
        return float(v), "ok"
    if isinstance(v, list):
        nums = [float(x) for x in v if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if nums:
            return sum(nums) / len(nums), "ok"
    return None, "non_numeric"


def gather_measurements(items: list[BenchmarkItem]) -> list[Measurement]:
    """For each item, read its experiment's latest ExperimentResult and pull metric_name.

    Joins Experiment/Paper/PaperAnalysis for the difficulty/source dimensions, selecting only
    the columns needed (never loads the corrupt paper_analysis.analyzed_at — see SP1 hardening).
    """
    from sqlmodel import Session, select
    from core.database import get_engine
    from core.models import Experiment, ExperimentResult, Paper, PaperAnalysis

    if not items:
        return []

    exp_ids = list({it.experiment_id for it in items})
    with Session(get_engine()) as session:
        experiments = {e.id: e for e in session.exec(
            select(Experiment).where(Experiment.id.in_(exp_ids))
        ).all()}
        paper_ids = list({e.paper_id for e in experiments.values()})
        results = {r.experiment_id: r for r in session.exec(
            select(ExperimentResult).where(ExperimentResult.experiment_id.in_(exp_ids))
        ).all()}
        sources = {pid: src for pid, src in session.exec(
            select(Paper.id, Paper.source).where(Paper.id.in_(paper_ids))
        ).all()} if paper_ids else {}
        difficulties = {pid: diff for pid, diff in session.exec(
            select(PaperAnalysis.paper_id, PaperAnalysis.reproducibility_difficulty)
            .where(PaperAnalysis.paper_id.in_(paper_ids))
        ).all()} if paper_ids else {}

    out: list[Measurement] = []
    for it in items:
        exp = experiments.get(it.experiment_id)
        paper_id = exp.paper_id if exp else None
        raw_diff = difficulties.get(paper_id, "unknown")
        difficulty = raw_diff if raw_diff in _DIFFICULTIES else "unknown"
        source = sources.get(paper_id, "unknown")

        result = results.get(it.experiment_id)
        if result is None or not result.metrics:
            out.append(Measurement(it, None, difficulty, source, "no_result"))
            continue
        try:
            metrics = json.loads(result.metrics)
        except (json.JSONDecodeError, TypeError):
            out.append(Measurement(it, None, difficulty, source, "non_numeric"))
            continue
        value, status = _extract_numeric(metrics, it.metric_name)
        out.append(Measurement(it, value, difficulty, source, status))
    return out
