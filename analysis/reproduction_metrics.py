"""Reproduction-rate computation + cycle-snapshot orchestration (SP1 of Eval Harness).

See docs/superpowers/specs/2026-05-22-reproduction-rate-tracking-design.md.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import structlog

from knowledge.eval_metric_store import MetricPoint

log = structlog.get_logger()

_COMPARABLE = {"fully_reproduced", "partially_reproduced", "not_reproduced"}
_UNKNOWN = {"", "unknown", None}


@dataclass
class VerdictRow:
    """One experiment-result verdict, denormalized for tallying."""
    overall: str
    difficulty: str
    target: str
    source: str
    recorded_at: datetime


def tally(rows: list[VerdictRow]) -> list[MetricPoint]:
    """Pure: bucket rows by every dimension; compute reproduction_rate + partial_rate per bucket.

    Always emits the "overall" point pair, even on empty input (denominator=0, value=None).
    Sub-dimension buckets are skipped when the key is unknown/empty.
    """
    points: list[MetricPoint] = []
    points.extend(_compute_for_bucket(rows, dimension="overall"))

    for dim_name, getter in (
        ("difficulty", lambda r: r.difficulty),
        ("target", lambda r: r.target),
        ("source", lambda r: r.source),
    ):
        groups: dict[str, list[VerdictRow]] = {}
        for r in rows:
            key = getter(r)
            if key in _UNKNOWN:
                continue
            groups.setdefault(key, []).append(r)
        for key, group_rows in groups.items():
            points.extend(_compute_for_bucket(group_rows, dimension=f"{dim_name}:{key}"))

    return points


def _compute_for_bucket(rows: list[VerdictRow], dimension: str) -> list[MetricPoint]:
    comparable = [r for r in rows if r.overall in _COMPARABLE]
    n = len(comparable)
    fully = sum(1 for r in comparable if r.overall == "fully_reproduced")
    partial = sum(1 for r in comparable if r.overall == "partially_reproduced")
    not_repro = n - fully - partial

    repro_value: Optional[float] = (fully / n) if n else None
    partial_value: Optional[float] = (partial / n) if n else None
    context = json.dumps({"fully": fully, "partial": partial, "not": not_repro})

    return [
        MetricPoint(
            metric="reproduction_rate", dimension=dimension,
            value=repro_value, numerator=fully, denominator=n, context=context,
        ),
        MetricPoint(
            metric="partial_rate", dimension=dimension,
            value=partial_value, numerator=partial, denominator=n, context=context,
        ),
    ]


def gather_verdicts(experiment_ids: Optional[Iterable[str]] = None) -> list[VerdictRow]:
    """Join ExperimentResult x Experiment x PaperAnalysis x Paper into VerdictRows.

    Filters:
      * non-ablation experiments only (parent_experiment_id IS NULL)
      * results with a baseline_comparison whose `overall` is one of the 3 real verdicts
      * if `experiment_ids` is given, restrict to that set

    Missing PaperAnalysis → difficulty='unknown'. Missing Paper → source='unknown'.
    Malformed baseline_comparison JSON → logged WARN and skipped.
    """
    from sqlmodel import Session, select
    from core.database import get_engine
    from core.models import Experiment, ExperimentResult, Paper, PaperAnalysis

    with Session(get_engine()) as session:
        exp_stmt = select(Experiment).where(Experiment.parent_experiment_id == None)  # noqa: E711
        if experiment_ids is not None:
            ids = list(experiment_ids)
            if not ids:
                return []
            exp_stmt = exp_stmt.where(Experiment.id.in_(ids))
        experiments = list(session.exec(exp_stmt).all())
        if not experiments:
            return []

        exp_by_id = {e.id: e for e in experiments}
        paper_ids = {e.paper_id for e in experiments}

        results = list(session.exec(
            select(ExperimentResult).where(ExperimentResult.experiment_id.in_(list(exp_by_id)))
        ).all())

        papers = {p.id: p for p in session.exec(
            select(Paper).where(Paper.id.in_(list(paper_ids)))
        ).all()}
        analyses = {a.paper_id: a for a in session.exec(
            select(PaperAnalysis).where(PaperAnalysis.paper_id.in_(list(paper_ids)))
        ).all()}

    out: list[VerdictRow] = []
    for r in results:
        if not r.baseline_comparison:
            continue
        try:
            bc = json.loads(r.baseline_comparison)
        except (json.JSONDecodeError, TypeError):
            log.warning("reproduction_metrics.malformed_baseline_comparison", result_id=r.id)
            continue
        overall = bc.get("overall")
        if overall not in _COMPARABLE:
            continue
        exp = exp_by_id.get(r.experiment_id)
        if not exp:
            continue
        paper = papers.get(exp.paper_id)
        analysis = analyses.get(exp.paper_id)
        out.append(VerdictRow(
            overall=overall,
            difficulty=(analysis.reproducibility_difficulty if analysis else "unknown"),
            target=exp.execution_target,
            source=(paper.source if paper else "unknown"),
            recorded_at=r.recorded_at,
        ))
    return out


def backfill_from_history() -> int:
    """One-shot: bucket all historical comparable results by ISO week of recorded_at.

    Idempotent: existing backfill-* cycle_ids are skipped. Returns the count of rows written.
    """
    from sqlmodel import Session, select
    from core.database import get_engine
    from core.models import EvalMetric
    from knowledge.eval_metric_store import save_metrics

    rows = gather_verdicts(experiment_ids=None)
    if not rows:
        return 0

    # Bucket by ISO week
    buckets: dict[str, list[VerdictRow]] = {}
    for r in rows:
        iso_year, iso_week, _ = r.recorded_at.isocalendar()
        cycle_id = f"backfill-{iso_year}-W{iso_week:02d}"
        buckets.setdefault(cycle_id, []).append(r)

    # Idempotency: skip cycle_ids already present.
    # NOTE: SQLModel single-column `select(EvalMetric.cycle_id)` returns SCALARS, not tuples
    # (see CLAUDE.md "Bugs Fixed" #2). Do NOT index `row[0]`.
    with Session(get_engine()) as session:
        existing = set(session.exec(
            select(EvalMetric.cycle_id).where(EvalMetric.cycle_id.in_(list(buckets)))
        ).all())

    written = 0
    for cycle_id, week_rows in sorted(buckets.items()):
        if cycle_id in existing:
            continue
        points = tally(week_rows)
        overall_repro = next(
            (p for p in points if p.metric == "reproduction_rate" and p.dimension == "overall"),
            None,
        )
        if overall_repro is None or overall_repro.denominator == 0:
            continue  # nothing comparable this week
        save_metrics(points, cycle_id=cycle_id)
        written += len(points)
    log.info("reproduction_metrics.backfill_done", rows_written=written)
    return written


def record_cycle_snapshot(state) -> None:
    """End-of-stage hook: lazy-backfill if store empty, then tally this cycle's experiments and save.

    `state` is a RunState. Empty `experiment_ids_this_cycle` still writes overall=None
    so the trend honestly records the gap.
    """
    from knowledge.eval_metric_store import count_rows, save_metrics

    if count_rows() == 0:
        backfill_from_history()

    verdicts = gather_verdicts(experiment_ids=state.experiment_ids_this_cycle)
    points = tally(verdicts)
    save_metrics(points, cycle_id=state.cycle_id)
    log.info(
        "reproduction_metrics.snapshot_recorded",
        cycle_id=state.cycle_id, points_written=len(points),
        verdict_rows=len(verdicts),
    )
