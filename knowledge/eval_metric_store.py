"""CRUD for EvalMetric — generic time-series store for eval-harness metrics (SP1+)."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from sqlmodel import Session, select, func

from core.database import get_engine
from core.models import EvalMetric


@dataclass
class MetricPoint:
    """In-memory shape of one metric row; produced by tally(), consumed by save_metrics()."""
    metric: str
    dimension: str
    value: Optional[float]
    numerator: int
    denominator: int
    context: str = "{}"


def save_metrics(points: list[MetricPoint], cycle_id: str) -> None:
    """Persist a snapshot's rows in one session. No-op on empty input."""
    if not points:
        return
    now = datetime.utcnow()
    rows = [
        EvalMetric(
            id=str(uuid.uuid4()),
            metric=p.metric,
            dimension=p.dimension,
            value=p.value,
            numerator=p.numerator,
            denominator=p.denominator,
            cycle_id=cycle_id,
            recorded_at=now,
            context=p.context,
        )
        for p in points
    ]
    with Session(get_engine(), expire_on_commit=False) as session:
        for r in rows:
            session.add(r)
        session.commit()


def count_rows() -> int:
    """Total rows in eval_metric. Used to gate the lazy backfill."""
    with Session(get_engine()) as session:
        return session.exec(select(func.count()).select_from(EvalMetric)).one()


def get_latest(metric: str, dimension: str = "overall") -> Optional[EvalMetric]:
    """Most recent row for (metric, dimension). None if no rows."""
    with Session(get_engine()) as session:
        return session.exec(
            select(EvalMetric)
            .where(EvalMetric.metric == metric)
            .where(EvalMetric.dimension == dimension)
            .order_by(EvalMetric.recorded_at.desc())
            .limit(1)
        ).first()


def get_previous(
    metric: str, dimension: str, before_cycle_id: str
) -> Optional[EvalMetric]:
    """Most recent row for (metric, dimension) excluding the given cycle_id."""
    with Session(get_engine()) as session:
        return session.exec(
            select(EvalMetric)
            .where(EvalMetric.metric == metric)
            .where(EvalMetric.dimension == dimension)
            .where(EvalMetric.cycle_id != before_cycle_id)
            .order_by(EvalMetric.recorded_at.desc())
            .limit(1)
        ).first()


def get_trend(
    metric: str, dimension: str = "overall", limit: int = 30
) -> list[EvalMetric]:
    """Most recent N snapshots, returned oldest-first (for charts/sparklines)."""
    with Session(get_engine()) as session:
        rows = list(session.exec(
            select(EvalMetric)
            .where(EvalMetric.metric == metric)
            .where(EvalMetric.dimension == dimension)
            .order_by(EvalMetric.recorded_at.desc())
            .limit(limit)
        ).all())
    return list(reversed(rows))
