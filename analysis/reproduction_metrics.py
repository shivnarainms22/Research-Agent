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
