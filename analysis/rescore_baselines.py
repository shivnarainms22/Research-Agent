"""One-time re-score of historical baseline comparisons after the comparator fix.

The pre-fix comparator marked nearly everything not_reproduced (prose metric names
never matched snake_case keys; see baseline_comparator). This recomputes every stored
comparison from current metrics with the fixed comparator, then regenerates the
reproduction_rate / partial_rate eval metrics from the corrected data.

Idempotent — safe to re-run. No Claude calls (conclusions are left untouched).

    uv run python -m analysis.rescore_baselines
"""
from __future__ import annotations

import json

import structlog
from sqlmodel import Session, select

from analysis import baseline_comparator, reproduction_metrics
from core.database import get_engine
from core.models import EvalMetric, Experiment, ExperimentResult

log = structlog.get_logger()


def rescore_all() -> dict:
    """Recompute stored baseline_comparison for all results, then rebuild repro metrics."""
    rescored = 0
    with Session(get_engine(), expire_on_commit=False) as session:
        results = list(session.exec(select(ExperimentResult)).all())
        for result in results:
            if not result.metrics or result.metrics == "{}":
                continue
            exp = session.get(Experiment, result.experiment_id)
            if exp is None:
                continue
            comparison = baseline_comparator.compare(
                result, exp.paper_id, experiment_title=exp.title
            )
            result.baseline_comparison = json.dumps(comparison)
            session.add(result)
            rescored += 1

        # Drop stale reproduction_rate/partial_rate points (computed from broken data).
        # Leave other metrics (e.g. benchmark_accuracy) intact.
        stale = list(session.exec(
            select(EvalMetric).where(
                EvalMetric.metric.in_(["reproduction_rate", "partial_rate"])
            )
        ).all())
        for row in stale:
            session.delete(row)
        session.commit()

    written = reproduction_metrics.backfill_from_history()
    log.info("rescore_baselines.done", rescored=rescored, metric_rows_written=written)
    return {"rescored": rescored, "metric_rows_written": written}


if __name__ == "__main__":
    summary = rescore_all()
    print(f"Re-scored {summary['rescored']} results; "
          f"wrote {summary['metric_rows_written']} reproduction-metric rows.")
