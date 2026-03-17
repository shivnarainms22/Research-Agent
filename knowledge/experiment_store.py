"""SQLite experiment registry helpers."""
from __future__ import annotations

from typing import Optional

import structlog
from sqlmodel import Session, select

from core.database import get_engine
from core.models import Experiment, ExperimentResult

log = structlog.get_logger()


def save_experiment(exp: Experiment) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        session.add(exp)
        session.commit()


def get_experiment(exp_id: str) -> Optional[Experiment]:
    with Session(get_engine()) as session:
        return session.get(Experiment, exp_id)


def get_experiments_by_status(status: str) -> list[Experiment]:
    with Session(get_engine()) as session:
        return list(
            session.exec(select(Experiment).where(Experiment.status == status)).all()
        )


def update_experiment_status(exp_id: str, status: str, error: str | None = None) -> None:
    from datetime import datetime
    with Session(get_engine()) as session:
        exp = session.get(Experiment, exp_id)
        if exp:
            exp.status = status
            if error:
                exp.error_message = error
            if status in ("completed", "failed", "skipped"):
                exp.completed_at = datetime.utcnow()
            session.add(exp)
            session.commit()


def increment_retry(exp_id: str) -> None:
    with Session(get_engine()) as session:
        exp = session.get(Experiment, exp_id)
        if exp:
            exp.retry_count += 1
            session.add(exp)
            session.commit()


def save_result(result: ExperimentResult) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        session.add(result)
        session.commit()


def get_result(exp_id: str) -> Optional[ExperimentResult]:
    with Session(get_engine()) as session:
        return session.exec(
            select(ExperimentResult).where(ExperimentResult.experiment_id == exp_id)
        ).first()


def delete_result(exp_id: str) -> None:
    with Session(get_engine()) as session:
        result = session.exec(
            select(ExperimentResult).where(ExperimentResult.experiment_id == exp_id)
        ).first()
        if result:
            session.delete(result)
            session.commit()


def get_experiments_by_paper_id(paper_id: str) -> list[Experiment]:
    with Session(get_engine()) as session:
        return list(
            session.exec(select(Experiment).where(Experiment.paper_id == paper_id)).all()
        )


def get_all_experiments(limit: int = 1000) -> list[Experiment]:
    with Session(get_engine()) as session:
        return list(session.exec(select(Experiment).limit(limit)).all())


def get_ablations_for_parent(parent_id: str) -> list[Experiment]:
    with Session(get_engine()) as session:
        return list(
            session.exec(
                select(Experiment).where(Experiment.parent_experiment_id == parent_id)
            ).all()
        )


def update_experiment_hypothesis(exp_id: str, new_hypothesis: str) -> None:
    with Session(get_engine()) as session:
        exp = session.get(Experiment, exp_id)
        if exp:
            exp.hypothesis = new_hypothesis
            session.add(exp)
            session.commit()


def get_completed_results(limit: int = 100) -> list[ExperimentResult]:
    with Session(get_engine()) as session:
        return list(
            session.exec(
                select(ExperimentResult).order_by(ExperimentResult.recorded_at.desc()).limit(limit)
            ).all()
        )


def get_recent_failed_results(limit: int = 10) -> list[tuple]:
    from sqlalchemy import or_
    with Session(get_engine()) as session:
        results = list(
            session.exec(
                select(ExperimentResult)
                .where(or_(ExperimentResult.exit_code != 0, ExperimentResult.metrics == "{}"))
                .order_by(ExperimentResult.id.desc())
                .limit(limit)
            ).all()
        )
    pairs = []
    for r in results:
        exp = get_experiment(r.experiment_id)
        if exp:
            pairs.append((exp, r))
    return pairs
