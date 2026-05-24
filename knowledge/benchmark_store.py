"""CRUD for the ground-truth benchmark set (SP2 of the Eval Harness). Sole DB interface."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Session, select

from core.database import get_engine
from core.models import BenchmarkItem, BenchmarkRun, BenchmarkItemResult


def save_item(item: BenchmarkItem) -> None:
    """Insert or update a golden-set item (upsert by primary key via merge)."""
    item.updated_at = datetime.utcnow()
    with Session(get_engine(), expire_on_commit=False) as session:
        session.merge(item)
        session.commit()


def get_items(active_only: bool = True) -> list[BenchmarkItem]:
    with Session(get_engine()) as session:
        stmt = select(BenchmarkItem)
        if active_only:
            stmt = stmt.where(BenchmarkItem.active == True)  # noqa: E712
        return list(session.exec(stmt).all())


def get_item(item_id: str) -> Optional[BenchmarkItem]:
    with Session(get_engine()) as session:
        return session.get(BenchmarkItem, item_id)


def deactivate_item(item_id: str) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        item = session.get(BenchmarkItem, item_id)
        if item:
            item.active = False
            item.updated_at = datetime.utcnow()
            session.add(item)
            session.commit()


def save_run(run: BenchmarkRun, item_results: list[BenchmarkItemResult]) -> None:
    """Persist a scoring run and all its per-item results in one session."""
    with Session(get_engine(), expire_on_commit=False) as session:
        session.add(run)
        for r in item_results:
            session.add(r)
        session.commit()


def get_runs(limit: int = 30) -> list[BenchmarkRun]:
    """Run history, most recent first."""
    with Session(get_engine()) as session:
        return list(session.exec(
            select(BenchmarkRun).order_by(BenchmarkRun.recorded_at.desc()).limit(limit)
        ).all())


def get_latest_run() -> Optional[BenchmarkRun]:
    with Session(get_engine()) as session:
        return session.exec(
            select(BenchmarkRun).order_by(BenchmarkRun.recorded_at.desc()).limit(1)
        ).first()


def get_item_results(run_id: str) -> list[BenchmarkItemResult]:
    with Session(get_engine()) as session:
        return list(session.exec(
            select(BenchmarkItemResult).where(BenchmarkItemResult.run_id == run_id)
        ).all())
