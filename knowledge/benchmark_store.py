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
