"""CRUD for ResearchGap table."""
from __future__ import annotations

from sqlmodel import Session, select

from core.database import get_engine
from core.models import ResearchGap


def save_gaps(gaps: list[ResearchGap]) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        for g in gaps:
            session.add(g)
        session.commit()


def clear_gaps_for_cycle(cycle_id: str) -> None:
    with Session(get_engine()) as session:
        existing = list(
            session.exec(select(ResearchGap).where(ResearchGap.cycle_id == cycle_id)).all()
        )
        for g in existing:
            session.delete(g)
        session.commit()


def get_gaps(cycle_id: str | None = None) -> list[ResearchGap]:
    with Session(get_engine()) as session:
        if cycle_id:
            return list(
                session.exec(select(ResearchGap).where(ResearchGap.cycle_id == cycle_id)).all()
            )
        return list(session.exec(select(ResearchGap)).all())
