"""CRUD for Contradiction table."""
from __future__ import annotations

from datetime import datetime, timedelta

from sqlmodel import Session, select

from core.database import get_engine
from core.models import Contradiction


def save_contradiction(c: Contradiction) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        session.add(c)
        session.commit()


def get_recent_contradictions(days: int = 30) -> list[Contradiction]:
    cutoff = datetime.utcnow() - timedelta(days=days)
    with Session(get_engine()) as session:
        return list(
            session.exec(
                select(Contradiction).where(Contradiction.detected_at >= cutoff)
            ).all()
        )


def get_all_contradictions() -> list[Contradiction]:
    with Session(get_engine()) as session:
        return list(session.exec(select(Contradiction)).all())


def get_contradictions_for_paper(paper_id: str) -> list[Contradiction]:
    with Session(get_engine()) as session:
        return list(
            session.exec(
                select(Contradiction).where(Contradiction.paper_id_new == paper_id)
            ).all()
        )
