"""CRUD for ThemeCluster table."""
from __future__ import annotations

from sqlmodel import Session, select

from core.database import get_engine
from core.models import ThemeCluster


def save_theme(t: ThemeCluster) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        session.add(t)
        session.commit()


def get_all_themes() -> list[ThemeCluster]:
    with Session(get_engine()) as session:
        return list(session.exec(select(ThemeCluster)).all())


def clear_themes() -> None:
    with Session(get_engine()) as session:
        existing = list(session.exec(select(ThemeCluster)).all())
        for t in existing:
            session.delete(t)
        session.commit()
