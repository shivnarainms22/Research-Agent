"""CRUD for Lesson — durable takeaways that feed back into experiment codegen."""
from __future__ import annotations

import uuid
from typing import Optional

import structlog
from sqlmodel import Session, select

from core.database import get_engine
from core.models import Lesson

log = structlog.get_logger()


def save_lesson(experiment_id: str, text: str, category: str = "repair",
                paper_id: Optional[str] = None) -> None:
    """Persist a lesson (best-effort — never raises to the caller)."""
    if not text:
        return
    try:
        with Session(get_engine(), expire_on_commit=False) as session:
            session.add(Lesson(
                id=str(uuid.uuid4()),
                experiment_id=experiment_id,
                paper_id=paper_id,
                category=category,
                text=text,
            ))
            session.commit()
    except Exception as e:
        log.warning("lesson_store.save_failed", error=str(e))


def get_recent_lessons(limit: int = 5) -> list[Lesson]:
    with Session(get_engine()) as session:
        return list(session.exec(
            select(Lesson).order_by(Lesson.created_at.desc()).limit(limit)
        ).all())
