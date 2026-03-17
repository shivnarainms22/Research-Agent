"""SQLite paper registry helpers."""
from __future__ import annotations

from typing import Optional

import structlog
from sqlmodel import Session, select

from core.database import get_engine
from core.models import Paper, PaperAnalysis

log = structlog.get_logger()


def get_paper(paper_id: str) -> Optional[Paper]:
    with Session(get_engine()) as session:
        return session.get(Paper, paper_id)


def get_papers_by_status(status: str) -> list[Paper]:
    with Session(get_engine()) as session:
        return list(session.exec(select(Paper).where(Paper.status == status)).all())


def update_paper_status(paper_id: str, status: str) -> None:
    with Session(get_engine()) as session:
        paper = session.get(Paper, paper_id)
        if paper:
            paper.status = status
            session.add(paper)
            session.commit()


def save_analysis(analysis: PaperAnalysis) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        existing = session.exec(
            select(PaperAnalysis).where(PaperAnalysis.paper_id == analysis.paper_id)
        ).first()
        if existing:
            session.delete(existing)
            session.flush()
        session.add(analysis)
        session.commit()


def get_analysis(paper_id: str) -> Optional[PaperAnalysis]:
    with Session(get_engine()) as session:
        return session.exec(
            select(PaperAnalysis).where(PaperAnalysis.paper_id == paper_id)
        ).first()


def update_paper_full_text(paper_id: str, full_text: Optional[str]) -> None:
    with Session(get_engine(), expire_on_commit=False) as session:
        paper = session.get(Paper, paper_id)
        if paper:
            paper.full_text = full_text
            session.add(paper)
            session.commit()


def get_all_papers(limit: int = 1000) -> list[Paper]:
    with Session(get_engine()) as session:
        return list(session.exec(select(Paper).limit(limit)).all())
