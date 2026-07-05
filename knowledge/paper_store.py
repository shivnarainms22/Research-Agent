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
        try:
            return session.exec(
                select(PaperAnalysis).where(PaperAnalysis.paper_id == paper_id)
            ).first()
        except ValueError:
            # Defensive: a legacy column-corrupt row (non-datetime text in the DATETIME
            # analyzed_at column — see core/repair.py) crashes the full-entity load. Fall
            # back to the non-DATETIME columns so callers still get the intact fields
            # (contributions/methods/scores) instead of a hard failure. Run core.repair to
            # fix the underlying data permanently.
            log.warning("paper_store.corrupt_analysis_fallback", paper_id=paper_id)
            row = session.exec(
                select(
                    PaperAnalysis.id, PaperAnalysis.paper_id,
                    PaperAnalysis.key_contributions, PaperAnalysis.methods_described,
                    PaperAnalysis.reproducible_experiments,
                    PaperAnalysis.novelty_score, PaperAnalysis.relevance_score,
                ).where(PaperAnalysis.paper_id == paper_id)
            ).first()
            if row is None:
                return None
            return PaperAnalysis(
                id=row[0], paper_id=row[1], key_contributions=row[2],
                methods_described=row[3], reproducible_experiments=row[4],
                novelty_score=row[5], relevance_score=row[6],
            )


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


def count_papers() -> int:
    from sqlalchemy import func
    with Session(get_engine()) as session:
        return session.exec(select(func.count()).select_from(Paper)).one()
