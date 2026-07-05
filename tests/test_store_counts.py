"""Tests for cheap count queries and the centralized cost estimate."""
from __future__ import annotations

from datetime import date, datetime

from sqlmodel import Session

from core.models import Experiment, Paper


def _make_paper(pid: str) -> Paper:
    return Paper(
        id=pid, title=f"Paper {pid}", abstract="abstract", source="arxiv",
        source_id=pid, url=f"https://arxiv.org/abs/{pid}",
        published_date=date(2024, 1, 1), status="fetched",
    )


def _make_experiment(eid: str, status: str) -> Experiment:
    return Experiment(
        id=eid, paper_id="p1", title=f"Exp {eid}", hypothesis="h",
        status=status, created_at=datetime.utcnow(),
    )


def test_count_papers(in_memory_engine):
    from knowledge.paper_store import count_papers

    assert count_papers() == 0
    with Session(in_memory_engine) as session:
        session.add(_make_paper("p1"))
        session.add(_make_paper("p2"))
        session.commit()
    assert count_papers() == 2


def test_count_by_status(in_memory_engine):
    from knowledge.experiment_store import count_by_status

    assert count_by_status() == {}
    with Session(in_memory_engine) as session:
        session.add(_make_experiment("e1", "pending"))
        session.add(_make_experiment("e2", "pending"))
        session.add(_make_experiment("e3", "completed"))
        session.commit()
    assert count_by_status() == {"pending": 2, "completed": 1}


def test_estimate_cost():
    from knowledge.token_log_store import estimate_cost

    assert estimate_cost(0, 0) == 0.0
    assert estimate_cost(1_000_000, 1_000_000) == 18.0
