"""Orchestrates all ingestion sources into SQLite."""
from __future__ import annotations

import structlog
from sqlmodel import Session

from config import settings
from core.database import get_engine
from core.models import Paper
from core.state import RunState, save_state
from ingestion import arxiv_client, semantic_scholar_client, substack_scraper
from ingestion.deduplicator import deduplicate

log = structlog.get_logger()


def run(state: RunState, days_back: int = 1) -> list[str]:
    """Run all ingestion sources and persist new papers. Returns list of new paper IDs."""
    all_papers: list[Paper] = []

    log.info("ingestion.start")

    # ArXiv
    try:
        arxiv_papers = arxiv_client.fetch_papers(days_back=days_back)
        all_papers.extend(arxiv_papers)
    except Exception as e:
        log.error("ingestion.arxiv_failed", error=str(e))

    # Semantic Scholar
    try:
        s2_papers = semantic_scholar_client.fetch_papers(days_back=days_back)
        all_papers.extend(s2_papers)
    except Exception as e:
        log.error("ingestion.s2_failed", error=str(e))

    # Substack
    try:
        sub_papers = substack_scraper.fetch_papers(days_back=max(days_back, 7))
        all_papers.extend(sub_papers)
    except Exception as e:
        log.error("ingestion.substack_failed", error=str(e))

    # Deduplicate
    new_papers = deduplicate(all_papers)

    # Enforce per-cycle limit
    new_papers = new_papers[: settings.max_papers_per_cycle]

    # Persist to SQLite
    engine = get_engine()
    with Session(engine, expire_on_commit=False) as session:
        for p in new_papers:
            session.add(p)
        session.commit()

    new_ids = [p.id for p in new_papers]
    state.paper_ids_this_cycle.extend(new_ids)
    save_state(state)

    log.info("ingestion.complete", new_papers=len(new_papers))
    return new_ids
