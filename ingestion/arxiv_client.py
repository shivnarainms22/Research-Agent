"""ArXiv ingestion client."""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from typing import Iterator

import arxiv
import structlog

from config import settings
from core.models import Paper

log = structlog.get_logger()


def _paper_id(arxiv_id: str) -> str:
    return hashlib.sha256(f"arxiv:{arxiv_id}".encode()).hexdigest()[:32]


def fetch_papers(days_back: int = 1, max_results: int | None = None) -> list[Paper]:
    """Fetch recent arXiv papers matching configured keywords and categories."""
    if max_results is None:
        max_results = settings.max_papers_per_cycle

    since = datetime.utcnow() - timedelta(days=days_back)

    # Build query: categories OR keywords
    cat_q = " OR ".join(f"cat:{c}" for c in settings.arxiv_categories)
    kw_q = " OR ".join(f'ti:"{k}" OR abs:"{k}"' for k in settings.arxiv_keywords[:8])
    query = f"({cat_q}) AND ({kw_q})"

    client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    papers: list[Paper] = []
    try:
        for result in client.results(search):
            if result.published.date() < since.date():
                break
            arxiv_id = result.entry_id.split("/abs/")[-1]
            paper = Paper(
                id=_paper_id(arxiv_id),
                title=result.title.strip(),
                abstract=result.summary.strip(),
                source="arxiv",
                source_id=arxiv_id,
                url=result.entry_id,
                pdf_url=result.pdf_url,
                published_date=result.published.date(),
                fetched_at=datetime.utcnow(),
                tags="[]",
                citation_count=None,
                status="fetched",
            )
            papers.append(paper)
        log.info("arxiv.fetched", count=len(papers), days_back=days_back)
    except Exception as e:
        log.error("arxiv.fetch_error", error=str(e))

    return papers
