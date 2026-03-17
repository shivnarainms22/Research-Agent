"""Semantic Scholar REST API client."""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta

import httpx
import structlog

from config import settings
from core.models import Paper

log = structlog.get_logger()

_BASE = "https://api.semanticscholar.org/graph/v1"
_FIELDS = "paperId,title,abstract,year,externalIds,url,citationCount,publicationDate,fieldsOfStudy"


def _paper_id(s2_id: str) -> str:
    return hashlib.sha256(f"s2:{s2_id}".encode()).hexdigest()[:32]


def _build_headers() -> dict:
    h = {"Accept": "application/json"}
    if settings.semantic_scholar_api_key:
        h["x-api-key"] = settings.semantic_scholar_api_key
    return h


def fetch_papers(days_back: int = 1, max_results: int | None = None) -> list[Paper]:
    if max_results is None:
        max_results = settings.max_papers_per_cycle

    since = (datetime.utcnow() - timedelta(days=days_back)).date()

    papers: list[Paper] = []
    headers = _build_headers()

    for keyword in settings.arxiv_keywords[:6]:
        if len(papers) >= max_results:
            break
        try:
            resp = httpx.get(
                f"{_BASE}/paper/search",
                params={
                    "query": keyword,
                    "fields": _FIELDS,
                    "limit": min(50, max_results - len(papers)),
                    "publicationDateOrYear": f"{since}:",
                },
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            for item in data.get("data", []):
                if not item.get("abstract"):
                    continue
                s2_id = item.get("paperId", "")
                pub_date_str = item.get("publicationDate") or f"{item.get('year', '2024')}-01-01"
                try:
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d").date()
                except ValueError:
                    pub_date = datetime.utcnow().date()

                doi = item.get("externalIds", {}).get("DOI", "")
                url = item.get("url") or f"https://www.semanticscholar.org/paper/{s2_id}"

                paper = Paper(
                    id=_paper_id(s2_id),
                    title=item["title"].strip(),
                    abstract=item["abstract"].strip(),
                    source="semantic_scholar",
                    source_id=s2_id,
                    url=url,
                    pdf_url=None,
                    published_date=pub_date,
                    fetched_at=datetime.utcnow(),
                    tags="[]",
                    citation_count=item.get("citationCount"),
                    status="fetched",
                )
                papers.append(paper)
            time.sleep(1.0)  # polite rate limiting
        except Exception as e:
            log.error("semantic_scholar.fetch_error", keyword=keyword, error=str(e))

    log.info("semantic_scholar.fetched", count=len(papers), days_back=days_back)
    return papers
