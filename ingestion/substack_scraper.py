"""Substack / RSS feed scraper."""
from __future__ import annotations

import hashlib
from datetime import datetime, timedelta

import feedparser
import structlog
from bs4 import BeautifulSoup

from config import settings
from core.models import Paper

log = structlog.get_logger()


def _paper_id(url: str) -> str:
    return hashlib.sha256(f"substack:{url}".encode()).hexdigest()[:32]


def fetch_papers(days_back: int = 7) -> list[Paper]:
    since = datetime.utcnow() - timedelta(days=days_back)
    papers: list[Paper] = []

    for feed_url in settings.substack_rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                # Parse publication date
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])
                elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                    published = datetime(*entry.updated_parsed[:6])
                else:
                    published = datetime.utcnow()

                if published < since:
                    continue

                link = getattr(entry, "link", "")
                title = getattr(entry, "title", "").strip()
                content_list = getattr(entry, "content", [])
                content_val = content_list[0].get("value", "") if content_list else ""
                summary = getattr(entry, "summary", "") or content_val

                # Strip HTML tags for abstract
                try:
                    abstract = BeautifulSoup(summary, "lxml").get_text()[:2000]
                except Exception:
                    abstract = summary[:2000]

                if not title or not abstract:
                    continue

                paper = Paper(
                    id=_paper_id(link),
                    title=title,
                    abstract=abstract,
                    source="substack",
                    source_id=link,
                    url=link,
                    pdf_url=None,
                    published_date=published.date(),
                    fetched_at=datetime.utcnow(),
                    tags="[]",
                    citation_count=None,
                    status="fetched",
                )
                papers.append(paper)
        except Exception as e:
            log.error("substack.fetch_error", feed=feed_url, error=str(e))

    log.info("substack.fetched", count=len(papers), days_back=days_back)
    return papers
