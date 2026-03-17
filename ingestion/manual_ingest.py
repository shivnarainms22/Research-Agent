"""Manual paper ingestion: PDF bytes, plain text, or URL → Paper object."""
from __future__ import annotations

import hashlib
import re
from datetime import date
from pathlib import Path

import structlog

from core.models import Paper

log = structlog.get_logger()

_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5}(?:v\d+)?)")


def _paper_id(source_id: str) -> str:
    return hashlib.sha256(source_id.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def from_pdf_bytes(data: bytes, filename: str = "upload.pdf") -> Paper:
    """Extract text + metadata from PDF bytes and return an unsaved Paper."""
    import fitz  # PyMuPDF

    doc = fitz.open(stream=data, filetype="pdf")
    meta_title = (doc.metadata.get("title") or "").strip()
    title = meta_title or Path(filename).stem

    pages_text = [page.get_text() for page in doc]
    full_text = "\n".join(pages_text)[:50_000]
    abstract = full_text[:800].strip()

    source_id = f"manual_pdf_{filename}"
    return Paper(
        id=_paper_id(source_id),
        title=title,
        abstract=abstract,
        source="manual",
        source_id=source_id,
        url="",
        published_date=date.today(),
        full_text=full_text,
        status="fetched",
    )


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def from_text(text: str, filename: str = "upload.txt") -> Paper:
    """Build a Paper from plain text (first line = title, next few = abstract)."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    title = lines[0][:200] if lines else Path(filename).stem
    abstract = " ".join(lines[1:6])[:800] if len(lines) > 1 else ""

    source_id = f"manual_text_{filename}"
    return Paper(
        id=_paper_id(source_id),
        title=title,
        abstract=abstract,
        source="manual",
        source_id=source_id,
        url="",
        published_date=date.today(),
        full_text=text[:50_000],
        status="fetched",
    )


# ---------------------------------------------------------------------------
# URL
# ---------------------------------------------------------------------------

def from_url(url: str) -> Paper:
    """Fetch a paper from a URL. Handles arXiv specially; falls back to generic HTML scrape."""
    m = _ARXIV_RE.search(url)
    if m:
        return _from_arxiv(m.group(1), url)
    return _from_generic_url(url)


def _from_arxiv(arxiv_id: str, original_url: str) -> Paper:
    import arxiv
    from ingestion.fulltext_fetcher import fetch_arxiv_fulltext

    base_id = arxiv_id.split("v")[0]
    results = list(arxiv.Client().results(arxiv.Search(id_list=[base_id])))
    if not results:
        raise ValueError(f"arXiv paper not found: {arxiv_id}")

    r = results[0]
    full_text = fetch_arxiv_fulltext(base_id)

    return Paper(
        id=_paper_id(base_id),
        title=r.title,
        abstract=r.summary,
        source="arxiv",
        source_id=base_id,
        url=str(r.entry_id),
        pdf_url=r.pdf_url,
        published_date=r.published.date() if r.published else date.today(),
        full_text=full_text,
        status="fetched",
    )


def _from_generic_url(url: str) -> Paper:
    import httpx
    from bs4 import BeautifulSoup

    resp = httpx.get(
        url, timeout=30, follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 research-agent/1.0"},
    )
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    if not title:
        h1 = soup.find("h1")
        title = h1.get_text(strip=True) if h1 else url

    # Strip noise tags
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    body_text = soup.get_text(separator="\n", strip=True)
    abstract = body_text[:800]

    source_id = f"manual_url_{url}"
    return Paper(
        id=_paper_id(source_id),
        title=title[:200],
        abstract=abstract,
        source="manual",
        source_id=source_id,
        url=url,
        published_date=date.today(),
        full_text=body_text[:50_000],
        status="fetched",
    )
