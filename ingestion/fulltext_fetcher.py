"""Fetch full text of arXiv papers from the HTML endpoint."""
from __future__ import annotations

import httpx
import structlog
from bs4 import BeautifulSoup

log = structlog.get_logger()

# Sections to extract, in priority order, with per-section char limits.
# Skips intro/related work which duplicate the abstract.
_KEY_SECTIONS: list[tuple[str, int]] = [
    ("method", 1800),
    ("approach", 1800),
    ("architecture", 1500),
    ("experiment", 1800),
    ("evaluation", 1500),
    ("result", 1500),
    ("ablation", 1000),
    ("analysis", 1000),
    ("conclusion", 800),
    ("discussion", 800),
    ("introduction", 600),  # lowest priority — abstract already covers this
]


def _extract_key_sections(soup: BeautifulSoup, max_total: int = 8000) -> str | None:
    """
    Extract signal-dense sections from arXiv HTML (latexml format).
    Tries <section> elements first, falls back to heading-based extraction.
    Returns labelled text blocks up to max_total chars, or None if nothing found.
    """
    sections_out: list[str] = []
    seen_labels: set[str] = set()
    total = 0

    # arXiv latexml HTML uses <section> elements with headings inside
    candidates = soup.find_all("section")
    use_sections = bool(candidates)
    if not use_sections:
        candidates = soup.find_all(["h2", "h3"])

    for elem in candidates:
        if use_sections:
            heading_tag = elem.find(["h2", "h3", "h4"])
            heading_text = heading_tag.get_text(strip=True) if heading_tag else ""
            body_text = elem.get_text(separator=" ", strip=True)
        else:
            heading_text = elem.get_text(strip=True)
            parts = []
            for sib in elem.find_next_siblings():
                if sib.name in ["h2", "h3"]:
                    break
                parts.append(sib.get_text(separator=" ", strip=True))
            body_text = " ".join(parts)

        heading_lower = heading_text.lower()
        limit = next(
            (lim for kw, lim in _KEY_SECTIONS if kw in heading_lower),
            None,
        )
        if limit is None:
            continue

        # Deduplicate sections with identical headings
        label = heading_text.strip()
        if label in seen_labels:
            continue
        seen_labels.add(label)

        snippet = body_text[:limit].strip()
        if not snippet:
            continue

        sections_out.append(f"[{label}]\n{snippet}")
        total += len(snippet)
        if total >= max_total:
            break

    return "\n\n".join(sections_out) if sections_out else None


def fetch_arxiv_fulltext(arxiv_id: str, max_chars: int = 20000) -> str | None:
    """Fetch full text from arXiv HTML endpoint. Returns None if unavailable.

    Attempts section-aware extraction (methods, experiments, results, conclusion)
    to maximise signal density. Falls back to raw text if no sections detected.
    """
    url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        structured = _extract_key_sections(soup, max_total=8000)
        if structured:
            log.debug("fulltext_fetcher.structured", arxiv_id=arxiv_id, chars=len(structured))
            return structured

        # Fallback: raw body text from the start
        body = soup.find("body") or soup
        text = body.get_text(separator=" ", strip=True)
        log.debug("fulltext_fetcher.raw_fallback", arxiv_id=arxiv_id, chars=min(len(text), max_chars))
        return text[:max_chars] if text else None

    except Exception as exc:
        log.debug("fulltext_fetcher.error", arxiv_id=arxiv_id, error=str(exc))
        return None
