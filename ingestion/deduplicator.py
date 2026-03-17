"""Cross-source paper deduplication."""
from __future__ import annotations

import structlog
from sqlmodel import select

from core.database import get_session
from core.models import Paper

log = structlog.get_logger()


def _semantic_dedup(papers: list[Paper]) -> list[Paper]:
    """Second-pass dedup: drop papers with near-identical embeddings already in ChromaDB."""
    try:
        from knowledge.vector_store import query_similar
        from core.database import get_chroma
        chroma = get_chroma()
        count = chroma.get_or_create_collection("papers").count()
        if count == 0:
            return papers

        kept = []
        for paper in papers:
            query_text = f"{paper.title} {paper.abstract[:500] if paper.abstract else ''}"
            results = query_similar(query_text, n_results=1)
            if results and results[0].get("distance", 1.0) < 0.05:
                log.info(
                    "deduplicator.semantic_dedup_skip",
                    extra={
                        "paper_id": paper.id,
                        "similar_to": results[0].get("id"),
                        "distance": results[0].get("distance"),
                    },
                )
                continue
            kept.append(paper)
        return kept
    except Exception as e:
        log.debug(f"semantic_dedup failed gracefully: {e}")
        return papers


def deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove papers already in SQLite + deduplicate within the incoming batch."""
    with get_session() as session:
        existing_ids = set(session.exec(select(Paper.id)).all())

    seen_ids: set[str] = set()
    unique: list[Paper] = []
    dupes = 0

    for p in papers:
        if p.id in existing_ids or p.id in seen_ids:
            dupes += 1
            continue
        seen_ids.add(p.id)
        unique.append(p)

    log.info("deduplicator.done", input=len(papers), unique=len(unique), dupes=dupes)

    unique = _semantic_dedup(unique)
    return unique
