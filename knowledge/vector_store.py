"""ChromaDB interface for paper embeddings."""
from __future__ import annotations

import structlog

from core.database import get_chroma
from core.models import Paper

log = structlog.get_logger()

_COLLECTION_NAME = "papers"


def get_collection():
    client = get_chroma()
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_paper(paper: Paper) -> None:
    """Add or update a paper's embedding in ChromaDB."""
    collection = get_collection()
    doc = f"{paper.title}\n\n{paper.abstract}"
    try:
        collection.upsert(
            ids=[paper.id],
            documents=[doc],
            metadatas=[{
                "title": paper.title[:500],
                "source": paper.source,
                "published_date": str(paper.published_date),
            }],
        )
    except Exception as e:
        log.error("vector_store.embed_error", paper_id=paper.id, error=str(e))


def embed_papers(papers: list[Paper]) -> None:
    if not papers:
        return
    collection = get_collection()
    ids = [p.id for p in papers]
    docs = [f"{p.title}\n\n{p.abstract}" for p in papers]
    metas = [{
        "title": p.title[:500],
        "source": p.source,
        "published_date": str(p.published_date),
    } for p in papers]
    try:
        collection.upsert(ids=ids, documents=docs, metadatas=metas)
        log.info("vector_store.embedded", count=len(papers))
    except Exception as e:
        log.error("vector_store.embed_batch_error", error=str(e))


def query_similar(text: str, n_results: int = 10) -> list[dict]:
    """Return list of {id, title, distance} dicts."""
    collection = get_collection()
    try:
        results = collection.query(query_texts=[text], n_results=n_results)
        ids = results["ids"][0] if results["ids"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []
        return [
            {"id": i, "title": m.get("title", ""), "distance": d}
            for i, m, d in zip(ids, metas, distances)
        ]
    except Exception as e:
        log.error("vector_store.query_error", error=str(e))
        return []


def count() -> int:
    try:
        return get_collection().count()
    except Exception:
        return 0
