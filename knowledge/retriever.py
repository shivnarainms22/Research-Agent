"""Hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import structlog
from rank_bm25 import BM25Okapi

from core.models import Paper
from knowledge import paper_store, vector_store

log = structlog.get_logger()

_K = 60  # RRF constant


def _rrf_score(rank: int) -> float:
    return 1.0 / (_K + rank + 1)


def search(query: str, n: int = 10) -> list[Paper]:
    """Hybrid search: BM25 over SQLite title+abstract + vector search, fused via RRF."""
    # --- Vector results ---
    vec_results = vector_store.query_similar(query, n_results=n * 2)
    vec_ids = [r["id"] for r in vec_results]

    # --- BM25 over in-memory corpus ---
    all_papers = paper_store.get_all_papers(limit=5000)
    if not all_papers:
        return []

    corpus = [f"{p.title} {p.abstract}".lower().split() for p in all_papers]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())
    bm25_ranked = sorted(range(len(all_papers)), key=lambda i: scores[i], reverse=True)[: n * 2]
    bm25_ids = [all_papers[i].id for i in bm25_ranked]

    # --- RRF fusion ---
    rrf: dict[str, float] = {}
    for rank, pid in enumerate(vec_ids):
        rrf[pid] = rrf.get(pid, 0.0) + _rrf_score(rank)
    for rank, pid in enumerate(bm25_ids):
        rrf[pid] = rrf.get(pid, 0.0) + _rrf_score(rank)

    top_ids = sorted(rrf, key=lambda pid: rrf[pid], reverse=True)[:n]

    # Fetch full Paper objects
    paper_map = {p.id: p for p in all_papers}
    results = [paper_map[pid] for pid in top_ids if pid in paper_map]
    log.info("retriever.search", query=query[:50], hits=len(results))
    return results
