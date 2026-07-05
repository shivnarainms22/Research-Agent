"""Hybrid BM25 + vector search with Reciprocal Rank Fusion (RRF)."""
from __future__ import annotations

import threading

import structlog
from rank_bm25 import BM25Okapi

from core.models import Paper
from knowledge import paper_store, vector_store

log = structlog.get_logger()

_K = 60  # RRF constant

# BM25 corpus cache — rebuilt only when the paper count changes (papers are
# append-only), instead of re-tokenizing the whole corpus on every search.
_corpus_lock = threading.Lock()
_corpus_cache: tuple[int, list[Paper], BM25Okapi] | None = None


def _get_corpus() -> tuple[list[Paper], BM25Okapi | None]:
    global _corpus_cache
    count = paper_store.count_papers()
    if count == 0:
        return [], None
    with _corpus_lock:
        if _corpus_cache is None or _corpus_cache[0] != count:
            papers = paper_store.get_all_papers(limit=5000)
            tokens = [f"{p.title} {p.abstract}".lower().split() for p in papers]
            _corpus_cache = (count, papers, BM25Okapi(tokens))
        return _corpus_cache[1], _corpus_cache[2]


def _rrf_score(rank: int) -> float:
    return 1.0 / (_K + rank + 1)


def search(query: str, n: int = 10) -> list[Paper]:
    """Hybrid search: BM25 over SQLite title+abstract + vector search, fused via RRF."""
    # --- Vector results ---
    vec_results = vector_store.query_similar(query, n_results=n * 2)
    vec_ids = [r["id"] for r in vec_results]

    # --- BM25 over in-memory corpus (cached) ---
    all_papers, bm25 = _get_corpus()
    if not all_papers:
        return []

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
