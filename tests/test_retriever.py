"""Tests for knowledge/retriever.py BM25 corpus caching."""
from __future__ import annotations

from datetime import date
from unittest.mock import patch

from sqlmodel import Session

from core.models import Paper


def _make_paper(pid: str, title: str) -> Paper:
    return Paper(
        id=pid, title=title, abstract="sparse autoencoders in transformers",
        source="arxiv", source_id=pid, url=f"https://arxiv.org/abs/{pid}",
        published_date=date(2024, 1, 1), status="analyzed",
    )


def test_search_uses_cached_corpus_until_count_changes(in_memory_engine, monkeypatch):
    import knowledge.retriever as retriever

    monkeypatch.setattr(retriever, "_corpus_cache", None)

    with Session(in_memory_engine) as session:
        session.add(_make_paper("p1", "Sparse autoencoder interpretability"))
        session.commit()

    with (
        patch("knowledge.vector_store.query_similar", return_value=[]),
        patch("knowledge.paper_store.get_all_papers", wraps=retriever.paper_store.get_all_papers) as mock_load,
    ):
        retriever.search("sparse autoencoder")
        retriever.search("sparse autoencoder")
        assert mock_load.call_count == 1  # second search served from cache

        # A new paper changes the count and invalidates the cache
        with Session(in_memory_engine) as session:
            session.add(_make_paper("p2", "Steering vectors in LLMs"))
            session.commit()
        results = retriever.search("steering vectors")
        assert mock_load.call_count == 2
        assert any(p.id == "p2" for p in results)


def test_search_empty_corpus_returns_empty(in_memory_engine, monkeypatch):
    import knowledge.retriever as retriever

    monkeypatch.setattr(retriever, "_corpus_cache", None)
    with patch("knowledge.vector_store.query_similar", return_value=[]):
        assert retriever.search("anything") == []
