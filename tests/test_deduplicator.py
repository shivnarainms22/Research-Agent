"""Tests for ingestion/deduplicator.py"""
from __future__ import annotations

from datetime import date, datetime
from unittest.mock import patch

import pytest
from sqlmodel import Session

from core.models import Paper


def _make_paper(pid: str) -> Paper:
    return Paper(
        id=pid,
        title=f"Paper {pid}",
        abstract="An abstract.",
        source="arxiv",
        source_id=pid,
        url=f"https://arxiv.org/abs/{pid}",
        published_date=date(2024, 1, 1),
        fetched_at=datetime.utcnow(),
    )


def test_dedup_removes_existing_db_paper(in_memory_engine):
    """Papers already in the DB should be removed."""
    from ingestion.deduplicator import deduplicate

    paper = _make_paper("abc123")
    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(paper)
        session.commit()

    # Patch semantic dedup to be a no-op
    with patch("ingestion.deduplicator._semantic_dedup", side_effect=lambda p: p):
        result = deduplicate([_make_paper("abc123")])

    assert result == []


def test_dedup_removes_intra_batch_duplicate(in_memory_engine):
    """Two papers with the same ID in one batch — only one should survive."""
    from ingestion.deduplicator import deduplicate

    p1 = _make_paper("dup001")
    p2 = _make_paper("dup001")

    with patch("ingestion.deduplicator._semantic_dedup", side_effect=lambda p: p):
        result = deduplicate([p1, p2])

    assert len(result) == 1
    assert result[0].id == "dup001"


def test_dedup_allows_new_paper(in_memory_engine):
    """A paper with a fresh ID not in the DB should pass through."""
    from ingestion.deduplicator import deduplicate

    paper = _make_paper("brand_new_xyz")

    with patch("ingestion.deduplicator._semantic_dedup", side_effect=lambda p: p):
        result = deduplicate([paper])

    assert len(result) == 1
    assert result[0].id == "brand_new_xyz"


def test_semantic_dedup_skips_near_identical(in_memory_engine):
    """_semantic_dedup should drop a paper when query_similar returns distance < 0.05."""
    from ingestion.deduplicator import _semantic_dedup

    paper = _make_paper("near_dup")

    mock_results = [{"id": "existing_paper", "distance": 0.01, "title": "Similar paper"}]

    with (
        patch("ingestion.deduplicator._semantic_dedup",
              wraps=lambda papers: _call_real_semantic_dedup(papers, mock_results)),
    ):
        # Call the real function directly with a mocked query_similar
        with (
            patch("knowledge.vector_store.query_similar", return_value=mock_results),
            patch("core.database.get_chroma") as mock_chroma,
        ):
            mock_collection = mock_chroma.return_value.get_or_create_collection.return_value
            mock_collection.count.return_value = 5

            result = _semantic_dedup([paper])

    assert result == []


def _call_real_semantic_dedup(papers, mock_results):
    """Helper to invoke the real _semantic_dedup with mocked dependencies."""
    from unittest.mock import patch as _patch
    from ingestion.deduplicator import _semantic_dedup as _real

    with (
        _patch("knowledge.vector_store.query_similar", return_value=mock_results),
        _patch("core.database.get_chroma") as mc,
    ):
        mc.return_value.get_or_create_collection.return_value.count.return_value = 5
        return _real(papers)
