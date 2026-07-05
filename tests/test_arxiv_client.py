"""Tests for ingestion/arxiv_client.py query construction.

Bug guard: the query must include every configured keyword — a silent [:8]
slice previously dropped the probing and VLA keywords from arXiv searches.
"""
from __future__ import annotations

from config import settings
from ingestion.arxiv_client import _build_query


def test_build_query_includes_all_keywords():
    query = _build_query()
    for kw in settings.arxiv_keywords:
        assert f'"{kw}"' in query


def test_build_query_includes_all_categories():
    query = _build_query()
    for cat in settings.arxiv_categories:
        assert f"cat:{cat}" in query
