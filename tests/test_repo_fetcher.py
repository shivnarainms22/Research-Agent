"""Tests for ingestion/repo_fetcher.py."""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from core.models import Paper
from ingestion.repo_fetcher import find_github_repo, get_repo_context


def _make_paper(abstract: str, full_text: str | None = None, source_id: str = "2401.00001") -> Paper:
    return Paper(
        id="p1", title="T", abstract=abstract, source="arxiv", source_id=source_id,
        url="https://arxiv.org/abs/2401.00001", published_date=date(2024, 1, 1),
        full_text=full_text,
    )


def test_find_repo_from_abstract():
    paper = _make_paper("Code at https://github.com/openai/whisper for details.")
    assert find_github_repo(paper) == "openai/whisper"


def test_find_repo_strips_git_suffix_and_trailing_dot():
    paper = _make_paper("See https://github.com/foo/bar.git and enjoy.")
    assert find_github_repo(paper) == "foo/bar"
    paper2 = _make_paper("Available at https://github.com/foo/baz.")
    assert find_github_repo(paper2) == "foo/baz"


def test_find_repo_from_fulltext():
    paper = _make_paper("No link here.", full_text="Our code: https://github.com/lab/proj")
    assert find_github_repo(paper) == "lab/proj"


def test_find_repo_falls_back_to_papers_with_code():
    paper = _make_paper("No link here.", source_id="2401.00001v2")
    resp = MagicMock(status_code=200)
    resp.json.return_value = {
        "results": [
            {"url": "https://github.com/other/mirror", "is_official": False},
            {"url": "https://github.com/lab/official", "is_official": True},
        ]
    }
    with patch("ingestion.repo_fetcher.httpx.get", return_value=resp) as mock_get:
        assert find_github_repo(paper) == "lab/official"
    # version suffix stripped from the arXiv id in the API call
    assert "arxiv:2401.00001/" in mock_get.call_args[0][0]


def test_get_repo_context_empty_when_no_repo():
    paper = _make_paper("No link.", source_id="2401.00001")
    resp = MagicMock(status_code=404)
    with patch("ingestion.repo_fetcher.httpx.get", return_value=resp):
        assert get_repo_context(paper) == ""


def test_get_repo_context_includes_readme():
    paper = _make_paper("Code: https://github.com/lab/proj")
    resp = MagicMock(status_code=200, text="# Proj\nInstall with pip.")
    with patch("ingestion.repo_fetcher.httpx.get", return_value=resp):
        ctx = get_repo_context(paper)
    assert "https://github.com/lab/proj" in ctx
    assert "Install with pip" in ctx
