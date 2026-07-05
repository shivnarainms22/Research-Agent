"""Tests for ui/views/ask.py context gathering."""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import patch

from core.models import Paper, PaperAnalysis


def _paper(pid: str, title: str) -> Paper:
    return Paper(
        id=pid, title=title, abstract="a", source="arxiv", source_id=pid,
        url=f"https://arxiv.org/abs/{pid}", published_date=date(2024, 1, 1),
    )


def test_gather_context_numbers_sources_and_includes_contributions():
    from ui.views.ask import _gather_context

    papers = [_paper("p1", "Steering Vectors"), _paper("p2", "SAE Features")]
    analysis = PaperAnalysis(
        id="a1", paper_id="p1",
        key_contributions=json.dumps(["contribution A", "contribution B"]),
    )
    with (
        patch("ui.views.ask.retriever.search", return_value=papers),
        patch("ui.views.ask.paper_store.get_analysis", side_effect=[analysis, None]),
    ):
        context, sources = _gather_context("steering", n=2)

    assert "[1] Steering Vectors" in context
    assert "contribution A" in context
    assert sources[0] == {"n": 1, "title": "Steering Vectors", "url": "https://arxiv.org/abs/p1"}
    assert sources[1]["n"] == 2


def test_answer_question_short_circuits_when_no_papers():
    from ui.views.ask import answer_question

    with patch("ui.views.ask.retriever.search", return_value=[]):
        answer, sources = answer_question("anything")
    assert sources == []
    assert "No relevant papers" in answer
