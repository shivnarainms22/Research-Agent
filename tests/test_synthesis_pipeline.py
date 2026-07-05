"""Integration tests for synthesis/synthesis_pipeline.py with mocked Claude."""
from __future__ import annotations

import json
from datetime import date, datetime
from unittest.mock import MagicMock, patch


from core.models import Paper, PaperAnalysis, RunState


def _make_paper(pid: str) -> Paper:
    return Paper(
        id=pid,
        title=f"Mechanistic interpretability sparse autoencoder {pid}",
        abstract="We study mechanistic interpretability and sparse autoencoders in transformers.",
        source="arxiv",
        source_id=pid,
        url=f"https://arxiv.org/abs/{pid}",
        published_date=date(2024, 1, 1),
        fetched_at=datetime.utcnow(),
        status="fetched",
    )


def _make_analysis(paper_id: str, novelty: float = 8.0, relevance: float = 8.0) -> PaperAnalysis:
    return PaperAnalysis(
        id=f"analysis_{paper_id}",
        paper_id=paper_id,
        key_contributions=json.dumps(["contribution 1"]),
        methods_described=json.dumps(["method 1"]),
        reproducible_experiments=json.dumps([]),
        novelty_score=novelty,
        relevance_score=relevance,
        datasets_used=json.dumps([]),
        key_hyperparameters=json.dumps({}),
        limitations=json.dumps([]),
        reproducibility_difficulty="easy",
    )


def test_synthesis_run_empty_paper_ids(in_memory_engine):
    """run() with no papers in DB should return empty list without error."""
    from synthesis.synthesis_pipeline import run

    state = RunState(
        cycle_id="test_cycle",
        started_at=datetime.utcnow(),
        paper_ids_this_cycle=[],
    )
    with patch("knowledge.paper_store.get_papers_by_status", return_value=[]):
        result = run(state)
    assert result == []


def test_synthesis_run_keyword_filter_skips_irrelevant(in_memory_engine):
    """Papers that don't match domain keywords should be skipped before analysis."""
    from synthesis.synthesis_pipeline import run

    state = RunState(
        cycle_id="test_cycle_2",
        started_at=datetime.utcnow(),
        paper_ids_this_cycle=["irrelevant_paper_001"],
    )

    irrelevant_paper = Paper(
        id="irrelevant_paper_001",
        title="Quantum Chemistry and Protein Folding",
        abstract="A study of quantum effects in protein folding simulations.",
        source="arxiv",
        source_id="irrelevant_paper_001",
        url="https://arxiv.org/abs/irrelevant",
        published_date=date(2024, 1, 1),
        fetched_at=datetime.utcnow(),
        status="fetched",
    )

    with (
        patch("knowledge.paper_store.get_paper", return_value=irrelevant_paper),
        patch("knowledge.paper_store.get_analysis", return_value=None),
        patch("knowledge.paper_store.update_paper_status") as mock_status,
        patch("core.state.save_state"),
    ):
        result = run(state)

    assert result == []
    mock_status.assert_called_once_with("irrelevant_paper_001", "done")


def test_synthesis_run_with_mock_analysis(in_memory_engine):
    """Papers passing keyword filter should go through analysis and be returned."""
    from synthesis.synthesis_pipeline import run

    paper = _make_paper("arxiv_001")
    analysis = _make_analysis("arxiv_001", novelty=8.5, relevance=8.0)

    state = RunState(
        cycle_id="test_cycle_3",
        started_at=datetime.utcnow(),
        paper_ids_this_cycle=["arxiv_001"],
    )

    with (
        patch("knowledge.paper_store.get_paper", return_value=paper),
        patch("knowledge.paper_store.get_analysis", return_value=None),
        patch("ingestion.fulltext_fetcher.fetch_arxiv_fulltext", return_value=None),
        patch("synthesis.paper_analyzer.analyze_paper", return_value=analysis),
        patch("knowledge.paper_store.save_analysis"),
        patch("knowledge.paper_store.update_paper_status"),
        patch("knowledge.contradiction_detector.check_new_paper"),
        patch("knowledge.contradiction_store.get_contradictions_for_paper", return_value=[]),
        patch("knowledge.vector_store.embed_paper"),
        patch("synthesis.experiment_extractor.extract_experiments", return_value=[]),
        patch("core.state.save_state"),
    ):
        result = run(state)

    assert result == []


def _fake_message(paper_id: str) -> MagicMock:
    """A Claude message whose tool_use block carries a minimal analysis."""
    block = MagicMock()
    block.type = "tool_use"
    block.input = {
        "key_contributions": ["c1"],
        "methods_described": ["m1"],
        "reproducible_experiments": [],
        "novelty_score": 8.0,
        "relevance_score": 8.0,
        "limitations": [],
        "datasets_used": [],
        "key_hyperparameters": {},
        "reproducibility_difficulty": "easy",
    }
    message = MagicMock()
    message.content = [block]
    message.usage.input_tokens = 100
    message.usage.output_tokens = 50
    return message


def test_request_params_omits_temperature():
    """Regression: Sonnet-5 rejects `temperature`; it must not be sent (batch or direct)."""
    from synthesis.paper_analyzer import _request_params

    params = _request_params(_make_paper("p1"))
    assert "temperature" not in params
    assert params["model"]  # model is still set


def test_analyze_papers_batch_parses_results():
    """Batch results map back to PaperAnalysis by paper id; errored items are skipped."""
    from synthesis.paper_analyzer import analyze_papers_batch

    papers = [_make_paper("p1"), _make_paper("p2")]

    ok = MagicMock()
    ok.custom_id = "p1"
    ok.result.type = "succeeded"
    ok.result.message = _fake_message("p1")

    errored = MagicMock()
    errored.custom_id = "p2"
    errored.result.type = "errored"

    batch = MagicMock()
    batch.id = "batch_1"
    batch.processing_status = "ended"

    client = MagicMock()
    client.messages.batches.create.return_value = batch
    client.messages.batches.results.return_value = [ok, errored]

    with patch("synthesis.paper_analyzer._get_client", return_value=client):
        analyses = analyze_papers_batch(papers)

    assert set(analyses) == {"p1"}
    assert analyses["p1"].novelty_score == 8.0
    assert analyses["p1"].paper_id == "p1"
