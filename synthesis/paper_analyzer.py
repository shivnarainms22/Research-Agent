"""Claude-powered paper analysis: extracts contributions, methods, reproducible experiments."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import Paper, PaperAnalysis

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


_SYSTEM = """\
You are a rigorous AI research analyst specializing in interpretability, computer vision, and vision-language-action models.
Analyze the provided paper and extract structured information.
Be precise and conservative: only report what is clearly stated in the paper.
"""

_ANALYSIS_TOOL = {
    "name": "analyze_paper",
    "description": "Extract structured analysis from an AI research paper",
    "input_schema": {
        "type": "object",
        "properties": {
            "key_contributions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of 3-7 key contributions of the paper",
            },
            "methods_described": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Technical methods/architectures described",
            },
            "reproducible_experiments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "compute_requirement": {"type": "string", "enum": ["cpu_only", "gpu_small", "gpu_large"]},
                        "expected_metric": {"type": "string"},
                        "baseline_claimed": {
                            "type": "object",
                            "properties": {
                                "metric_name": {"type": "string"},
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                            },
                            "required": ["metric_name", "value"],
                        },
                    },
                    "required": ["title", "description", "compute_requirement"],
                },
                "description": "Experiments that could be reproduced computationally",
            },
            "novelty_score": {
                "type": "number",
                "description": "Novelty score 1-10 (10 = breakthrough)",
            },
            "relevance_score": {
                "type": "number",
                "description": "Relevance to interpretability/CV/VLA score 1-10",
            },
            "limitations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Limitations or caveats explicitly stated or implied by the paper",
            },
            "datasets_used": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of datasets used in experiments",
            },
            "key_hyperparameters": {
                "type": "object",
                "description": "Important hyperparameters stated in the paper (e.g. learning_rate, batch_size, layers, hidden_dim)",
                "additionalProperties": {"type": "string"},
            },
            "reproducibility_difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"],
                "description": "How hard is it to reproduce: easy=standard datasets+code, medium=custom setup, hard=proprietary data or massive compute",
            },
        },
        "required": ["key_contributions", "methods_described", "reproducible_experiments", "novelty_score", "relevance_score", "limitations", "datasets_used", "key_hyperparameters", "reproducibility_difficulty"],
    },
}


def _request_params(paper: Paper) -> dict:
    """Messages API params for analyzing one paper (shared by direct and batch calls)."""
    content = f"# {paper.title}\n\n## Abstract\n{paper.abstract}"
    if paper.full_text:
        content += f"\n\n## Key Sections\n{paper.full_text[:30000]}"
    return {
        "model": settings.claude_model,
        "max_tokens": 4096,
        "system": [
            {
                "type": "text",
                "text": _SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": content}],
        "tools": [_ANALYSIS_TOOL],
        "tool_choice": {"type": "tool", "name": "analyze_paper"},
    }


def _to_analysis(paper_id: str, message) -> PaperAnalysis:
    """Build a PaperAnalysis from a Claude message (raises if no tool_use block)."""
    tool_result = next(
        (b.input for b in message.content if b.type == "tool_use"),
        None,
    )
    if not tool_result:
        raise ValueError("No tool_use block in Claude response")

    analysis_id = hashlib.sha256(f"analysis:{paper_id}".encode()).hexdigest()[:32]

    return PaperAnalysis(
        id=analysis_id,
        paper_id=paper_id,
        key_contributions=json.dumps(tool_result.get("key_contributions", [])),
        methods_described=json.dumps(tool_result.get("methods_described", [])),
        reproducible_experiments=json.dumps(tool_result.get("reproducible_experiments", [])),
        novelty_score=float(tool_result.get("novelty_score", 5.0)),
        relevance_score=float(tool_result.get("relevance_score", 5.0)),
        limitations=json.dumps(tool_result.get("limitations", [])),
        datasets_used=json.dumps(tool_result.get("datasets_used", [])),
        key_hyperparameters=json.dumps(tool_result.get("key_hyperparameters", {})),
        reproducibility_difficulty=tool_result.get("reproducibility_difficulty", "medium"),
        raw_claude_response="",
        analyzed_at=datetime.utcnow(),
    )


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    stop=stop_after_attempt(3),
)
def analyze_paper(paper: Paper) -> PaperAnalysis:
    response = _get_client().messages.create(**_request_params(paper))

    log.info(
        "claude.usage",
        module="paper_analyzer",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
    )
    token_tracker.track("paper_analyzer", response.usage.input_tokens, response.usage.output_tokens)

    return _to_analysis(paper.id, response)


_BATCH_POLL_SECONDS = 15
_BATCH_TIMEOUT_SECONDS = 2 * 3600


def analyze_papers_batch(papers: list[Paper]) -> dict[str, PaperAnalysis]:
    """Analyze papers via the Message Batches API (50% cheaper than direct calls).

    Returns {paper_id: PaperAnalysis}; papers whose request errored are omitted
    and stay in 'fetched' status for the next cycle.
    """
    import time

    client = _get_client()
    batch = client.messages.batches.create(
        requests=[{"custom_id": p.id, "params": _request_params(p)} for p in papers]
    )
    log.info("paper_analyzer.batch_submitted", batch_id=batch.id, papers=len(papers))

    deadline = time.time() + _BATCH_TIMEOUT_SECONDS
    while batch.processing_status != "ended":
        if time.time() > deadline:
            raise TimeoutError(f"Batch {batch.id} not finished after {_BATCH_TIMEOUT_SECONDS}s")
        time.sleep(_BATCH_POLL_SECONDS)
        batch = client.messages.batches.retrieve(batch.id)

    analyses: dict[str, PaperAnalysis] = {}
    for entry in client.messages.batches.results(batch.id):
        if entry.result.type != "succeeded":
            log.error("paper_analyzer.batch_item_failed",
                      paper_id=entry.custom_id, result_type=entry.result.type)
            continue
        message = entry.result.message
        token_tracker.track("paper_analyzer", message.usage.input_tokens, message.usage.output_tokens)
        try:
            analyses[entry.custom_id] = _to_analysis(entry.custom_id, message)
        except Exception as e:
            log.error("paper_analyzer.batch_parse_failed", paper_id=entry.custom_id, error=str(e))

    log.info("paper_analyzer.batch_done", batch_id=batch.id,
             succeeded=len(analyses), failed=len(papers) - len(analyses))
    return analyses
