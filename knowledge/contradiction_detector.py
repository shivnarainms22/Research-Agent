"""Detect contradictions between a newly analyzed paper and similar existing papers."""
from __future__ import annotations

import uuid

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import Contradiction, PaperAnalysis, parse_json_list
from knowledge import vector_store
from knowledge.contradiction_store import save_contradiction
from knowledge.paper_store import get_analysis

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    stop=stop_after_attempt(3),
)
def _call_claude(new_paper_summary: str, similar_summaries: list[dict]) -> list[dict]:
    client = _get_client()

    similar_text = "\n\n".join(
        f"Paper ID: {s['paper_id']}\nTitle: {s['title']}\nClaims: {s['claims']}"
        for s in similar_summaries
    )

    tool = {
        "name": "detect_contradictions",
        "description": "Identify claims in the new paper that contradict claims in existing papers",
        "input_schema": {
            "type": "object",
            "properties": {
                "contradictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "paper_id_old": {"type": "string"},
                            "metric": {"type": "string"},
                            "description": {"type": "string"},
                            "severity": {"type": "string", "enum": ["direct", "partial", "methodological"]},
                        },
                        "required": ["paper_id_old", "metric", "description", "severity"],
                    },
                }
            },
            "required": ["contradictions"],
        },
    }

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        temperature=0.2,
        system=[{
            "type": "text",
            "text": "You are a research analyst. Identify factual contradictions between research papers — cases where papers make opposing empirical claims about the same metric or benchmark. Ignore differences in methodology or scope unless they lead to directly conflicting conclusions.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": f"""New paper claims:
{new_paper_summary}

Existing similar papers:
{similar_text}

Identify any direct contradictions between the new paper and the existing ones.""",
        }],
        tools=[tool],
        tool_choice={"type": "tool", "name": "detect_contradictions"},
    )

    token_tracker.track("contradiction_detector", response.usage.input_tokens, response.usage.output_tokens)

    result = next((b.input for b in response.content if b.type == "tool_use"), None)
    return result.get("contradictions", []) if result else []


def _build_claims_summary(analysis: PaperAnalysis) -> str:
    contributions = parse_json_list(analysis.key_contributions)
    return "; ".join(contributions[:5]) if contributions else "(no claims extracted)"


def check_new_paper(paper_id: str, analysis: PaperAnalysis) -> None:
    """Check newly analyzed paper against similar existing papers for contradictions."""
    from knowledge.paper_store import get_paper
    paper = get_paper(paper_id)
    if paper is None:
        return

    # Find similar papers via vector search
    similar = vector_store.query_similar(paper.abstract, n_results=5)
    if not similar:
        return

    # Build summaries for similar papers (skip the paper itself)
    similar_summaries = []
    for s in similar:
        if s["id"] == paper_id:
            continue
        existing_analysis = get_analysis(s["id"])
        if existing_analysis is None:
            continue
        similar_summaries.append({
            "paper_id": s["id"],
            "title": s["title"],
            "claims": _build_claims_summary(existing_analysis),
        })

    if not similar_summaries:
        return

    new_summary = _build_claims_summary(analysis)

    try:
        contradictions = _call_claude(new_summary, similar_summaries)
    except Exception as e:
        log.error("contradiction_detector.claude_error", paper_id=paper_id, error=str(e))
        return

    for c in contradictions:
        contradiction = Contradiction(
            id=str(uuid.uuid4()),
            paper_id_new=paper_id,
            paper_id_old=c["paper_id_old"],
            metric=c["metric"],
            description=c["description"],
            severity=c.get("severity", "partial"),
        )
        try:
            save_contradiction(contradiction)
            log.info(
                "contradiction_detector.saved",
                paper_id_new=paper_id,
                paper_id_old=c["paper_id_old"],
                severity=c.get("severity"),
            )
        except Exception as e:
            log.error("contradiction_detector.save_error", error=str(e))
