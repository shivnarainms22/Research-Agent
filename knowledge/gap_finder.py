"""Identify research gaps from the recent paper corpus."""
from __future__ import annotations

import json
import uuid

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import Experiment, ResearchGap, parse_json_list
from knowledge.gap_store import save_gaps, clear_gaps_for_cycle
from knowledge.paper_store import get_all_papers, get_analysis

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


_GAP_CODE_GEN_SYSTEM = """\
You are an expert AI researcher. Given a research gap description, generate a concrete Python
experiment that would directly address this gap. The experiment should be:
- Self-contained and runnable
- Targeted specifically at the identified gap
- Feasible within the available compute budget
- Write all results to /workspace/results/metrics.json
"""

_GAP_CODE_GEN_TOOL = {
    "name": "generate_gap_experiment",
    "description": "Generate an experiment to address a research gap",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short experiment title"},
            "hypothesis": {"type": "string", "description": "What gap this addresses and how"},
            "python_code": {"type": "string", "description": "Complete Python script"},
            "execution_target": {
                "type": "string",
                "enum": ["local", "cloud_modal"],
                "description": "Where to run this experiment",
            },
        },
        "required": ["title", "hypothesis", "python_code", "execution_target"],
    },
}


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    stop=stop_after_attempt(3),
)
def _call_claude(corpus_summary: str) -> list[dict]:
    client = _get_client()

    tool = {
        "name": "identify_gaps",
        "description": "Identify research gaps — questions or directions not yet addressed by the corpus",
        "input_schema": {
            "type": "object",
            "properties": {
                "gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "supporting_paper_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["description", "supporting_paper_ids"],
                    },
                }
            },
            "required": ["gaps"],
        },
    }

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=2048,
        temperature=0.3,
        system=[{
            "type": "text",
            "text": "You are a research strategist. Based on a corpus of recent papers, identify important open questions and research gaps — things the community has not yet studied, tested, or resolved. Write gaps as plain-English questions or statements.",
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{
            "role": "user",
            "content": f"""Recent research corpus (30 most recent papers):

{corpus_summary}

Identify 5-10 important research gaps or open questions suggested by this body of work.""",
        }],
        tools=[tool],
        tool_choice={"type": "tool", "name": "identify_gaps"},
    )

    token_tracker.track("gap_finder", response.usage.input_tokens, response.usage.output_tokens)

    result = next((b.input for b in response.content if b.type == "tool_use"), None)
    return result.get("gaps", []) if result else []


def _suggest_experiments_for_gaps(gaps: list[ResearchGap]) -> list[Experiment]:
    """Generate experiment stubs for each gap using a lighter Claude model."""
    if not gaps:
        return []

    model = getattr(settings, "claude_haiku_model", None) or settings.claude_model
    client = _get_client()
    experiments: list[Experiment] = []

    for gap in gaps:
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0.3,
                system=[{
                    "type": "text",
                    "text": _GAP_CODE_GEN_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Research gap identified:\n{gap.description}\n\n"
                        "Generate a concrete Python experiment to address this gap."
                    ),
                }],
                tools=[_GAP_CODE_GEN_TOOL],
                tool_choice={"type": "tool", "name": "generate_gap_experiment"},
            )

            token_tracker.track("gap_finder_codegen", response.usage.input_tokens, response.usage.output_tokens)

            tool_result = next((b.input for b in response.content if b.type == "tool_use"), None)
            if not tool_result or "python_code" not in tool_result:
                continue

            exp = Experiment(
                id=str(uuid.uuid4()),
                paper_id=gap.id,  # use gap ID as the paper_id reference
                title=tool_result["title"],
                hypothesis=tool_result["hypothesis"],
                generated_code=tool_result["python_code"],
                execution_target=tool_result.get("execution_target", "local"),
                status="pending_review",
            )
            experiments.append(exp)
            log.info("gap_finder.experiment_suggested", title=exp.title, gap=gap.description[:60])

        except Exception as e:
            log.warning("gap_finder.suggest_failed", gap=gap.description[:60], error=str(e))

    return experiments


def find_gaps(cycle_id: str) -> list[ResearchGap]:
    """Find research gaps from the 30 most recently analyzed papers."""
    papers = get_all_papers(limit=10000)
    # Get papers with analyses, most recent first
    analyzed = []
    for paper in reversed(papers):
        analysis = get_analysis(paper.id)
        if analysis:
            analyzed.append((paper, analysis))
        if len(analyzed) >= 30:
            break

    if len(analyzed) < 3:
        log.info("gap_finder.insufficient_papers", count=len(analyzed))
        return []

    corpus_lines = []
    paper_id_map = {}
    for paper, analysis in analyzed:
        contributions = parse_json_list(analysis.key_contributions)
        limitations = parse_json_list(analysis.limitations)
        line = (
            f"[{paper.id[:8]}] {paper.title}"
            + (f" | contributions: {'; '.join(contributions[:2])}" if contributions else "")
            + (f" | limitations: {limitations[0]}" if limitations else "")
        )
        corpus_lines.append(line)
        paper_id_map[paper.id[:8]] = paper.id

    corpus_summary = "\n".join(corpus_lines)

    try:
        raw_gaps = _call_claude(corpus_summary)
    except Exception as e:
        log.error("gap_finder.claude_error", error=str(e))
        return []

    # Clear old gaps for this cycle before saving new ones
    clear_gaps_for_cycle(cycle_id)

    gaps = []
    for g in raw_gaps:
        # Resolve short IDs back to full IDs where possible
        full_ids = []
        for short_id in g.get("supporting_paper_ids", []):
            full_id = paper_id_map.get(short_id[:8], short_id)
            full_ids.append(full_id)

        gap = ResearchGap(
            id=str(uuid.uuid4()),
            description=g["description"],
            supporting_paper_ids=json.dumps(full_ids),
            cycle_id=cycle_id,
        )
        gaps.append(gap)

    save_gaps(gaps)
    log.info("gap_finder.complete", count=len(gaps), cycle_id=cycle_id)

    # Generate experiment suggestions for the discovered gaps
    try:
        _suggest_experiments_for_gaps(gaps)
    except Exception as e:
        log.warning("gap_finder.suggest_experiments_failed", error=str(e))

    return gaps
