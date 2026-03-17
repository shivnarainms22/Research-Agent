"""Claude-powered experiment extraction and Experiment record creation."""
from __future__ import annotations

import json
import uuid
from datetime import datetime

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import Experiment, PaperAnalysis
from knowledge.experiment_store import get_experiments_by_paper_id, get_recent_failed_results
from knowledge import paper_store, retriever

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


_CODE_GEN_SYSTEM = """\
You are an expert AI researcher specializing in reproducing and validating ML experiments.
Your goal is to write a rigorous, faithful Python experiment that tests a specific claim from a paper.

Requirements:
- Implement the exact method described in the paper — not a simplified proxy
- Use the paper's stated hyperparameters wherever possible
- Use the stated datasets; if unavailable, use the closest public equivalent and note the substitution
- Treat the paper's claimed baseline metric as the target: log whether you meet, exceed, or fall short
- Be fully self-contained (all imports at top, no external files needed)
- Write all results to /workspace/results/metrics.json as a JSON dict
- Be executable in under 1 hour on the specified compute tier
- Use only standard ML libraries (torch, transformers, sklearn, numpy, scipy, pandas)
- Log progress at each major step with timing information
- Handle failures gracefully: catch exceptions, log them, write partial results if possible

Code style: be concise. Use helper functions to avoid repetition. Avoid verbose comments —
the code should be self-explanatory. Target under 150 lines of Python.
"""

_EXPERIMENT_CODE_TOOL = {
    "name": "generate_experiment",
    "description": "Generate a Python experiment to test a paper's claim",
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Short experiment title"},
            "hypothesis": {"type": "string", "description": "What we're testing"},
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
def extract_experiments(
    paper_id: str,
    analysis: PaperAnalysis,
    has_direct_contradiction: bool = False,
) -> list[Experiment]:
    """Generate Experiment records from a PaperAnalysis."""
    reproducible = json.loads(analysis.reproducible_experiments)
    if not reproducible:
        return []

    # Skip if experiments already exist for this paper (cross-source dedup)
    existing = get_experiments_by_paper_id(paper_id)
    active = [e for e in existing if e.status != "skipped"]
    if active:
        log.info(
            "experiment_extractor.skip_duplicate",
            paper_id=paper_id,
            existing_count=len(active),
            titles=[e.title for e in active],
        )
        return []

    # Only generate experiments for high-novelty papers; lower threshold for contradictions
    effective_threshold = settings.min_novelty_score_to_experiment
    if has_direct_contradiction:
        effective_threshold = max(0.0, effective_threshold - 1.0)

    if analysis.novelty_score < effective_threshold:
        log.info(
            "experiment_extractor.skip_low_novelty",
            paper_id=paper_id,
            score=analysis.novelty_score,
            threshold=effective_threshold,
            has_direct_contradiction=has_direct_contradiction,
        )
        return []

    client = _get_client()
    experiments: list[Experiment] = []

    # Build related paper context from vector store (uses already-embedded papers)
    related_context = ""
    try:
        query = " ".join(json.loads(analysis.key_contributions)[:2])
        related_papers = retriever.search(query, n=3)
        related_parts = []
        for rp in related_papers:
            if rp.id == paper_id:
                continue
            rp_analysis = paper_store.get_analysis(rp.id)
            if rp_analysis:
                related_parts.append(
                    f"- {rp.title}: {'; '.join(json.loads(rp_analysis.key_contributions)[:2])}"
                )
        if related_parts:
            related_context = "Related work in vector store:\n" + "\n".join(related_parts)
    except Exception:
        pass  # related context is best-effort

    # Build failure feedback context from recent failed experiments
    failure_context = ""
    try:
        failed_pairs = get_recent_failed_results(limit=5)
        if failed_pairs:
            lines = []
            for failed_exp, failed_result in failed_pairs:
                exit_note = f"exit_code={failed_result.exit_code}"
                metrics_note = "no metrics" if failed_result.metrics == "{}" else "empty metrics"
                lines.append(
                    f"- '{failed_exp.title}': {exit_note}, {metrics_note}"
                )
            failure_context = (
                "Recent failed experiments (avoid similar mistakes):\n"
                + "\n".join(lines)
            )
    except Exception:
        pass  # failure context is best-effort

    # Take up to 3 experiments per paper
    for exp_spec in reproducible[:3]:
        try:
            prompt = f"""Paper analysis:
Key contributions: {json.loads(analysis.key_contributions)}
Methods: {json.loads(analysis.methods_described)}
Datasets used: {json.loads(analysis.datasets_used)}
Key hyperparameters: {json.loads(analysis.key_hyperparameters)}
Limitations: {json.loads(analysis.limitations)}
Reproducibility difficulty: {analysis.reproducibility_difficulty}
{related_context}
{failure_context}

Experiment to implement:
Title: {exp_spec['title']}
Description: {exp_spec['description']}
Compute tier: {exp_spec.get('compute_requirement', 'cpu_only')}
Expected metric: {exp_spec.get('expected_metric', 'N/A')}
Baseline claimed by paper: {exp_spec.get('baseline_claimed', 'N/A')}

Generate a complete Python script that faithfully reproduces this experiment.
Use the exact hyperparameters and datasets above. If a dataset is unavailable, use the closest
public equivalent and log the substitution. Log whether your result meets the claimed baseline."""

            response = client.messages.create(
                model=settings.claude_model,
                max_tokens=16000,
                temperature=0.2,
                system=[{"type": "text", "text": _CODE_GEN_SYSTEM, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": prompt}],
                tools=[_EXPERIMENT_CODE_TOOL],
                tool_choice={"type": "tool", "name": "generate_experiment"},
            )

            log.info(
                "claude.usage",
                module="experiment_extractor",
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
            )
            token_tracker.track("experiment_extractor", response.usage.input_tokens, response.usage.output_tokens)

            tool_result = next(
                (b.input for b in response.content if b.type == "tool_use"),
                None,
            )
            if not tool_result:
                log.warning("experiment_extractor.no_tool_result", exp_spec=exp_spec.get("title"))
                continue
            if "python_code" not in tool_result:
                log.warning("experiment_extractor.truncated_response",
                            exp_spec=exp_spec.get("title"),
                            output_tokens=response.usage.output_tokens)
                continue

            # Determine execution target from compute requirement
            compute = exp_spec.get("compute_requirement", "cpu_only")
            target = tool_result.get("execution_target", "local")
            if compute == "gpu_large":
                target = "cloud_modal"

            exp = Experiment(
                id=str(uuid.uuid4()),
                paper_id=paper_id,
                title=tool_result["title"],
                hypothesis=tool_result["hypothesis"],
                generated_code=tool_result["python_code"],
                execution_target=target,
                status="pending_review",
                created_at=datetime.utcnow(),
                retry_count=0,
            )
            experiments.append(exp)
            log.info("experiment_extractor.created", title=exp.title, target=target)

        except Exception as e:
            log.error("experiment_extractor.error", exp_spec=exp_spec.get("title"), error=str(e))

    return experiments
