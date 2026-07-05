"""Orchestrates statistical analysis and baseline comparison."""
from __future__ import annotations

import json

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from analysis import statistical_analyzer, baseline_comparator
from config import settings
from core import token_tracker
from core.models import Experiment, ExperimentResult, RunState
from core.state import save_state
from knowledge.experiment_store import (
    get_experiments_by_status,
    get_result,
    save_result,
    save_experiment,
)
from analysis.ablation_manager import generate_ablations

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def analyze_result(exp: Experiment, result: ExperimentResult) -> dict:
    """Attach stats, baseline comparison, and conclusion to a result, then save it.

    Returns the baseline comparison dict. Shared by the pipeline and the UI.
    """
    metrics = json.loads(result.metrics) if result.metrics else {}

    if metrics:
        result.statistical_summary = json.dumps(statistical_analyzer.analyze(metrics))

    comparison = baseline_comparator.compare(result, exp.paper_id, experiment_title=exp.title)
    result.baseline_comparison = json.dumps(comparison)

    if result.statistical_summary:
        try:
            result.conclusion = _generate_conclusion(exp.title, exp.hypothesis, metrics, comparison)
        except Exception as e:
            log.error("analysis_pipeline.conclusion_error", error=str(e))

    save_result(result)
    return comparison


def run(state: RunState) -> None:
    """Attach statistical summaries and baseline comparisons to completed results."""
    completed = get_experiments_by_status("completed")
    log.info("analysis_pipeline.start", completed=len(completed))

    for exp in completed:
        result = get_result(exp.id)
        if result is None:
            continue

        # Skip if already analyzed
        if result.statistical_summary and result.baseline_comparison:
            continue

        comparison = analyze_result(exp, result)

        # Generate ablations for successful experiments (only for non-ablation parents)
        # Skip ablations when baseline comparison actively shows not_reproduced
        baseline_ok = comparison.get("overall") in (
            "fully_reproduced", "partially_reproduced", "no_experiments", "no_baselines"
        )
        if result.exit_code == 0 and exp.parent_experiment_id is None and baseline_ok:
            try:
                ablations = generate_ablations(exp, result)
                for abl in ablations:
                    save_experiment(abl)
                    state.experiment_ids_this_cycle.append(abl.id)
            except Exception as e:
                log.error("analysis_pipeline.ablation_error", exp_id=exp.id, error=str(e))

    save_state(state)

    # Eval-harness SP1: persist this cycle's reproduction-rate snapshot.
    # Failure here must not abort the pipeline (mirrors contradiction_detector pattern).
    try:
        from analysis import reproduction_metrics
        reproduction_metrics.record_cycle_snapshot(state)
    except Exception as e:
        log.error("analysis_pipeline.snapshot_error", error=str(e))

    # Eval-harness SP2: score the golden set against latest results (no compute).
    # Skipped when the golden set is empty; failure never aborts the cycle.
    try:
        from knowledge.benchmark_store import get_items
        if get_items(active_only=True):
            from analysis import benchmark_scorer
            benchmark_scorer.record_benchmark_run(cycle_id=state.cycle_id, trigger="cycle")
    except Exception as e:
        log.error("analysis_pipeline.benchmark_error", error=str(e))

    log.info("analysis_pipeline.done")


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    stop=stop_after_attempt(3),
)
def _generate_conclusion(title: str, hypothesis: str, metrics: dict, comparison: dict) -> str:
    """Generate a brief conclusion via Claude."""
    resp = _get_client().messages.create(
        model=settings.claude_haiku_model,
        max_tokens=512,
        temperature=0.3,
        messages=[{
            "role": "user",
            "content": f"""Experiment: {title}
Hypothesis: {hypothesis}
Metrics: {json.dumps(metrics, indent=2)[:500]}
Baseline comparison: {json.dumps(comparison)[:500]}

Write a 2-3 sentence scientific conclusion about what these results mean.
Be precise and honest about limitations."""
        }],
    )
    log.info(
        "claude.usage",
        module="analysis_conclusion",
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        cache_read_tokens=getattr(resp.usage, "cache_read_input_tokens", 0),
    )
    token_tracker.track("analysis_conclusion", resp.usage.input_tokens, resp.usage.output_tokens)
    return resp.content[0].text if resp.content else ""
