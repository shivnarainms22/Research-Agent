"""Orchestrates statistical analysis and baseline comparison."""
from __future__ import annotations

import json

import structlog

from analysis import statistical_analyzer, baseline_comparator
from core.models import RunState
from core.state import save_state
from knowledge.experiment_store import (
    get_experiments_by_status,
    get_result,
    save_result,
    save_experiment,
)
from analysis.ablation_manager import generate_ablations

log = structlog.get_logger()


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

        metrics = json.loads(result.metrics) if result.metrics else {}

        # Statistical summary
        if metrics:
            summary = statistical_analyzer.analyze(metrics)
            result.statistical_summary = json.dumps(summary)

        # Baseline comparison
        comparison = baseline_comparator.compare(result, exp.paper_id)
        result.baseline_comparison = json.dumps(comparison)

        # Claude-generated conclusion
        if result.statistical_summary:
            result.conclusion = _generate_conclusion(exp.title, exp.hypothesis, metrics, comparison)

        save_result(result)

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
    log.info("analysis_pipeline.done")


def _generate_conclusion(title: str, hypothesis: str, metrics: dict, comparison: dict) -> str:
    """Generate a brief conclusion via Claude."""
    import anthropic
    from config import settings

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        resp = client.messages.create(
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
        from core import token_tracker
        token_tracker.track("analysis_conclusion", resp.usage.input_tokens, resp.usage.output_tokens)
        return resp.content[0].text if resp.content else ""
    except Exception as e:
        log.error("analysis_pipeline.conclusion_error", error=str(e))
        return ""
