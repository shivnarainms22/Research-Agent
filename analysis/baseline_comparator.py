"""Compare experiment results vs paper-claimed baselines."""
from __future__ import annotations

import json
import re

import structlog

from core.models import ExperimentResult
from knowledge.paper_store import get_analysis

log = structlog.get_logger()

# Generic connectors only — never metric words like "accuracy"/"rate", which are
# frequently the actual metric name and must remain matchable.
_STOPWORDS = {
    "the", "a", "an", "of", "vs", "versus", "and", "for", "per", "on", "in", "to",
    "with", "using", "under", "over",
}

# Metrics keys whose boolean value is the experiment's own verdict on the paper's claim.
_VERDICT_KEYS = (
    "claim_verified", "claim_supported", "claim_met", "hypothesis_supported",
    "reproduced", "paper_claim_verified",
)


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _STOPWORDS}


def _matches(term: str, key: str) -> bool:
    """A measured key matches a claimed metric name if they share a meaningful token."""
    tt, kt = _tokens(term), _tokens(key)
    return bool(tt and kt and (tt & kt))


def _self_reported_verdict(metrics: dict) -> str | None:
    """Return an overall verdict if the experiment reported a claim-verification boolean."""
    for key in _VERDICT_KEYS:
        if isinstance(metrics.get(key), bool):
            return "fully_reproduced" if metrics[key] else "not_reproduced"
    return None


def _relevant_specs(exps: list[dict], experiment_title: str) -> list[dict]:
    """Scope to the paper's claimed experiment matching this run; all specs if unclear."""
    if not experiment_title:
        return exps
    title_tokens = _tokens(experiment_title)
    if not title_tokens:
        return exps
    best, best_overlap = None, 0
    for spec in exps:
        overlap = len(title_tokens & _tokens(spec.get("title", "")))
        if overlap > best_overlap:
            best, best_overlap = spec, overlap
    return [best] if best is not None else exps


def _claimed_value(baseline_raw) -> float | None:
    """Extract a numeric claimed value from dict (new) or string (legacy) format."""
    if isinstance(baseline_raw, dict):
        v = baseline_raw.get("value")
        return float(v) if v is not None else None
    if isinstance(baseline_raw, str) and baseline_raw:
        numbers = re.findall(r"[\d.]+", baseline_raw)
        if numbers:
            return float(numbers[0])
    return None


def _measured_value(raw) -> float | None:
    if isinstance(raw, dict):
        raw = raw.get("mean", raw.get("value"))
    elif isinstance(raw, list):
        valid = [float(v) for v in raw if isinstance(v, (int, float))]
        return sum(valid) / len(valid) if valid else None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def compare(result: ExperimentResult, paper_id: str, experiment_title: str = "") -> dict:
    """Compare result metrics against claimed baselines from PaperAnalysis.

    Scoped to the running experiment (via experiment_title). Honors an explicit
    self-reported verification boolean before falling back to value comparison.
    """
    analysis = get_analysis(paper_id)
    if not analysis:
        return {"status": "no_analysis"}

    exps = json.loads(analysis.reproducible_experiments)
    if not exps:
        return {"status": "no_baselines"}

    metrics = json.loads(result.metrics)
    if not metrics:
        return {"status": "no_metrics"}

    # Tier 1: the experiment explicitly reported whether it met the paper's claim.
    verdict = _self_reported_verdict(metrics)
    if verdict is not None:
        return {"overall": verdict, "comparisons": [], "source": "self_report"}

    # Tier 2: numeric comparison, scoped to the spec(s) this experiment ran.
    comparisons = []
    for exp_spec in _relevant_specs(exps, experiment_title):
        expected_metric = exp_spec.get("expected_metric", "")
        if not expected_metric:
            continue

        claimed_value = _claimed_value(exp_spec.get("baseline_claimed"))
        if claimed_value is None:
            continue

        search_terms = [expected_metric]
        baseline_raw = exp_spec.get("baseline_claimed")
        if isinstance(baseline_raw, dict) and baseline_raw.get("metric_name"):
            search_terms.insert(0, baseline_raw["metric_name"])

        matched_key = next(
            (k for term in search_terms for k in metrics if _matches(term, k)), None
        )
        if matched_key is None:
            comparisons.append({
                "experiment": exp_spec.get("title", ""),
                "metric": expected_metric,
                "claimed": claimed_value,
                "actual": None,
                "status": "metric_not_found",
            })
            continue

        actual = _measured_value(metrics[matched_key])
        if actual is None:
            status = "no_actual_value"
        else:
            pct_diff = abs(actual - claimed_value) / max(abs(claimed_value), 1e-8)
            status = "reproduced" if pct_diff <= 0.05 else "partial" if pct_diff <= 0.15 else "failed"

        comparisons.append({
            "experiment": exp_spec.get("title", ""),
            "metric": expected_metric,
            "claimed": claimed_value,
            "actual": actual,
            "status": status,
        })

    overall = "no_experiments"
    if comparisons:
        statuses = [c["status"] for c in comparisons]
        if all(s == "reproduced" for s in statuses):
            overall = "fully_reproduced"
        elif any(s == "reproduced" for s in statuses):
            overall = "partially_reproduced"
        else:
            overall = "not_reproduced"

    return {"overall": overall, "comparisons": comparisons}
