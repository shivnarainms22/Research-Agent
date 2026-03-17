"""Compare experiment results vs paper-claimed baselines."""
from __future__ import annotations

import json
import re

import structlog

from core.models import ExperimentResult
from knowledge.paper_store import get_analysis

log = structlog.get_logger()


def compare(result: ExperimentResult, paper_id: str) -> dict:
    """
    Compare result metrics against claimed baselines from PaperAnalysis.
    Returns comparison dict with reproduced/failed/partial status per metric.
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

    comparisons = []
    for exp_spec in exps:
        baseline_raw = exp_spec.get("baseline_claimed")
        expected_metric = exp_spec.get("expected_metric", "")
        if not expected_metric:
            continue

        # Extract claimed value — handle both dict (new) and string (old) formats
        claimed_value = None
        if isinstance(baseline_raw, dict):
            claimed_value = float(baseline_raw["value"]) if baseline_raw.get("value") is not None else None
        elif isinstance(baseline_raw, str) and baseline_raw:
            numbers = re.findall(r"[\d.]+", baseline_raw)
            if numbers:
                claimed_value = float(numbers[0])

        if claimed_value is None:
            continue

        # Find matching metric — prefer metric_name from dict, fall back to expected_metric
        search_terms = [expected_metric]
        if isinstance(baseline_raw, dict) and baseline_raw.get("metric_name"):
            search_terms.insert(0, baseline_raw["metric_name"])

        matched_key = None
        for term in search_terms:
            for k in metrics:
                if term.lower() in k.lower() or k.lower() in term.lower():
                    matched_key = k
                    break
            if matched_key:
                break

        if matched_key is None:
            comparisons.append({
                "experiment": exp_spec.get("title", ""),
                "metric": expected_metric,
                "claimed": claimed_value,
                "actual": None,
                "status": "metric_not_found",
            })
            continue

        actual_raw = metrics[matched_key]
        if isinstance(actual_raw, dict):
            actual = actual_raw.get("mean", actual_raw.get("value"))
        elif isinstance(actual_raw, list):
            valid = [float(v) for v in actual_raw if v is not None]
            actual = float(sum(valid) / len(valid)) if valid else None
        else:
            actual = float(actual_raw)

        if actual is None:
            status = "no_actual_value"
        else:
            pct_diff = abs(actual - claimed_value) / max(abs(claimed_value), 1e-8)
            if pct_diff <= 0.05:
                status = "reproduced"
            elif pct_diff <= 0.15:
                status = "partial"
            else:
                status = "failed"

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
