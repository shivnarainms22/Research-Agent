"""Statistical analysis of experiment results using scipy + pingouin."""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import structlog

log = structlog.get_logger()


def analyze(metrics: dict) -> dict:
    """
    Given a metrics dict from an experiment, compute statistical summaries.
    Handles: single values, lists (for CI/t-test), and nested dicts.
    """
    summary = {}

    for key, value in metrics.items():
        if isinstance(value, list) and len(value) > 1:
            numeric = []
            for v in value:
                if v is None:
                    continue
                try:
                    numeric.append(float(v))
                except (TypeError, ValueError):
                    pass
            if len(numeric) < 2:
                continue
            arr = np.array(numeric)
            summary[key] = _describe(arr)
        elif isinstance(value, (int, float)):
            summary[key] = {"value": float(value)}
        elif isinstance(value, dict):
            summary[key] = value  # pass through nested dicts

    return summary


def _describe(arr: np.ndarray) -> dict:
    from scipy import stats

    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    sem = float(stats.sem(arr)) if n > 1 else 0.0
    ci = stats.t.interval(0.95, df=n - 1, loc=mean, scale=sem) if n > 1 else (mean, mean)

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_95_lower": float(ci[0]),
        "ci_95_upper": float(ci[1]),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def compare_groups(group_a: list[float], group_b: list[float]) -> dict:
    """Independent samples t-test + effect size (Cohen's d)."""
    from scipy import stats

    a = np.array(group_a)
    b = np.array(group_b)

    if len(a) < 2 or len(b) < 2:
        return {"error": "insufficient samples"}

    t_stat, p_val = stats.ttest_ind(a, b)
    pooled_std = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    cohens_d = float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else 0.0

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_val),
        "cohens_d": cohens_d,
        "significant_at_05": bool(p_val < 0.05),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
    }
