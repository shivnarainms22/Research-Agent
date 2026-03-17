"""Parse experiment stdout, metrics.json, and artifacts into ExperimentResult."""
from __future__ import annotations

import json
import re
from pathlib import Path

import structlog

from config import settings
from core.models import ExperimentResult

log = structlog.get_logger()


def collect(experiment_id: str) -> dict:
    """Read metrics.json and artifacts from the experiment workspace."""
    workspace = settings.experiments_dir / experiment_id / "results"
    if not workspace.exists():
        return {}

    metrics = {}
    metrics_path = workspace / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    artifacts = [str(p.relative_to(workspace.parent)) for p in workspace.glob("**/*") if p.is_file()]
    return {"metrics": metrics, "artifacts": artifacts}


def parse_metrics_from_stdout(stdout: str) -> dict:
    """Attempt to extract key=value pairs from stdout as a fallback."""
    metrics = {}
    pattern = re.compile(r"(\w+)\s*[:=]\s*([\d.]+)")
    for match in pattern.finditer(stdout):
        key, val = match.group(1), match.group(2)
        try:
            metrics[key] = float(val)
        except ValueError:
            pass
    return metrics
