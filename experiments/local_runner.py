"""Run experiments in a Docker sandbox."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import structlog

from config import settings
from core.models import Experiment, ExperimentResult

log = structlog.get_logger()


def _prepare_workspace(exp: Experiment) -> Path:
    """Write experiment code to workspace dir, return path."""
    workspace = settings.experiments_dir / exp.id
    results_dir = workspace / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (workspace / "run.py").write_text(exp.generated_code, encoding="utf-8")
    return workspace


def run(exp: Experiment) -> ExperimentResult:
    """Execute experiment in Docker sandbox, return result."""
    import docker

    workspace = _prepare_workspace(exp)
    start_time = time.time()

    try:
        client = docker.from_env()

        container = client.containers.run(
            image="research-sandbox:latest",
            command=["python", "/workspace/run.py"],
            volumes={str(workspace): {"bind": "/workspace", "mode": "rw"}},
            mem_limit=settings.docker_memory_limit,
            nano_cpus=int(settings.docker_cpu_limit * 1e9),
            network_disabled=not settings.enable_experiment_network,
            user="nobody",
            remove=False,
            detach=True,
            stdout=True,
            stderr=True,
        )

        # Wait with timeout
        exit_code = None
        try:
            result = container.wait(timeout=settings.experiment_timeout_seconds)
            exit_code = result.get("StatusCode", 1)
        except Exception:
            container.kill()
            exit_code = 124  # timeout

        stdout = container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
        container.remove(force=True)

    except Exception as e:
        log.error("local_runner.docker_error", exp_id=exp.id, error=str(e))
        runtime = time.time() - start_time
        return ExperimentResult(
            id=f"result_{exp.id}",
            experiment_id=exp.id,
            stdout=f"Docker error: {e}",
            exit_code=1,
            metrics="{}",
            artifacts="[]",
            runtime_seconds=runtime,
            recorded_at=datetime.utcnow(),
        )

    runtime = time.time() - start_time

    # Parse metrics.json if present
    metrics_path = workspace / "results" / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Collect artifacts as paths relative to the workspace
    results_root = workspace / "results"
    artifacts = [
        str(p.relative_to(workspace)) for p in results_root.glob("**/*") if p.is_file()
    ]

    log.info("local_runner.complete", exp_id=exp.id, exit_code=exit_code, runtime=runtime)

    return ExperimentResult(
        id=f"result_{exp.id}",
        experiment_id=exp.id,
        stdout=stdout[:50000],  # cap at 50KB
        exit_code=exit_code,
        metrics=json.dumps(metrics),
        artifacts=json.dumps(artifacts),
        runtime_seconds=runtime,
        recorded_at=datetime.utcnow(),
    )
