"""Orchestrates pending experiment execution."""
from __future__ import annotations

import structlog

from core.models import RunState
from core.state import save_state
from experiments import code_validator, local_runner, cloud_runner, router
from experiments.result_collector import parse_metrics_from_stdout
from knowledge.experiment_store import (
    get_experiments_by_status,
    get_result,
    update_experiment_status,
    increment_retry,
    save_result,
    delete_result,
)

log = structlog.get_logger()

_MAX_RETRIES = 3


def run(state: RunState) -> None:
    """Run all pending experiments."""
    pending = get_experiments_by_status("pending")
    log.info("experiment_pipeline.start", pending=len(pending))

    for exp in pending:
        if exp.retry_count >= _MAX_RETRIES:
            update_experiment_status(exp.id, "skipped", error="max retries exceeded")
            continue

        # Validate code
        validated_code, ok = code_validator.validate_with_retry(
            exp.generated_code, exp.paper_id
        )
        if not ok:
            update_experiment_status(exp.id, "skipped", error="code validation failed")
            log.warning("experiment_pipeline.skipped_invalid", exp_id=exp.id)
            continue

        exp.generated_code = validated_code

        # Determine execution target
        target = router.decide_target(exp)
        exp.execution_target = target

        update_experiment_status(exp.id, "running")
        log.info("experiment_pipeline.running", exp_id=exp.id, target=target)

        try:
            if target == "local":
                result = local_runner.run(exp)
            else:
                result = cloud_runner.run(exp)

            # Fallback: parse metrics from stdout if metrics.json empty
            if result.metrics == "{}" and result.stdout:
                fallback = parse_metrics_from_stdout(result.stdout)
                if fallback:
                    import json
                    result.metrics = json.dumps(fallback)
            result.stdout = ""  # free storage — metrics extracted, raw output not needed

            # Delete stale result from a previous failed run before saving
            if get_result(exp.id):
                delete_result(exp.id)
            save_result(result)

            no_metrics = result.metrics == "{}"
            if result.exit_code == 0 and not no_metrics:
                update_experiment_status(exp.id, "completed")
            elif result.exit_code == 0 and no_metrics:
                increment_retry(exp.id)
                update_experiment_status(exp.id, "failed", error="exit_code=0 but no metrics produced")
                log.warning("experiment_pipeline.no_metrics", exp_id=exp.id)
            else:
                increment_retry(exp.id)
                update_experiment_status(exp.id, "failed", error=f"exit_code={result.exit_code}")

        except Exception as e:
            log.error("experiment_pipeline.error", exp_id=exp.id, error=str(e))
            increment_retry(exp.id)
            update_experiment_status(exp.id, "failed", error=str(e))

    save_state(state)
    log.info("experiment_pipeline.done")
