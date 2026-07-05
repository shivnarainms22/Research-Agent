"""Orchestrates pending experiment execution."""
from __future__ import annotations

import structlog

from core.models import Experiment, RunState
from experiments import (
    code_repairer, code_validator, experiment_critic, local_runner, cloud_runner, router,
)
from experiments.result_collector import parse_metrics_from_stdout
from knowledge.experiment_store import (
    get_experiments_by_status,
    get_result,
    update_experiment_code,
    update_experiment_status,
    increment_retry,
    save_result,
    delete_result,
)

log = structlog.get_logger()

_MAX_RETRIES = 3
_STDOUT_TAIL_CHARS = 5000


def _repair_before_run(exp: Experiment, reason: str) -> bool:
    """Repair a critic-flagged experiment and requeue it. Returns True if requeued."""
    try:
        repaired = code_repairer.repair(exp, f"Pre-run critic flagged this script: {reason}")
    except Exception as e:
        log.error("experiment_pipeline.critic_repair_error", exp_id=exp.id, error=str(e))
        return False
    if repaired is None:
        return False
    fixed_code, diagnosis = repaired
    update_experiment_code(exp.id, fixed_code)
    increment_retry(exp.id)
    update_experiment_status(exp.id, "pending", error=f"critic: {reason[:200]}; repaired: {diagnosis[:200]}")
    log.info("experiment_pipeline.critic_requeued", exp_id=exp.id, reason=reason[:120])
    return True


def _repair_or_fail(exp: Experiment, failure_context: str, error: str) -> None:
    """Repair the code and requeue as pending, or mark failed when out of retries.

    Called after increment_retry, so exp.retry_count is one behind the DB value.
    """
    if exp.retry_count + 1 >= _MAX_RETRIES:
        update_experiment_status(exp.id, "failed", error=error)
        return

    try:
        repaired = code_repairer.repair(exp, failure_context)
    except Exception as e:
        log.error("experiment_pipeline.repair_error", exp_id=exp.id, error=str(e))
        repaired = None

    if repaired is None:
        update_experiment_status(exp.id, "failed", error=error)
        return

    fixed_code, diagnosis = repaired
    update_experiment_code(exp.id, fixed_code)
    update_experiment_status(exp.id, "pending", error=f"{error}; repaired: {diagnosis[:300]}")
    log.info("experiment_pipeline.repaired_requeued", exp_id=exp.id, diagnosis=diagnosis[:120])


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

        # Pre-run critic: if the script won't faithfully test the claim, repair it
        # before spending compute. Advisory — runs anyway if repair is unavailable
        # or the retry budget is exhausted.
        if exp.retry_count < _MAX_RETRIES - 1:
            verdict, reason = experiment_critic.review(exp)
            if verdict == "flawed" and _repair_before_run(exp, reason):
                continue

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

            # Clear stdout on success; keep the tail on failure — it's the repair signal
            success = result.exit_code == 0 and result.metrics != "{}"
            result.stdout = "" if success else result.stdout[-_STDOUT_TAIL_CHARS:]

            # Delete stale result from a previous failed run before saving
            if get_result(exp.id):
                delete_result(exp.id)
            save_result(result)

            if success:
                update_experiment_status(exp.id, "completed")
            else:
                error = ("exit_code=0 but no metrics produced" if result.exit_code == 0
                         else f"exit_code={result.exit_code}")
                log.warning("experiment_pipeline.run_failed", exp_id=exp.id, error=error)
                increment_retry(exp.id)
                _repair_or_fail(exp, f"{error}\nOutput tail:\n{result.stdout}", error)

        except Exception as e:
            log.error("experiment_pipeline.error", exp_id=exp.id, error=str(e))
            increment_retry(exp.id)
            update_experiment_status(exp.id, "failed", error=str(e))

    log.info("experiment_pipeline.done")
