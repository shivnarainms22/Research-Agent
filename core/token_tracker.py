"""Module-level token usage accumulator — reset per pipeline cycle."""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_totals: dict = {}
_lock = threading.Lock()
_current_cycle_id: str = ""


def set_cycle_id(cycle_id: str) -> None:
    """Set the active cycle ID so track() can persist logs without callers passing it."""
    global _current_cycle_id
    with _lock:
        _current_cycle_id = cycle_id


def track(module: str, input_tokens: int, output_tokens: int, cycle_id: str = "") -> None:
    """Accumulate token usage for a module and optionally persist to DB."""
    global _totals, _current_cycle_id
    with _lock:
        if module not in _totals:
            _totals[module] = {"input": 0, "output": 0}
        _totals[module]["input"] += input_tokens
        _totals[module]["output"] += output_tokens
        effective_cycle = cycle_id or _current_cycle_id

    if effective_cycle:
        try:
            from knowledge.token_log_store import save_log
            save_log(effective_cycle, module, input_tokens, output_tokens)
        except Exception as e:
            logger.debug(f"token_tracker DB write failed: {e}")


def get_totals() -> dict:
    """Return aggregated totals and per-module breakdown."""
    with _lock:
        input_total = sum(v["input"] for v in _totals.values())
        output_total = sum(v["output"] for v in _totals.values())
        by_module = {k: {"input": v["input"], "output": v["output"]} for k, v in _totals.items()}
    return {
        "input_total": input_total,
        "output_total": output_total,
        "by_module": by_module,
    }


def reset() -> None:
    """Clear all accumulated totals (call at start of each cycle)."""
    global _totals
    with _lock:
        _totals = {}
