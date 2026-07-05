"""Claude-powered repair of failed experiment code."""
from __future__ import annotations

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import Experiment

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


_REPAIR_SYSTEM = """\
You are an expert ML engineer debugging a failed experiment script.
Diagnose the failure from the output, then produce a corrected, complete script.
- Fix the root cause, not the symptom; keep the experiment's scientific intent unchanged
- The script must stay fully self-contained and write results to /workspace/results/metrics.json
- If the failure is a missing/gated dataset or model, substitute the closest public equivalent and log it
"""

_REPAIR_TOOL = {
    "name": "repair_experiment",
    "description": "Diagnose a failed experiment and return corrected code",
    "input_schema": {
        "type": "object",
        "properties": {
            "diagnosis": {"type": "string", "description": "1-2 sentence root-cause diagnosis"},
            "python_code": {"type": "string", "description": "Complete corrected Python script"},
        },
        "required": ["diagnosis", "python_code"],
    },
}


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    stop=stop_after_attempt(3),
)
def repair(exp: Experiment, failure_context: str) -> tuple[str, str] | None:
    """Return (fixed_code, diagnosis), or None if no usable repair was produced."""
    prompt = f"""Experiment: {exp.title}
Hypothesis: {exp.hypothesis}

Script:
```python
{exp.generated_code}
```

Failure:
{failure_context}

Diagnose the root cause and return the corrected script."""

    response = _get_client().messages.create(
        model=settings.claude_model,
        max_tokens=16000,
        system=[{"type": "text", "text": _REPAIR_SYSTEM, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": prompt}],
        tools=[_REPAIR_TOOL],
        tool_choice={"type": "tool", "name": "repair_experiment"},
    )
    token_tracker.track("code_repairer", response.usage.input_tokens, response.usage.output_tokens)

    result = next((b.input for b in response.content if b.type == "tool_use"), None)
    if not result or "python_code" not in result:
        log.warning("code_repairer.no_repair", exp_id=exp.id)
        return None
    return result["python_code"], result.get("diagnosis", "")
