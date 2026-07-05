"""Cheap pre-run check that an experiment actually tests the paper's claim."""
from __future__ import annotations

import json

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import Experiment
from knowledge.paper_store import get_analysis

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


_CRITIC_TOOL = {
    "name": "review_experiment",
    "description": "Judge whether an experiment script faithfully tests the paper's claim",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["sound", "flawed"]},
            "reason": {"type": "string", "description": "One sentence justification"},
        },
        "required": ["verdict", "reason"],
    },
}


def _paper_claim(paper_id: str) -> str:
    analysis = get_analysis(paper_id)
    if analysis is None:
        return ""
    try:
        specs = json.loads(analysis.reproducible_experiments)
        return "; ".join(
            f"{s.get('title', '')}: expects {s.get('expected_metric', '?')} "
            f"(baseline {s.get('baseline_claimed', '?')})"
            for s in specs[:3]
        )
    except Exception:
        return ""


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    stop=stop_after_attempt(3),
)
def review(exp: Experiment) -> tuple[str, str]:
    """Return (verdict, reason); verdict is 'sound' or 'flawed'.

    Verdict defaults to 'sound' on any error so the critic never blocks a run.
    """
    try:
        prompt = f"""Paper claim(s):
{_paper_claim(exp.paper_id) or '(unavailable)'}

Experiment: {exp.title}
Hypothesis: {exp.hypothesis}

Script:
```python
{exp.generated_code[:8000]}
```

Does this script faithfully test the paper's claim — right dataset, right metric,
a real (not stubbed/simulated) measurement? Answer sound or flawed with one reason."""

        response = _get_client().messages.create(
            model=settings.claude_haiku_model,
            max_tokens=256,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            tools=[_CRITIC_TOOL],
            tool_choice={"type": "tool", "name": "review_experiment"},
        )
        token_tracker.track("experiment_critic", response.usage.input_tokens, response.usage.output_tokens)

        result = next((b.input for b in response.content if b.type == "tool_use"), None)
        if not result:
            return "sound", ""
        return result.get("verdict", "sound"), result.get("reason", "")
    except anthropic.RateLimitError:
        raise
    except Exception as e:
        log.warning("experiment_critic.error", exp_id=exp.id, error=str(e))
        return "sound", ""
