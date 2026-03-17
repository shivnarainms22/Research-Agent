"""Generate ablation variant experiments via Claude."""
from __future__ import annotations

import json
import uuid
from datetime import datetime

import anthropic
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from core import token_tracker
from core.models import Experiment, ExperimentResult

log = structlog.get_logger()

_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


_ABLATION_TOOL = {
    "name": "generate_ablations",
    "description": "Generate ablation variant experiments",
    "input_schema": {
        "type": "object",
        "properties": {
            "ablations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "hypothesis": {"type": "string"},
                        "modified_code": {"type": "string"},
                        "what_changed": {"type": "string"},
                    },
                    "required": ["title", "hypothesis", "modified_code", "what_changed"],
                },
            }
        },
        "required": ["ablations"],
    },
}


@retry(
    retry=retry_if_exception_type(anthropic.RateLimitError),
    wait=wait_exponential(multiplier=1, min=60, max=300),
    stop=stop_after_attempt(3),
)
def generate_ablations(exp: Experiment, result: ExperimentResult) -> list[Experiment]:
    """Generate 2-3 ablation variants of a completed experiment."""
    client = _get_client()

    metrics = json.loads(result.metrics) if result.metrics else {}

    prompt = f"""Experiment: {exp.title}
Hypothesis: {exp.hypothesis}
Results: {json.dumps(metrics, indent=2)[:1000]}
Exit code: {result.exit_code}

Original code:
```python
{exp.generated_code[:4000]}
```

Generate 2-3 ablation variants that:
1. Change one component at a time (learning rate, architecture, data size, etc.)
2. Test what contributes to the result
3. Are runnable on the same compute tier"""

    response = client.messages.create(
        model=settings.claude_model,
        max_tokens=2048,
        temperature=0.2,
        messages=[{"role": "user", "content": prompt}],
        tools=[_ABLATION_TOOL],
        tool_choice={"type": "tool", "name": "generate_ablations"},
    )

    log.info(
        "claude.usage",
        module="ablation_manager",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
    )
    token_tracker.track("ablation_manager", response.usage.input_tokens, response.usage.output_tokens)

    tool_result = next(
        (b.input for b in response.content if b.type == "tool_use"),
        None,
    )
    if not tool_result:
        return []

    ablations: list[Experiment] = []
    for abl in tool_result.get("ablations", []):
        ablation = Experiment(
            id=str(uuid.uuid4()),
            paper_id=exp.paper_id,
            title=f"[Ablation] {abl['title']}",
            hypothesis=abl["hypothesis"],
            generated_code=abl["modified_code"],
            execution_target=exp.execution_target,
            status="pending",
            parent_experiment_id=exp.id,
            created_at=datetime.utcnow(),
            retry_count=0,
        )
        ablations.append(ablation)
        log.info("ablation.created", title=ablation.title, parent=exp.id)

    return ablations
