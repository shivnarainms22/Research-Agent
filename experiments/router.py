"""Route experiments to local Docker or cloud Modal based on resource requirements."""
from __future__ import annotations

import re
import structlog

from core.models import Experiment

log = structlog.get_logger()

_LARGE_MODEL_PATTERNS = [
    r"\b(\d+)B\b",       # e.g. "7B", "13B"
    r"\b(\d+)b\b",
    r"llama",
    r"mistral",
    r"gpt-4",
]


def _needs_gpu(code: str) -> bool:
    gpu_markers = [
        "torch.cuda", "cuda()", ".to('cuda')", "to(device)", "torch.device('cuda'",
        'device_map="auto"', "device_map='auto'", "device_map=\"cuda\"",
        "load_in_4bit", "load_in_8bit",
    ]
    return any(m in code for m in gpu_markers)


def _is_large_model(code: str) -> bool:
    for pattern in _LARGE_MODEL_PATTERNS:
        match = re.search(pattern, code)
        if match:
            if pattern.startswith(r"\b(\d+)"):
                size = int(match.group(1))
                if size >= 1:  # >=1B params
                    return True
            else:
                return True
    return False


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def decide_target(exp: Experiment) -> str:
    """Return 'local' or 'cloud_modal'.

    Respects exp.execution_target if already set; falls back to code analysis.
    """
    if exp.execution_target in ("local", "cloud_modal"):
        return exp.execution_target

    code = exp.generated_code
    gpu_needed = _needs_gpu(code)
    large_model = _is_large_model(code)

    if not gpu_needed:
        return "local"

    if large_model:
        return "cloud_modal"

    # Small GPU model: use local if CUDA available, else cloud
    if _has_cuda():
        return "local"
    return "cloud_modal"
