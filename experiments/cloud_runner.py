"""Modal cloud experiment runner (GPU workloads)."""
from __future__ import annotations

import base64
import json
import re
import time
from datetime import datetime
from pathlib import Path

import structlog

from config import settings
from core.models import Experiment, ExperimentResult

log = structlog.get_logger()


def run(exp: Experiment) -> ExperimentResult:
    """Dispatch experiment to Modal for GPU execution."""
    try:
        import modal
    except ImportError:
        return _error_result(exp, "modal package not installed")

    workspace = settings.experiments_dir / exp.id
    results_dir = workspace / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (workspace / "run.py").write_text(exp.generated_code, encoding="utf-8")

    start_time = time.time()

    try:
        app = modal.App.lookup("research-sandbox", create_if_missing=True)
        # debian_slim gives a clean pip-managed Python (no conda).
        # pip_install() uses `python -m pip install` into /usr/local/lib/python3.11/site-packages.
        # Layer 1: torch CUDA — index_url replaces PyPI entirely, guaranteeing the cu118 wheel.
        #           (extra_index_url would let pip pick the CPU wheel from PyPI by version comparison.)
        # Layer 2: lightweight packages — cached independently, rebuilds fast when list changes.
        # nvidia/cuda base image provides CUDA 11.8 runtime + cuDNN + Ubuntu 22.04
        # system libs (libgomp, libstdc++, etc.) that torch requires.
        # add_python="3.11" installs a clean pip-managed Python — no conda.
        # pip_install() then targets that Python interpreter exclusively.
        # extra_index_url keeps PyPI for torch deps (filelock, sympy, etc.)
        # while also finding the +cu118 CUDA wheels on the PyTorch index.
        image = (
            modal.Image.from_registry(
                "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
                add_python="3.11",
            )
            .pip_install(
                "torch==2.4.0+cu118",
                "torchvision==0.19.0+cu118",
                extra_index_url="https://download.pytorch.org/whl/cu118",
            )
            .pip_install(
                "transformers", "datasets", "numpy<2", "scipy", "scikit-learn", "pandas",
                "tqdm", "matplotlib", "seaborn", "huggingface_hub", "accelerate", "evaluate",
                "pillow", "peft", "bitsandbytes>=0.46.1",
            )
        )

        # Patch /workspace → /tmp/ws (writable container path)
        code = exp.generated_code.replace("/workspace", "/tmp/ws")

        # Prepend a numpy-aware JSON encoder so experiments can freely use
        # numpy scalars (bool_, int64, float32, ndarray) in json.dump/dumps
        # without crashing on "Object of type bool_ is not JSON serializable".
        numpy_preamble = (
            "import json as _j, json\n"
            "class _NpEncoder(_j.JSONEncoder):\n"
            "    def default(self, o):\n"
            "        import numpy as _np\n"
            "        if isinstance(o, _np.integer): return int(o)\n"
            "        if isinstance(o, _np.floating): return float(o)\n"
            "        if isinstance(o, _np.bool_): return bool(o)\n"
            "        if isinstance(o, _np.ndarray): return o.tolist()\n"
            "        return super().default(o)\n"
            "_j_dump, _j_dumps = _j.dump, _j.dumps\n"
            "def _pd(o,f,**k): k.setdefault('cls',_NpEncoder); return _j_dump(o,f,**k)\n"
            "def _ps(o,**k): k.setdefault('cls',_NpEncoder); return _j_dumps(o,**k)\n"
            "_j.dump = _pd; _j.dumps = _ps; json.dump = _pd; json.dumps = _ps\n"
        )
        code = numpy_preamble + code

        # Fix deprecated transformers API: evaluation_strategy → eval_strategy (>=4.41)
        code = re.sub(r'\bevaluation_strategy\b', 'eval_strategy', code)

        # Fix deprecated sklearn API removed in 1.5: multi_class="multinomial"
        # is no longer accepted by LogisticRegression (lbfgs is now always multinomial).
        code = re.sub(r',\s*multi_class\s*=\s*["\']multinomial["\']', '', code)
        code = re.sub(r'multi_class\s*=\s*["\']multinomial["\'],\s*', '', code)

        # Fix floating-point dict key lookups: frac - 0.1 can produce
        # 0.19999...98 instead of 0.2, causing KeyError. Round arithmetic
        # results used as dict keys to 10 decimal places.
        code = re.sub(
            r'\[(\w+)\s*-\s*0\.1\]',
            r'[round(\1 - 0.1, 10)]',
            code,
        )
        code = re.sub(
            r'\[(\w+)\s*\+\s*0\.1\]',
            r'[round(\1 + 0.1, 10)]',
            code,
        )

        code_path = workspace / "run.py"
        code_path.write_text(code, encoding="utf-8")

        # Embed code as base64 to avoid file mounting complexity
        code_b64 = base64.b64encode(code.encode()).decode()
        bash_cmd = (
            f"mkdir -p /tmp/ws/results && "
            f"echo '{code_b64}' | base64 -d > /tmp/ws/run.py && "
            f"python /tmp/ws/run.py; "
            f"_exit=$?; "
            f"echo '---METRICS_JSON---'; "
            f"cat /tmp/ws/results/metrics.json 2>/dev/null || echo '{{}}'; "
            f"exit $_exit"
        )

        sb = modal.Sandbox.create(
            "bash", "-c", bash_cmd,
            image=image,
            gpu="T4",
            timeout=settings.experiment_timeout_seconds,
            app=app,
        )
        sb.wait()

        stdout_text = sb.stdout.read() or ""
        stderr_text = sb.stderr.read() or ""
        exit_code = sb.returncode

        # Parse metrics from the sentinel line we emitted
        metrics = {}
        if "---METRICS_JSON---" in stdout_text:
            try:
                metrics_str = stdout_text.split("---METRICS_JSON---")[-1].strip()
                metrics = json.loads(metrics_str)
            except Exception:
                pass

        full_output = stdout_text + stderr_text
        runtime = time.time() - start_time

        metrics_path = results_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        return ExperimentResult(
            id=f"result_{exp.id}",
            experiment_id=exp.id,
            stdout=full_output[:50000],
            exit_code=exit_code,
            metrics=json.dumps(metrics),
            artifacts=json.dumps([str(metrics_path)]),
            runtime_seconds=runtime,
            recorded_at=datetime.utcnow(),
        )

    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        log.error("cloud_runner.error", exp_id=exp.id, error=err_msg)
        return _error_result(exp, err_msg)


def _error_result(exp: Experiment, error: str) -> ExperimentResult:
    return ExperimentResult(
        id=f"result_{exp.id}",
        experiment_id=exp.id,
        stdout=f"Cloud runner error: {error}",
        exit_code=1,
        metrics="{}",
        artifacts="[]",
        runtime_seconds=0.0,
        recorded_at=datetime.utcnow(),
    )
