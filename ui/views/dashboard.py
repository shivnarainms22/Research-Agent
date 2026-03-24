"""Dashboard page — cycle status, experiment counts, token/cost metrics."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from config import settings

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_LOG_FILE = settings.state_dir / "pipeline_output.log"

STAGES = ["ingestion", "synthesis", "experiments", "analysis", "reporting"]

STAGE_DESCRIPTIONS = {
    "ingestion":   "Fetching new papers from arXiv, Semantic Scholar & Substack",
    "synthesis":   "Analyzing papers with Claude — scoring relevance/novelty, detecting contradictions, generating experiment hypotheses",
    "experiments": "Running approved experiments inside Docker sandbox",
    "analysis":    "Comparing results against paper-claimed baselines, running statistical tests & ablations",
    "reporting":   "Generating weekly Markdown report with Claude narrative",
}


@st.cache_data(ttl=8)
def _get_experiment_counts() -> dict[str, int]:
    from knowledge.experiment_store import get_experiments_by_status
    statuses = ["pending_review", "pending", "running", "completed", "failed", "skipped"]
    return {s: len(get_experiments_by_status(s)) for s in statuses}


@st.cache_data(ttl=8)
def _get_paper_count() -> int:
    from knowledge.paper_store import get_all_papers
    return len(get_all_papers(limit=100_000))


def _get_recent_cycles() -> list[dict]:
    cycles = []
    for p in sorted(settings.state_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            inp = data.get("total_input_tokens", 0)
            out = data.get("total_output_tokens", 0)
            cost = inp * 3 / 1_000_000 + out * 15 / 1_000_000
            cycles.append({
                "cycle_id": data.get("cycle_id", p.stem)[:20],
                "stage": data.get("current_stage", "?"),
                "status": "Done" if data.get("is_complete") else "Running",
                "new papers": len(data.get("paper_ids_this_cycle", [])),
                "experiments": len(data.get("experiment_ids_this_cycle", [])),
                "input_tok": f"{inp:,}",
                "output_tok": f"{out:,}",
                "cost": f"${cost:.4f}",
            })
        except Exception:
            pass
    return cycles


def _get_active_cycle() -> dict | None:
    candidates = []
    for p in settings.state_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if not data.get("is_complete"):
                candidates.append((p.stat().st_mtime, data))
        except Exception:
            pass
    if not candidates:
        return None
    _, data = max(candidates, key=lambda x: x[0])
    return data


def _get_last_completed_cycle() -> dict | None:
    candidates = []
    for p in settings.state_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("is_complete"):
                candidates.append((p.stat().st_mtime, data))
        except Exception:
            pass
    if not candidates:
        return None
    _, data = max(candidates, key=lambda x: x[0])
    return data


def _is_pipeline_locked() -> bool:
    lock_path = settings.state_dir / "pipeline.lock"
    if not lock_path.exists():
        return False
    from filelock import FileLock, Timeout
    lock = FileLock(str(lock_path))
    try:
        lock.acquire(timeout=0)
        lock.release()
        return False
    except Timeout:
        return True


def _launch(cmd: list[str]) -> None:
    """Launch a hidden background subprocess, capturing output to a log file."""
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(_LOG_FILE, "w", encoding="utf-8", errors="replace")
    kwargs: dict = {
        "cwd": str(_PROJECT_ROOT),
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
    }
    if sys.platform == "win32":
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = si
        # CREATE_NO_WINDOW, CREATE_NEW_CONSOLE, and DETACHED_PROCESS are mutually
        # exclusive on Windows — use CREATE_NO_WINDOW only.
        kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP
            | subprocess.CREATE_NO_WINDOW
        )
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)


def _read_log_tail(n: int = 40) -> str:
    if not _LOG_FILE.exists():
        return ""
    try:
        lines = _LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-n:])
    except Exception:
        return ""


def _log_started_after(launch_time: float) -> bool:
    if not _LOG_FILE.exists():
        return False
    return _LOG_FILE.stat().st_mtime > launch_time


def _log_still_active(launch_time: float, idle_secs: float = 8.0) -> bool:
    if not _LOG_FILE.exists():
        return False
    mtime = _LOG_FILE.stat().st_mtime
    return mtime > launch_time and (time.time() - mtime) < idle_secs


def _render_pipeline_status(state: dict) -> None:
    stage = state.get("current_stage", "unknown")
    error = state.get("last_error")

    st.subheader("Pipeline is running")
    stage_idx = STAGES.index(stage) if stage in STAGES else 0
    progress = (stage_idx + 1) / len(STAGES)
    st.progress(progress, text=f"Stage {stage_idx + 1} of {len(STAGES)}: **{stage}**")

    desc = STAGE_DESCRIPTIONS.get(stage, "")
    if desc:
        st.info(f"**Now:** {desc}")

    cols = st.columns(len(STAGES))
    for i, (s, col) in enumerate(zip(STAGES, cols)):
        if i < stage_idx:
            col.success(f"✓ {s}")
        elif i == stage_idx:
            col.warning(f"⏳ {s}")
        else:
            col.markdown(f"<span style='color:#555'>○ {s}</span>", unsafe_allow_html=True)

    st.markdown("")
    inp = state.get("total_input_tokens", 0)
    out = state.get("total_output_tokens", 0)
    cost = inp * 3 / 1_000_000 + out * 15 / 1_000_000
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Papers ingested", len(state.get("paper_ids_this_cycle", [])))
    m2.metric("Experiments run", len(state.get("experiment_ids_this_cycle", [])))
    m3.metric("Tokens so far", f"{inp + out:,}")
    m4.metric("Cost so far", f"${cost:.4f}")

    if error:
        st.error(f"Last error: {error}")

    log_tail = _read_log_tail(40)
    if log_tail:
        with st.expander("Live log (last 40 lines)", expanded=True):
            st.code(log_tail, language=None)


def _render_last_cycle_summary(state: dict) -> None:
    cycle_id = state.get("cycle_id", "?")
    papers = len(state.get("paper_ids_this_cycle", []))
    exps = len(state.get("experiment_ids_this_cycle", []))
    inp = state.get("total_input_tokens", 0)
    out = state.get("total_output_tokens", 0)
    cost = inp * 3 / 1_000_000 + out * 15 / 1_000_000

    st.success(f"Last run complete: **{cycle_id}**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Papers ingested", papers)
    c2.metric("Experiments queued", exps)
    c3.metric("Cost", f"${cost:.4f}")

    if papers == 0 and exps == 0 and inp == 0:
        st.warning("No new papers found and no analysis ran. arXiv posts no papers on weekends — try Ingest (7 days).")
    elif papers == 0 and exps > 0:
        st.info(f"0 new papers ingested — existing DB papers were synthesized. {exps} experiment(s) queued — go to **Review Queue** to approve them.")
    elif papers == 0 and inp > 0:
        st.info("0 new papers ingested — existing DB papers were analyzed (no experiments above novelty threshold).")
    elif exps == 0 and inp == 0:
        st.warning(
            "Papers ingested but no Claude analysis ran — no papers matched your keyword threshold. "
            "Go to **Settings** and lower **Min keyword matches to analyze** to 1."
        )
    elif exps == 0:
        st.info("Papers analyzed but no experiments generated (novelty scores below threshold). Check Review Queue.")
    else:
        st.info(f"{exps} experiment(s) queued — go to **Review Queue** to approve them.")


@st.fragment(run_every=3)
def _pipeline_status_fragment() -> None:
    """Auto-refreshes every 3s without dimming the full page."""
    locked = _is_pipeline_locked()
    active = _get_active_cycle()
    launch_time = st.session_state.get("launch_time")
    launch_retries = st.session_state.get("launch_retries", 0)
    launch_cmd = st.session_state.get("launch_cmd", "run")
    just_launched = launch_time is not None and (time.time() - launch_time) < 20

    if locked and active:
        _render_pipeline_status(active)
        st.session_state.launch_time = None
        st.session_state.launch_retries = 0
    elif locked and not active:
        st.info("Pipeline is starting up...")
        log_tail = _read_log_tail(20)
        if log_tail:
            with st.expander("Log", expanded=True):
                st.code(log_tail, language=None)
    elif just_launched:
        log_started = _log_started_after(launch_time)
        log_active = _log_still_active(launch_time)
        if log_active:
            st.info(f"Running **{launch_cmd}**...")
            log_tail = _read_log_tail(40)
            if log_tail:
                with st.expander("Live log", expanded=True):
                    st.code(log_tail, language=None)
        elif log_started:
            st.success(f"**{launch_cmd.capitalize()}** completed.")
            log_tail = _read_log_tail(40)
            if log_tail:
                with st.expander("Output log", expanded=True):
                    st.code(log_tail, language=None)
            st.session_state.launch_time = None
            st.session_state.launch_retries = 0
        elif launch_retries < 8:
            st.info(f"Launching **{launch_cmd}**... ({launch_retries * 3}s)")
            st.session_state.launch_retries = launch_retries + 1
        else:
            st.error(
                f"**{launch_cmd}** failed to start from `{_PROJECT_ROOT}`. "
                "Check that `uv` is in your PATH."
            )
            st.session_state.launch_time = None
            st.session_state.launch_retries = 0
    else:
        last = _get_last_completed_cycle()
        if last:
            _render_last_cycle_summary(last)


def render() -> None:
    if "launch_time" not in st.session_state:
        st.session_state.launch_time = None
    if "launch_retries" not in st.session_state:
        st.session_state.launch_retries = 0
    if "launch_cmd" not in st.session_state:
        st.session_state.launch_cmd = "run"

    # ── Topbar: title left, action buttons right ──
    locked = _is_pipeline_locked()
    col_title, _, col_actions = st.columns([3, 1, 2])
    with col_title:
        st.markdown("## Dashboard")
        st.caption("Autonomous PhD research pipeline")
    with col_actions:
        if locked:
            st.warning("Pipeline running")
        else:
            c1, c2, c3 = st.columns(3)
            if c1.button("▶ Run", use_container_width=True):
                _launch(["uv", "run", "python", "main.py", "run"])
                st.session_state.launch_time = time.time()
                st.session_state.launch_retries = 0
                st.session_state.launch_cmd = "run"
            if c2.button("📥 Ingest", use_container_width=True):
                _launch(["uv", "run", "python", "main.py", "ingest", "--source", "arxiv", "--days", "7"])
                st.session_state.launch_time = time.time()
                st.session_state.launch_retries = 0
                st.session_state.launch_cmd = "ingest"
            if c3.button("📊 Report", use_container_width=True):
                _launch(["uv", "run", "python", "main.py", "report"])
                st.session_state.launch_time = time.time()
                st.session_state.launch_retries = 0
                st.session_state.launch_cmd = "report"

    counts = _get_experiment_counts()
    total_papers = _get_paper_count()
    total_cost = 0.0
    if settings.state_dir.exists():
        for p in settings.state_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                total_cost += d.get("total_input_tokens", 0) * 3 / 1_000_000
                total_cost += d.get("total_output_tokens", 0) * 15 / 1_000_000
            except Exception:
                pass

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Papers", total_papers)
    c2.metric("Pending Review", counts["pending_review"])
    c3.metric("Completed Experiments", counts["completed"])
    c4.metric("Est. Total Cost", f"${total_cost:.2f}")

    st.divider()

    # Pipeline status — auto-refreshes every 3s without full-page dim
    _pipeline_status_fragment()

    st.divider()

    # Experiment status
    st.subheader("Experiment Status")
    status_cols = st.columns(6)
    labels = ["pending_review", "pending", "running", "completed", "failed", "skipped"]
    colors = ["🟡", "🔵", "🟠", "🟢", "🔴", "⚫"]
    for col, label, color in zip(status_cols, labels, colors):
        col.metric(f"{color} {label.replace('_', ' ').title()}", counts[label])

    st.divider()

    # Recent cycles
    st.subheader("Recent Cycles")
    cycles = _get_recent_cycles()
    if cycles:
        import pandas as pd
        st.dataframe(pd.DataFrame(cycles), use_container_width=True, hide_index=True)
    else:
        st.info("No cycles found. Run the pipeline to get started.")

    st.divider()

    with st.expander("What does each pipeline stage do?"):
        st.markdown(f"""
**Project root:** `{_PROJECT_ROOT}`

**1. Ingestion** — Queries arXiv (cs.LG/AI/CL/CV/RO), Semantic Scholar, and Substack for new papers matching your keywords. Deduplicates against existing DB entries.

**2. Synthesis** — For each new paper: fetches full text, scores relevance & novelty with Claude (1–10). Papers need ≥ `min_keyword_matches_to_analyze` keyword matches (set in **Settings**). Qualifying papers get contradiction detection and experiment hypothesis generation — all go to `pending_review`.

**3. Experiments** — Runs approved (`pending`) experiments inside Docker (8 GB RAM, 4 CPUs). Code is AST + bandit validated first.

**4. Analysis** — Compares results against paper-claimed metrics. Runs t-tests, Cohen's d. Generates ablation variants.

**5. Reporting** — Claude writes a Markdown research report.

**Note:** arXiv posts no papers on weekends. Use "Ingest (7 days)" on weekends.
""")
