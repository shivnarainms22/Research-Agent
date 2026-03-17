"""Dashboard page — cycle status, experiment counts, token/cost metrics."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import streamlit as st

from config import settings


@st.cache_data(ttl=10)
def _get_experiment_counts() -> dict[str, int]:
    from knowledge.experiment_store import get_experiments_by_status
    statuses = ["pending_review", "pending", "running", "completed", "failed", "skipped"]
    return {s: len(get_experiments_by_status(s)) for s in statuses}


@st.cache_data(ttl=10)
def _get_paper_count() -> int:
    from knowledge.paper_store import get_all_papers
    return len(get_all_papers(limit=100_000))


@st.cache_data(ttl=10)
def _get_recent_cycles() -> list[dict]:
    from core.state import load_state
    cycles = []
    for p in sorted(settings.state_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:10]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            inp = data.get("total_input_tokens", 0)
            out = data.get("total_output_tokens", 0)
            cost = inp * 3 / 1_000_000 + out * 15 / 1_000_000
            cycles.append({
                "cycle_id": data.get("cycle_id", p.stem)[:16],
                "stage": data.get("current_stage", "?"),
                "complete": "✓" if data.get("is_complete") else "…",
                "input_tokens": inp,
                "output_tokens": out,
                "cost_usd": f"${cost:.4f}",
            })
        except Exception:
            pass
    return cycles


def _launch(cmd: list[str]) -> None:
    subprocess.Popen(cmd, cwd=str(settings.base_dir))
    st.info("Pipeline started in background — check status in a moment.")


def render() -> None:
    st.title("Research Agent — Dashboard")

    counts = _get_experiment_counts()
    total_papers = _get_paper_count()
    total_cost = sum(
        (json.loads(p.read_text(encoding="utf-8")).get("total_input_tokens", 0) * 3 / 1_000_000 +
         json.loads(p.read_text(encoding="utf-8")).get("total_output_tokens", 0) * 15 / 1_000_000)
        for p in settings.state_dir.glob("*.json")
        if p.is_file()
    ) if settings.state_dir.exists() else 0.0

    # Top metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Papers", total_papers)
    c2.metric("Pending Review", counts["pending_review"])
    c3.metric("Completed Experiments", counts["completed"])
    c4.metric("Est. Total Cost", f"${total_cost:.2f}")

    st.divider()

    # Experiment status breakdown
    st.subheader("Experiment Status")
    status_cols = st.columns(6)
    labels = ["pending_review", "pending", "running", "completed", "failed", "skipped"]
    colors = ["🟡", "🔵", "🟠", "🟢", "🔴", "⚫"]
    for col, label, color in zip(status_cols, labels, colors):
        col.metric(f"{color} {label.replace('_', ' ').title()}", counts[label])

    st.divider()

    # Active cycles
    st.subheader("Recent Cycles")
    cycles = _get_recent_cycles()
    if cycles:
        import pandas as pd
        df = pd.DataFrame(cycles)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No cycles found. Run the pipeline to get started.")

    st.divider()

    # Action buttons
    st.subheader("Pipeline Actions")
    col1, col2, col3 = st.columns(3)
    if col1.button("▶ Run Full Cycle", use_container_width=True):
        _launch(["uv", "run", "python", "main.py", "run"])
    if col2.button("📥 Ingest (7 days)", use_container_width=True):
        _launch(["uv", "run", "python", "main.py", "ingest", "--source", "arxiv", "--days", "7"])
    if col3.button("📊 Generate Report", use_container_width=True):
        _launch(["uv", "run", "python", "main.py", "report"])
