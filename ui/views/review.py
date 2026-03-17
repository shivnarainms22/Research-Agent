"""Review Queue page — approve / reject / edit pending_review experiments."""
from __future__ import annotations

import json

import streamlit as st

from knowledge.experiment_store import (
    get_experiments_by_status,
    update_experiment_status,
    update_experiment_hypothesis,
)
from knowledge.paper_store import get_paper, get_analysis
from core.models import parse_json_list


@st.cache_data(ttl=5)
def _get_pending() -> list:
    return get_experiments_by_status("pending_review")


def _approve(exp_id: str) -> None:
    update_experiment_status(exp_id, "pending")
    st.cache_data.clear()


def _reject(exp_id: str) -> None:
    update_experiment_status(exp_id, "skipped", error="Rejected by user")
    st.cache_data.clear()


def _edit_and_approve(exp_id: str, new_hypothesis: str) -> None:
    update_experiment_hypothesis(exp_id, new_hypothesis)
    update_experiment_status(exp_id, "pending")
    st.cache_data.clear()


def render() -> None:
    st.title("Experiment Review Queue")

    experiments = _get_pending()

    if not experiments:
        st.success("No experiments awaiting review.")
        return

    # Sort control
    sort_by = st.selectbox("Sort by", ["Novelty Score (desc)", "Created (newest)"], index=0)

    # Build enriched list
    enriched = []
    for exp in experiments:
        paper = get_paper(exp.paper_id)
        analysis = get_analysis(exp.paper_id)
        enriched.append((exp, paper, analysis))

    if sort_by == "Novelty Score (desc)":
        enriched.sort(key=lambda t: t[2].novelty_score if t[2] else 0, reverse=True)
    else:
        enriched.sort(key=lambda t: t[0].created_at, reverse=True)

    col_left, col_right = st.columns([3, 1])
    col_left.subheader(f"{len(experiments)} experiments awaiting review")

    if col_right.button("✅ Approve All", use_container_width=True):
        for exp, _, _ in enriched:
            update_experiment_status(exp.id, "pending")
        st.cache_data.clear()
        st.rerun()

    st.divider()

    for exp, paper, analysis in enriched:
        paper_title = paper.title if paper else exp.paper_id
        novelty = analysis.novelty_score if analysis else "?"
        relevance = analysis.relevance_score if analysis else "?"

        label = f"**{exp.title}** — {paper_title[:80]}"
        with st.expander(label, expanded=False):
            meta_cols = st.columns(3)
            meta_cols[0].metric("Novelty", f"{novelty}/10")
            meta_cols[1].metric("Relevance", f"{relevance}/10")
            meta_cols[2].metric("Target", exp.execution_target)

            st.markdown(f"**Hypothesis:** {exp.hypothesis}")

            with st.expander("Generated Code", expanded=False):
                st.code(exp.generated_code, language="python")

            btn_col1, btn_col2, btn_col3 = st.columns(3)

            if btn_col1.button("✅ Approve", key=f"approve_{exp.id}", use_container_width=True):
                _approve(exp.id)
                st.rerun()

            if btn_col2.button("❌ Reject", key=f"reject_{exp.id}", use_container_width=True):
                _reject(exp.id)
                st.rerun()

            with btn_col3:
                if st.button("✏️ Edit & Approve", key=f"edit_btn_{exp.id}", use_container_width=True):
                    st.session_state[f"editing_{exp.id}"] = True

            if st.session_state.get(f"editing_{exp.id}"):
                new_hyp = st.text_area(
                    "Edit hypothesis",
                    value=exp.hypothesis,
                    key=f"hyp_edit_{exp.id}",
                )
                if st.button("Submit", key=f"submit_edit_{exp.id}"):
                    _edit_and_approve(exp.id, new_hyp)
                    st.session_state.pop(f"editing_{exp.id}", None)
                    st.rerun()
