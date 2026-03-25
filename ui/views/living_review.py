"""Living Review page — Themes, Gaps, Contradictions."""
from __future__ import annotations

import streamlit as st

from config import settings
from core.models import parse_json_list
from knowledge.theme_store import get_all_themes
from knowledge.gap_store import get_gaps
from knowledge.contradiction_store import get_recent_contradictions
from knowledge.paper_store import get_paper


@st.cache_data(ttl=10)
def _themes():
    return get_all_themes()


@st.cache_data(ttl=10)
def _gaps():
    return get_gaps()


@st.cache_data(ttl=10)
def _contradictions():
    return get_recent_contradictions(days=30)


def _run_theme_clusterer() -> None:
    with st.status("Running Theme Clusterer…", expanded=True) as s:
        try:
            st.write("Clustering papers by topic…")
            from knowledge.theme_clusterer import cluster_themes
            cluster_themes()
            st.write("Done — reloading themes…")
            st.cache_data.clear()
            s.update(label="Theme clustering complete", state="complete")
        except Exception as e:
            s.update(label=f"Failed: {e}", state="error")
            st.error(str(e))


def _run_gap_finder() -> None:
    with st.status("Running Gap Finder…", expanded=True) as s:
        try:
            st.write("Analysing corpus for research gaps…")
            import uuid
            from knowledge.gap_finder import find_gaps
            find_gaps(str(uuid.uuid4()))
            st.write("Done — reloading gaps…")
            st.cache_data.clear()
            s.update(label="Gap finding complete", state="complete")
        except Exception as e:
            s.update(label=f"Failed: {e}", state="error")
            st.error(str(e))


def render() -> None:
    st.title("Living Review")

    themes_tab, gaps_tab, contradictions_tab = st.tabs(
        ["Research Themes", "Research Gaps", "Contradictions"]
    )

    # ------------------------------------------------------------------ Themes
    with themes_tab:
        themes = _themes()
        col1, col2 = st.columns([4, 1])
        col1.subheader(f"{len(themes)} research themes")
        if col2.button("Run Theme Clusterer", use_container_width=True):
            _run_theme_clusterer()
            st.rerun()

        if not themes:
            st.info("No themes yet. Run the theme clusterer to generate them.")
        else:
            for theme in themes:
                paper_ids = parse_json_list(theme.paper_ids)
                with st.expander(f"**{theme.name}** — {theme.paper_count} papers", expanded=False):
                    st.markdown(theme.description)
                    if paper_ids:
                        st.markdown("**Papers:**")
                        for pid in paper_ids[:20]:
                            paper = get_paper(pid)
                            title = paper.title if paper else pid[:16]
                            st.markdown(f"- {title}")
                        if len(paper_ids) > 20:
                            st.caption(f"… and {len(paper_ids) - 20} more")

    # -------------------------------------------------------------------- Gaps
    with gaps_tab:
        gaps = _gaps()
        col1, col2 = st.columns([4, 1])
        col1.subheader(f"{len(gaps)} research gaps")
        if col2.button("Run Gap Finder", use_container_width=True):
            _run_gap_finder()
            st.rerun()

        if not gaps:
            st.info("No gaps yet. Run the gap finder (requires ≥3 analyzed papers).")
        else:
            for i, gap in enumerate(gaps, 1):
                supporting = parse_json_list(gap.supporting_paper_ids)
                with st.expander(f"Gap {i}: {gap.description[:80]}", expanded=False):
                    st.markdown(gap.description)
                    if supporting:
                        st.markdown("**Supporting Papers:**")
                        for pid in supporting:
                            paper = get_paper(pid)
                            title = paper.title if paper else pid[:16]
                            st.markdown(f"- {title}")

    # --------------------------------------------------------- Contradictions
    with contradictions_tab:
        contradictions = _contradictions()
        st.subheader(f"{len(contradictions)} contradictions (last 30 days)")

        if not contradictions:
            st.info("No contradictions detected in the last 30 days.")
        else:
            import pandas as pd
            rows = []
            for c in contradictions:
                new_paper = get_paper(c.paper_id_new)
                old_paper = get_paper(c.paper_id_old)
                rows.append({
                    "metric": c.metric,
                    "severity": c.severity,
                    "description": c.description[:100],
                    "new_paper": new_paper.title[:40] if new_paper else c.paper_id_new[:16],
                    "old_paper": old_paper.title[:40] if old_paper else c.paper_id_old[:16],
                    "detected_at": c.detected_at.strftime("%Y-%m-%d"),
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True, hide_index=True)
