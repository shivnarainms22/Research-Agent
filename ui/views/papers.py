"""Paper Browser page — search and drill-down into paper analyses."""
from __future__ import annotations

import json

import streamlit as st

from core.models import parse_json_list
from knowledge.paper_store import get_all_papers, get_analysis


@st.cache_data(ttl=10)
def _all_papers(limit: int = 500):
    return get_all_papers(limit=limit)


def _search_papers(query: str, n: int = 20):
    try:
        from knowledge.retriever import search
        return search(query, n=n)
    except Exception:
        # Fallback: simple title/abstract substring match
        papers = _all_papers(limit=10_000)
        q = query.lower()
        return [p for p in papers if q in p.title.lower() or q in p.abstract.lower()][:n]


def render() -> None:
    st.title("Paper Browser")

    query = st.text_input("Search papers (hybrid BM25 + vector)", placeholder="e.g. sparse autoencoder superposition")
    sources = st.multiselect("Filter by source", ["arxiv", "semantic_scholar", "substack"], default=[])

    if query:
        papers = _search_papers(query)
    else:
        papers = _all_papers(limit=500)

    if sources:
        papers = [p for p in papers if p.source in sources]

    st.caption(f"{len(papers)} papers shown")

    for paper in papers:
        analysis = get_analysis(paper.id)
        novelty = f"{analysis.novelty_score:.1f}" if analysis else "—"
        relevance = f"{analysis.relevance_score:.1f}" if analysis else "—"

        source_badge = {"arxiv": "📄", "semantic_scholar": "🔬", "substack": "📝"}.get(paper.source, "📌")
        label = f"{source_badge} **{paper.title}** | Novelty: {novelty} | Relevance: {relevance}"

        with st.expander(label, expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Source", paper.source)
            col2.metric("Published", str(paper.published_date))
            col3.metric("Novelty", novelty)
            col4.metric("Relevance", relevance)

            st.markdown(f"**Abstract:** {paper.abstract[:500]}{'…' if len(paper.abstract) > 500 else ''}")

            if analysis:
                contributions = parse_json_list(analysis.key_contributions)
                if contributions:
                    st.markdown("**Key Contributions:**")
                    for c in contributions:
                        st.markdown(f"- {c}")

                datasets = parse_json_list(analysis.datasets_used)
                if datasets:
                    st.markdown(f"**Datasets:** {', '.join(datasets)}")

                limitations = parse_json_list(analysis.limitations)
                if limitations:
                    st.markdown("**Caveats:**")
                    for lim in limitations:
                        st.markdown(f"- {lim}")

                st.markdown(f"**Reproducibility Difficulty:** {analysis.reproducibility_difficulty}")

            col_a, col_b, col_c = st.columns(3)
            col_a.link_button("Open Paper", paper.url)
            if col_b.button("View Experiments →", key=f"view_exp_{paper.id}"):
                st.session_state["exp_filter_paper_id"] = paper.id
                st.info("Switch to the Experiments page to see filtered results.")

            btn_label = "Synthesize" if not analysis else "Run Experiments"
            if col_c.button(btn_label, key=f"synth_{paper.id}"):
                _run_synthesis(paper)


def _run_synthesis(paper) -> None:
    """Run analysis (if needed) + experiment extraction on a single paper."""
    from config import settings
    from knowledge.paper_store import (
        get_analysis, save_analysis, update_paper_status, update_paper_full_text,
    )
    from knowledge.experiment_store import save_experiment
    from synthesis import paper_analyzer, experiment_extractor
    from knowledge import vector_store

    with st.status(f"Processing: {paper.title[:60]}…", expanded=True) as status_box:
        # 1. Use existing analysis or run Claude if not yet analyzed
        analysis = get_analysis(paper.id)
        if analysis:
            st.write(f"✓ Using existing analysis — novelty={analysis.novelty_score:.1f} relevance={analysis.relevance_score:.1f}")
        else:
            # Fetch full text for arXiv papers
            if paper.source == "arxiv" and not paper.full_text:
                st.write("Fetching full text from arXiv…")
                try:
                    from ingestion.fulltext_fetcher import fetch_arxiv_fulltext
                    ft = fetch_arxiv_fulltext(paper.source_id)
                    if ft:
                        paper.full_text = ft
                        update_paper_full_text(paper.id, ft)
                        st.write(f"✓ Full text: {len(ft):,} chars")
                except Exception as e:
                    st.write(f"⚠ Full text fetch failed: {e}")

            st.write("Analyzing paper with Claude…")
            try:
                analysis = paper_analyzer.analyze_paper(paper)
                save_analysis(analysis)
                update_paper_status(paper.id, "analyzed")
                update_paper_full_text(paper.id, None)
                st.write(f"✓ Analysis done — novelty={analysis.novelty_score:.1f} relevance={analysis.relevance_score:.1f} difficulty={analysis.reproducibility_difficulty}")
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                status_box.update(label="Analysis failed", state="error")
                return

        # 2. Contradiction check (best-effort)
        try:
            from knowledge import contradiction_detector
            contradiction_detector.check_new_paper(paper.id, analysis)
        except Exception:
            pass

        # 4. Embed
        try:
            vector_store.embed_paper(paper)
            st.write("✓ Embedded into vector store")
        except Exception as e:
            st.write(f"⚠ Embedding failed: {e}")

        # 5. Experiment generation
        if analysis.relevance_score < settings.min_relevance_score_to_experiment:
            st.warning(
                f"Relevance {analysis.relevance_score:.1f} < threshold "
                f"{settings.min_relevance_score_to_experiment} — skipping experiment generation."
            )
            status_box.update(label="Analysis done (relevance too low for experiments)", state="complete")
            st.cache_data.clear()
            return

        st.write("Extracting experiments…")
        try:
            experiments = experiment_extractor.extract_experiments(paper.id, analysis)
            if not experiments:
                st.write("No new experiments generated (may already exist — check Review Queue)")
            else:
                for exp in experiments:
                    save_experiment(exp)
                st.write(f"✓ {len(experiments)} experiment(s) → pending_review")
        except Exception as e:
            st.error(f"Experiment extraction failed: {e}")
            status_box.update(label="Experiment extraction failed", state="error")
            st.cache_data.clear()
            return

        status_box.update(label="Done! Check Review Queue to approve experiments.", state="complete")
        st.cache_data.clear()
