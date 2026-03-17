"""Add Paper page — ingest a paper from PDF, text file, or URL."""
from __future__ import annotations

import streamlit as st


def render() -> None:
    st.title("Add Paper")
    st.caption(
        "Upload a PDF or text file, or paste any URL (arXiv, direct PDF link, or web page). "
        "The paper is saved to the database and you can synthesize it immediately."
    )

    tab_pdf, tab_text, tab_url = st.tabs(["PDF Upload", "Text File", "URL / Link"])

    with tab_pdf:
        uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
        if uploaded_pdf:
            if st.button("Ingest PDF", key="ingest_pdf"):
                _ingest_and_show(lambda: _from_pdf(uploaded_pdf))

    with tab_text:
        uploaded_txt = st.file_uploader("Upload a .txt file", type=["txt"], key="txt_up")
        if uploaded_txt:
            if st.button("Ingest Text File", key="ingest_txt"):
                _ingest_and_show(lambda: _from_text_file(uploaded_txt))

    with tab_url:
        url = st.text_input(
            "Paper URL",
            placeholder="https://arxiv.org/abs/2401.12345  or any PDF/web link",
        )
        if url:
            if st.button("Ingest URL", key="ingest_url"):
                _ingest_and_show(lambda: _from_url(url))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _from_pdf(uploaded) -> "Paper":
    from ingestion.manual_ingest import from_pdf_bytes
    return from_pdf_bytes(uploaded.read(), uploaded.name)


def _from_text_file(uploaded) -> "Paper":
    from ingestion.manual_ingest import from_text
    text = uploaded.read().decode("utf-8", errors="replace")
    return from_text(text, uploaded.name)


def _from_url(url: str) -> "Paper":
    from ingestion.manual_ingest import from_url
    return from_url(url)


def _ingest_and_show(fetch_fn) -> None:
    """Fetch → deduplicate → save → show summary card with inline Synthesize."""
    from sqlmodel import Session
    from core.database import get_engine
    from knowledge.paper_store import get_paper

    with st.spinner("Fetching and parsing…"):
        try:
            paper = fetch_fn()
        except Exception as e:
            st.error(f"Ingestion failed: {e}")
            return

    existing = get_paper(paper.id)
    if existing:
        st.info(f"Already in database: **{existing.title}**")
        paper = existing
    else:
        with Session(get_engine(), expire_on_commit=False) as session:
            session.add(paper)
            session.commit()
        st.success(f"Saved: **{paper.title}**")
        st.cache_data.clear()

    # Summary card
    cols = st.columns(3)
    cols[0].metric("Source", paper.source)
    cols[1].metric("Status", paper.status)
    cols[2].metric("Full text", f"{len(paper.full_text or ''):,} chars")
    st.markdown(f"**Abstract preview:** {paper.abstract[:400]}{'…' if len(paper.abstract) > 400 else ''}")

    if paper.url:
        st.link_button("Open paper", paper.url)

    st.divider()
    if st.button("Synthesize now →", key=f"synth_add_{paper.id}"):
        from ui.views.papers import _run_synthesis
        _run_synthesis(paper)
