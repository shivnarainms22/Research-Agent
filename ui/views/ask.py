"""Ask-the-corpus page — natural-language Q&A over analyzed papers."""
from __future__ import annotations

import json

import streamlit as st

from config import settings
from core.models import parse_json_list
from knowledge import paper_store, retriever


def _gather_context(question: str, n: int = 6) -> tuple[str, list[dict]]:
    """Retrieve relevant papers and build a context block + source list."""
    papers = retriever.search(question, n=n)
    blocks, sources = [], []
    for i, paper in enumerate(papers, start=1):
        analysis = paper_store.get_analysis(paper.id)
        contributions = parse_json_list(analysis.key_contributions) if analysis else []
        block = f"[{i}] {paper.title}"
        if contributions:
            block += "\n   " + "; ".join(contributions[:3])
        blocks.append(block)
        sources.append({"n": i, "title": paper.title, "url": paper.url})
    return "\n".join(blocks), sources


def answer_question(question: str) -> tuple[str, list[dict]]:
    """Answer a question grounded in the corpus. Returns (answer, sources)."""
    context, sources = _gather_context(question)
    if not sources:
        return "No relevant papers found in the corpus yet.", []

    import anthropic
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    resp = client.messages.create(
        model=settings.claude_model,
        max_tokens=1024,
        temperature=0.3,
        system=[{
            "type": "text",
            "text": ("You answer questions using only the provided paper summaries. "
                     "Cite sources inline as [n]. If the papers don't cover it, say so."),
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": f"Papers:\n{context}\n\nQuestion: {question}"}],
    )
    from core import token_tracker
    token_tracker.track("ask_corpus", resp.usage.input_tokens, resp.usage.output_tokens)
    answer = resp.content[0].text if resp.content else ""
    return answer, sources


def render() -> None:
    st.markdown("## Ask the Corpus")
    st.caption("Natural-language questions answered from your analyzed papers.")

    question = st.text_input("Question", placeholder="e.g. What have we learned about steering vectors?")
    if st.button("Ask", use_container_width=False) and question.strip():
        with st.spinner("Searching the corpus…"):
            try:
                answer, sources = answer_question(question.strip())
            except Exception as e:
                st.error(f"Failed to answer: {e}")
                return
        st.markdown(answer)
        if sources:
            st.divider()
            st.caption("Sources")
            for s in sources:
                st.markdown(f"**[{s['n']}]** [{s['title']}]({s['url']})")
