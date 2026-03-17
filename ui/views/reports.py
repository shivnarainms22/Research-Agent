"""Reports page — browse and render Markdown research reports."""
from __future__ import annotations

import streamlit as st

from config import settings


def _get_report_files() -> list:
    if not settings.reports_dir.exists():
        return []
    return sorted(settings.reports_dir.glob("*.md"), key=lambda f: f.stat().st_mtime, reverse=True)


def render() -> None:
    st.title("Research Reports")

    report_files = _get_report_files()
    if not report_files:
        st.info("No reports found. Run the pipeline to generate a report.")
        return

    options = {f.name: f for f in report_files}
    selected_name = st.selectbox("Select report", list(options.keys()))
    if not selected_name:
        return

    selected_file = options[selected_name]
    content = selected_file.read_text(encoding="utf-8")

    col1, col2 = st.columns([4, 1])
    col1.markdown(f"**{selected_name}** — {len(content):,} chars")
    col2.download_button(
        label="Download .md",
        data=content.encode("utf-8"),
        file_name=selected_name,
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()
    st.markdown(content)
