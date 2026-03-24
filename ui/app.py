"""Research Agent — Streamlit web UI entry point."""
import os
import sys

os.environ.setdefault("PYTHONUTF8", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import streamlit as st

st.set_page_config(layout="wide", page_title="Research Agent", page_icon="🔬")


@st.cache_resource
def _init_db():
    from core.database import init_db, init_chroma
    init_db()
    init_chroma()


_init_db()

PAGES = [
    "Dashboard",
    "Add Paper",
    "Review Queue",
    "Papers",
    "Experiments",
    "Living Review",
    "Reports",
    "Settings",
]

if "nav_page" not in st.session_state:
    _from_url = st.query_params.get("page", "Dashboard")
    st.session_state.nav_page = _from_url if _from_url in PAGES else "Dashboard"
page = st.sidebar.radio("Navigate", PAGES, key="nav_page")
if st.query_params.get("page") != page:
    st.query_params["page"] = page

if page == "Dashboard":
    from ui.views.dashboard import render
    render()
elif page == "Add Paper":
    from ui.views.add_paper import render
    render()
elif page == "Review Queue":
    from ui.views.review import render
    render()
elif page == "Papers":
    from ui.views.papers import render
    render()
elif page == "Experiments":
    from ui.views.experiments import render
    render()
elif page == "Living Review":
    from ui.views.living_review import render
    render()
elif page == "Reports":
    from ui.views.reports import render
    render()
elif page == "Settings":
    from ui.views.settings import render
    render()
