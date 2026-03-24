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

st.markdown("""
<style>
/* ── Sidebar shell ── */
[data-testid="stSidebar"] {
    background-color: #080808;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 0;
}

/* ── Hide the "Navigate" label (collapsed) ── */
[data-testid="stSidebar"] .stRadio > label {
    display: none !important;
}

/* ── Radio group container ── */
[data-testid="stSidebar"] .stRadio > div {
    gap: 2px !important;
    padding: 8px 10px;
}

/* ── Each nav item wrapper ── */
[data-testid="stSidebar"] .stRadio > div > label {
    display: flex !important;
    align-items: center;
    padding: 9px 14px !important;
    border-radius: 8px !important;
    cursor: pointer;
    transition: background 0.15s ease;
    color: rgba(255,255,255,0.45) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em;
    white-space: nowrap;
    border: 1px solid transparent !important;
}
[data-testid="stSidebar"] .stRadio > div > label:hover {
    background: rgba(255,255,255,0.06) !important;
    color: rgba(255,255,255,0.75) !important;
}

/* ── Active nav item ── */
[data-testid="stSidebar"] .stRadio > div > label:has(input:checked) {
    background: rgba(255,255,255,0.10) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}

/* ── Hide radio circles ── */
[data-testid="stSidebar"] .stRadio input[type="radio"] {
    display: none !important;
}
[data-testid="stSidebar"] .stRadio > div > label > div:first-child {
    display: none !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background-color: #111111;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 18px 20px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(255,255,255,0.18);
    box-shadow: 0 4px 24px rgba(0,0,0,0.5);
}
[data-testid="stMetric"] label {
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.35) !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #ffffff !important;
    letter-spacing: -0.5px !important;
}

/* ── Buttons ── */
[data-testid="stButton"] > button {
    background-color: #ffffff !important;
    color: #000000 !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    letter-spacing: 0.01em;
    padding: 8px 18px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.4);
    transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s ease;
}
[data-testid="stButton"] > button:hover {
    background-color: #e8e8e8 !important;
    color: #000000 !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
}
[data-testid="stButton"] > button:active {
    transform: translateY(0px) scale(0.98);
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    background: #111111 !important;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(255,255,255,0.14) !important;
}

/* ── Dividers ── */
hr { border-color: rgba(255,255,255,0.07); margin: 24px 0; }

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div > div {
    background: #ffffff !important;
    border-radius: 4px;
}
[data-testid="stProgressBar"] > div {
    background: rgba(255,255,255,0.1) !important;
    border-radius: 4px;
    height: 6px !important;
}

/* ── Dataframe / table ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    overflow: hidden;
}

/* ── Code blocks ── */
[data-testid="stCode"] {
    background: #0d0d0d !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
}

/* ── Inputs / selects ── */
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color: #111111 !important;
    border-color: rgba(255,255,255,0.12) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
[data-baseweb="input"] input:focus,
[data-baseweb="textarea"] textarea:focus {
    border-color: rgba(255,255,255,0.45) !important;
    box-shadow: 0 0 0 2px rgba(255,255,255,0.08) !important;
}

/* ── Multiselect / select containers ── */
[data-baseweb="select"] > div {
    background-color: #111111 !important;
    border-color: rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
/* Selected tag pills */
[data-baseweb="tag"] {
    background-color: #2a2a2a !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 5px !important;
}
[data-baseweb="tag"] span {
    color: #ffffff !important;
}
[data-baseweb="tag"] button span {
    color: rgba(255,255,255,0.6) !important;
}
/* Dropdown list */
[data-baseweb="popover"] {
    background-color: #111111 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
}
[data-baseweb="menu"] li {
    background-color: #111111 !important;
    color: #ffffff !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="option"]:hover {
    background-color: rgba(255,255,255,0.08) !important;
}
/* Placeholder + typed search text */
[data-baseweb="select"] input {
    color: #ffffff !important;
}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] {
    color: #ffffff !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: rgba(255,255,255,0.4) !important;
    border-radius: 6px 6px 0 0 !important;
    padding: 8px 16px !important;
    transition: color 0.15s;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    border-bottom: 2px solid #ffffff !important;
}

/* ── Hide running spinner & top decoration bar ── */
[data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stDecoration"]   { display: none !important; }

/* ── Focus ring ── */
*:focus { outline-color: rgba(255,255,255,0.35) !important; }

/* ── Alerts / info / warning / success / error ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border-left-width: 3px !important;
}
</style>
""", unsafe_allow_html=True)

PAGE_LABELS = [
    "⊞  Dashboard",
    "＋  Add Paper",
    "✓  Review Queue",
    "◫  Papers",
    "⚗  Experiments",
    "◉  Living Review",
    "≡  Reports",
    "⚙  Settings",
]

# Sidebar — branded header
st.sidebar.markdown("""
<div style="padding:20px 16px 16px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:8px;">
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="width:32px;height:32px;background:#fff;border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;">🔬</div>
    <div>
      <div style="font-size:14px;font-weight:700;color:#fff;letter-spacing:-0.2px;">Research Agent</div>
      <div style="font-size:10px;color:rgba(255,255,255,0.3);margin-top:1px;">Autonomous Research</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Seed session state from URL (match by route name, strip icon prefix)
if "nav_page" not in st.session_state:
    _from_url = st.query_params.get("page", "Dashboard")
    st.session_state.nav_page = next(
        (lbl for lbl in PAGE_LABELS if lbl.split("  ", 1)[-1] == _from_url),
        PAGE_LABELS[0],
    )

page_label = st.sidebar.radio("Navigate", PAGE_LABELS, key="nav_page", label_visibility="collapsed")
page = page_label.split("  ", 1)[-1]   # e.g. "Dashboard", "Papers", ...

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
