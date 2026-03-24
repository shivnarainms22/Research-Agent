"""Settings page — view and edit domain.yaml (keywords, categories, thresholds)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st
import yaml

_DOMAIN_YAML = Path(__file__).resolve().parents[2] / "domain.yaml"


def _load() -> dict:
    try:
        return yaml.safe_load(_DOMAIN_YAML.read_text(encoding="utf-8")) or {}
    except Exception as e:
        st.error(f"Could not read domain.yaml: {e}")
        return {}


def _save(data: dict) -> None:
    _DOMAIN_YAML.write_text(yaml.dump(data, default_flow_style=False, allow_unicode=True), encoding="utf-8")


def render() -> None:
    st.title("Settings — Domain Configuration")
    st.caption(f"Editing `{_DOMAIN_YAML}` — changes take effect on the next pipeline run.")

    data = _load()
    if not data:
        return

    # ── Keywords ──────────────────────────────────────────────────────────────
    st.subheader("Keywords")
    st.markdown("""
Papers are filtered by these keywords at two points:
- **arXiv fetch-time**: first 8 keywords are included in the arXiv query (OR-combined within your categories)
- **Semantic Scholar**: first 6 keywords searched individually
- **Synthesis gate**: papers must match ≥ `min_keyword_matches_to_analyze` keywords in title+abstract before Claude analyzes them

**Tip:** If no papers are reaching Claude analysis, either add broader keywords or lower the match threshold below.
""")

    current_keywords = data.get("keywords", [])
    kw_text = st.text_area(
        "Keywords (one per line)",
        value="\n".join(current_keywords),
        height=300,
        help="One keyword or phrase per line. Case-insensitive. arXiv uses the first 8 at fetch-time.",
    )

    if st.button("Save Keywords", type="primary"):
        new_keywords = [k.strip() for k in kw_text.splitlines() if k.strip()]
        if not new_keywords:
            st.error("Must have at least one keyword.")
        else:
            data["keywords"] = new_keywords
            _save(data)
            st.success(f"Saved {len(new_keywords)} keywords. Takes effect on next pipeline run.")
            if len(new_keywords) > 8:
                st.info(f"Note: only the first 8 keywords are used in the arXiv query. All {len(new_keywords)} apply at synthesis time.")

    st.divider()

    # ── arXiv Categories ──────────────────────────────────────────────────────
    st.subheader("arXiv Categories")
    st.markdown("Papers are only fetched from these arXiv categories. See [arXiv category taxonomy](https://arxiv.org/category_taxonomy) for valid codes.")

    raw_cats = data.get("arxiv_categories", [])
    # handle both list and inline sequence formats from yaml
    if isinstance(raw_cats, list):
        cats_str = "\n".join(raw_cats)
    else:
        cats_str = str(raw_cats)

    cat_text = st.text_area(
        "Categories (one per line)",
        value=cats_str,
        height=150,
        help="e.g. cs.LG, cs.AI, cs.CL, cs.CV, cs.RO, stat.ML",
    )

    if st.button("Save Categories", type="primary"):
        new_cats = [c.strip() for c in cat_text.splitlines() if c.strip()]
        if not new_cats:
            st.error("Must have at least one category.")
        else:
            data["arxiv_categories"] = new_cats
            _save(data)
            st.success(f"Saved {len(new_cats)} categories.")

    st.divider()

    # ── Thresholds ────────────────────────────────────────────────────────────
    st.subheader("Thresholds")

    thresholds = data.get("thresholds", {})

    col1, col2 = st.columns(2)

    with col1:
        min_matches = st.number_input(
            "Min keyword matches to analyze",
            min_value=1,
            max_value=10,
            value=int(thresholds.get("min_keyword_matches_to_analyze", 2)),
            step=1,
            help="Papers must match this many keywords in title+abstract before Claude analyzes them. "
                 "Default is 2. Set to 1 to let more papers through.",
        )
        st.caption("**Lower = more papers reach Claude.** If you're getting 0 Claude calls, set this to 1.")

    with col2:
        novelty = st.slider(
            "Min novelty score to generate experiment",
            min_value=5.0,
            max_value=9.5,
            value=float(thresholds.get("min_novelty_score_to_experiment", 7.5)),
            step=0.5,
            help="Claude scores novelty 1–10. Papers above this threshold get experiment hypotheses generated.",
        )
        relevance = st.slider(
            "Min relevance score to generate experiment",
            min_value=5.0,
            max_value=9.5,
            value=float(thresholds.get("min_relevance_score_to_experiment", 7.0)),
            step=0.5,
            help="Claude scores relevance 1–10. Both novelty AND relevance must exceed their thresholds.",
        )

    if st.button("Save Thresholds", type="primary"):
        data["thresholds"] = {
            "min_keyword_matches_to_analyze": int(min_matches),
            "min_novelty_score_to_experiment": float(novelty),
            "min_relevance_score_to_experiment": float(relevance),
        }
        _save(data)
        st.success("Thresholds saved. Takes effect on next pipeline run.")

    st.divider()

    # ── Live preview ──────────────────────────────────────────────────────────
    with st.expander("Preview current domain.yaml"):
        st.code(_DOMAIN_YAML.read_text(encoding="utf-8"), language="yaml")
