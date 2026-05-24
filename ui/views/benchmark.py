"""Benchmark page — curate the golden set, run scoring, view accuracy trend + per-item results."""
from __future__ import annotations

import json
import uuid

import streamlit as st


@st.cache_data(ttl=8)
def _completed_experiments() -> list[dict]:
    """Completed experiments with their paper title and available metric keys."""
    from knowledge.experiment_store import get_experiments_by_status, get_result
    from knowledge.paper_store import get_paper
    rows = []
    for exp in get_experiments_by_status("completed"):
        result = get_result(exp.id)
        metric_keys, measured = [], {}
        if result and result.metrics:
            try:
                m = json.loads(result.metrics)
                metric_keys = [k for k, v in m.items() if isinstance(v, (int, float, list))]
                measured = m
            except (json.JSONDecodeError, TypeError):
                pass
        paper = get_paper(exp.paper_id)
        rows.append({
            "experiment_id": exp.id, "title": exp.title,
            "paper_title": paper.title if paper else exp.paper_id,
            "metric_keys": metric_keys, "measured": measured,
        })
    return rows


def _add_item_form(experiments: list[dict]) -> None:
    st.subheader("Add a golden-set item")
    if not experiments:
        st.info("No completed experiments yet — run some experiments first.")
        return
    labels = {f"{e['title']} — {e['paper_title']}": e for e in experiments}
    choice = st.selectbox("Experiment", list(labels.keys()), key="bench_exp")
    exp = labels[choice]
    keys = exp["metric_keys"] or ["(no numeric metrics found)"]
    with st.form("bench_add", clear_on_submit=True):
        metric_name = st.selectbox("Metric key", keys)
        if metric_name in exp["measured"]:
            st.caption(f"Latest measured value: {exp['measured'][metric_name]}")
        c1, c2 = st.columns(2)
        expected = c1.number_input("Expected value", value=0.0, format="%.6f")
        tolerance = c2.number_input("Tolerance", value=0.05, format="%.6f", min_value=0.0)
        ttype = st.radio("Tolerance type", ["relative", "absolute"], horizontal=True)
        unit = st.text_input("Unit (optional)")
        note = st.text_area("Note / provenance", placeholder="e.g. paper Table 2, top-1 accuracy")
        if st.form_submit_button("Add to golden set"):
            from core.models import BenchmarkItem
            from knowledge.benchmark_store import save_item
            save_item(BenchmarkItem(
                id=str(uuid.uuid4()), experiment_id=exp["experiment_id"],
                metric_name=metric_name, expected_value=float(expected),
                tolerance=float(tolerance), tolerance_type=ttype,
                unit=unit or None, note=note,
            ))
            st.success("Added.")
            st.cache_data.clear()
            st.rerun()


def _golden_set_table() -> None:
    import pandas as pd
    from knowledge.benchmark_store import get_items, deactivate_item
    items = get_items(active_only=True)
    st.subheader(f"Golden set ({len(items)} active)")
    if not items:
        st.caption("Empty — add items above.")
        return
    st.dataframe(pd.DataFrame([{
        "metric": i.metric_name, "expected": i.expected_value,
        "tol": f"{i.tolerance}{'%' if i.tolerance_type == 'relative' else ''}",
        "type": i.tolerance_type, "unit": i.unit or "", "note": i.note,
        "experiment_id": i.experiment_id,
    } for i in items]), use_container_width=True, hide_index=True)
    to_remove = st.selectbox("Deactivate item", ["—"] + [i.id for i in items], key="bench_rm")
    if to_remove != "—" and st.button("Deactivate"):
        deactivate_item(to_remove)
        st.cache_data.clear()
        st.rerun()


def _run_and_view() -> None:
    import pandas as pd
    from knowledge.benchmark_store import get_latest_run, get_item_results
    from knowledge.eval_metric_store import get_trend

    if st.button("▶ Run benchmark"):
        from analysis import benchmark_scorer
        with st.status("Scoring golden set…", expanded=False):
            run = benchmark_scorer.record_benchmark_run(trigger="manual")
        acc = "—" if run.accuracy is None else f"{run.accuracy * 100:.1f}%"
        st.success(f"Accuracy: {acc}  ·  pass {run.n_pass} / fail {run.n_fail} / unscorable {run.n_unscorable}")
        st.cache_data.clear()

    trend = get_trend("benchmark_accuracy", "overall", limit=12)
    pts = [{"cycle": r.cycle_id, "accuracy_pct": r.value * 100} for r in trend if r.value is not None]
    if pts:
        st.line_chart(pd.DataFrame(pts), x="cycle", y="accuracy_pct", height=200)

    latest = get_latest_run()
    if latest:
        st.caption(f"Latest run: {latest.recorded_at:%Y-%m-%d %H:%M} ({latest.trigger})")
        results = get_item_results(latest.id)
        if results:
            st.dataframe(pd.DataFrame([{
                "metric": r.metric_name, "expected": r.expected_value,
                "measured": r.measured_value, "status": r.status,
            } for r in results]), use_container_width=True, hide_index=True)


def render() -> None:
    st.title("Benchmark")
    st.caption("Curate a ground-truth golden set and track how accurately the pipeline reproduces known results.")
    experiments = _completed_experiments()
    _add_item_form(experiments)
    st.divider()
    _golden_set_table()
    st.divider()
    _run_and_view()
