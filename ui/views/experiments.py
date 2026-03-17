"""Experiment Browser page — status filter, result drill-down."""
from __future__ import annotations

import json

import streamlit as st
import pandas as pd

from core.models import parse_json_list, parse_json_dict
from knowledge.experiment_store import get_all_experiments, get_result
from knowledge.paper_store import get_paper


@st.cache_data(ttl=10)
def _get_all_experiments():
    return get_all_experiments(limit=1000)


BASELINE_LABELS = {
    "reproduced": "✓ Reproduced",
    "partial": "~ Partial",
    "failed": "✗ Failed",
    "not_applicable": "? N/A",
}


def render() -> None:
    st.title("Experiment Browser")

    experiments = _get_all_experiments()

    all_statuses = sorted({e.status for e in experiments})
    all_targets = sorted({e.execution_target for e in experiments})

    col1, col2 = st.columns(2)
    selected_statuses = col1.multiselect("Status filter", all_statuses, default=all_statuses)
    selected_targets = col2.multiselect("Target filter", all_targets, default=all_targets)

    # Paper ID filter from session state (set by papers page)
    paper_filter = st.session_state.get("exp_filter_paper_id")
    if paper_filter:
        st.info(f"Filtering by paper_id: `{paper_filter[:16]}…`")
        if st.button("Clear paper filter"):
            st.session_state.pop("exp_filter_paper_id", None)
            st.rerun()

    filtered = [
        e for e in experiments
        if e.status in selected_statuses
        and e.execution_target in selected_targets
        and (not paper_filter or e.paper_id == paper_filter)
    ]

    if not filtered:
        st.info("No experiments match the current filters.")
        return

    # Build table data
    rows = []
    for exp in filtered:
        paper = get_paper(exp.paper_id)
        result = get_result(exp.id)
        baseline_status = "?"
        if result and result.baseline_comparison:
            bc = parse_json_dict(result.baseline_comparison)
            baseline_status = bc.get("status", "?")
        rows.append({
            "id": exp.id[:8],
            "title": exp.title[:60],
            "paper": (paper.title[:40] if paper else exp.paper_id[:16]),
            "status": exp.status,
            "target": exp.execution_target,
            "runtime_s": f"{result.runtime_seconds:.1f}" if result else "—",
            "baseline": BASELINE_LABELS.get(baseline_status, baseline_status),
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Detail View")

    exp_options = {f"{e.title[:50]} [{e.id[:8]}]": e.id for e in filtered}
    selected_label = st.selectbox("Select experiment", list(exp_options.keys()))
    if not selected_label:
        return

    selected_id = exp_options[selected_label]
    exp = next((e for e in filtered if e.id == selected_id), None)
    if not exp:
        return

    result = get_result(exp.id)

    st.markdown(f"**Hypothesis:** {exp.hypothesis}")
    st.markdown(f"**Status:** `{exp.status}` | **Target:** `{exp.execution_target}`")
    if exp.error_message:
        st.error(f"Error: {exp.error_message}")

    if exp.status in ("pending", "failed"):
        if st.button("▶ Run experiment", key=f"run_{exp.id}"):
            _run_experiment(exp)

    if result:
        metrics = parse_json_dict(result.metrics)
        if metrics:
            st.markdown("**Metrics:**")
            metrics_rows = [{"metric": k, "value": v} for k, v in metrics.items()
                            if len(k) > 1 and not k.isdigit()]
            if metrics_rows:
                st.table(pd.DataFrame(metrics_rows))

        if result.statistical_summary:
            stat = parse_json_dict(result.statistical_summary)
            if stat:
                st.markdown("**Statistical Summary:**")
                st.json(stat)

        if result.baseline_comparison:
            bc = parse_json_dict(result.baseline_comparison)
            status = bc.get("status", "?")
            label = BASELINE_LABELS.get(status, status)
            st.markdown(f"**Baseline Comparison:** {label}")

        if result.conclusion:
            st.markdown(f"**Conclusion:** {result.conclusion}")

        with st.expander("stdout", expanded=False):
            st.code(result.stdout or "(empty)", language="text")


def _run_experiment(exp) -> None:
    from experiments import code_validator, local_runner, cloud_runner, router
    from knowledge.experiment_store import (
        update_experiment_status, save_result, get_result, delete_result,
    )
    from experiments.result_collector import parse_metrics_from_stdout
    import json

    with st.status(f"Running: {exp.title[:60]}…", expanded=True) as status_box:
        # Validate
        st.write("Validating code…")
        validated_code, ok = code_validator.validate_with_retry(exp.generated_code, exp.paper_id)
        if not ok:
            st.error("Code validation failed — experiment skipped.")
            update_experiment_status(exp.id, "skipped", error="code validation failed")
            status_box.update(label="Validation failed", state="error")
            return
        exp.generated_code = validated_code

        # Route
        target = router.decide_target(exp)
        exp.execution_target = target
        st.write(f"Target: **{target}**")

        update_experiment_status(exp.id, "running")
        st.write("Running…")
        try:
            result = local_runner.run(exp) if target == "local" else cloud_runner.run(exp)

            if result.metrics == "{}" and result.stdout:
                fallback = parse_metrics_from_stdout(result.stdout)
                if fallback:
                    result.metrics = json.dumps(fallback)

            if get_result(exp.id):
                delete_result(exp.id)
            save_result(result)

            if result.exit_code == 0 and result.metrics != "{}":
                update_experiment_status(exp.id, "completed")
                status_box.update(label=f"Completed — exit {result.exit_code}", state="complete")
                st.write(f"Metrics: {result.metrics}")
            else:
                err = "no metrics" if result.exit_code == 0 else f"exit_code={result.exit_code}"
                update_experiment_status(exp.id, "failed", error=err)
                status_box.update(label=f"Failed — {err}", state="error")
                if result.stdout:
                    st.code(result.stdout[-2000:], language="text")

        except Exception as e:
            update_experiment_status(exp.id, "failed", error=str(e))
            st.error(str(e))
            status_box.update(label="Error", state="error")

        st.cache_data.clear()
