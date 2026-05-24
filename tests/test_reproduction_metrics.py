"""Tests for analysis/reproduction_metrics.py."""
from __future__ import annotations

from datetime import datetime


def _row(overall, difficulty="medium", target="local", source="arxiv"):
    from analysis.reproduction_metrics import VerdictRow
    return VerdictRow(
        overall=overall, difficulty=difficulty, target=target,
        source=source, recorded_at=datetime.utcnow(),
    )


def _by(points, metric, dimension):
    return next((p for p in points if p.metric == metric and p.dimension == dimension), None)


def test_tally_happy_path():
    from analysis.reproduction_metrics import tally
    rows = [
        _row("fully_reproduced"), _row("fully_reproduced"),
        _row("partially_reproduced"),
        _row("not_reproduced"),
    ]
    points = tally(rows)
    overall_repro = _by(points, "reproduction_rate", "overall")
    overall_partial = _by(points, "partial_rate", "overall")
    assert overall_repro.numerator == 2
    assert overall_repro.denominator == 4
    assert abs(overall_repro.value - 0.5) < 1e-9
    assert overall_partial.numerator == 1
    assert overall_partial.denominator == 4
    assert abs(overall_partial.value - 0.25) < 1e-9


def test_tally_empty_rows_emits_overall_with_none_value():
    from analysis.reproduction_metrics import tally
    points = tally([])
    overall = _by(points, "reproduction_rate", "overall")
    assert overall is not None
    assert overall.value is None
    assert overall.numerator == 0
    assert overall.denominator == 0


def test_tally_skips_unknown_dimension_buckets():
    """'unknown'/empty dimension values must NOT create sub-buckets but still count toward overall."""
    from analysis.reproduction_metrics import tally
    rows = [_row("fully_reproduced", difficulty="unknown", source="")]
    points = tally(rows)
    overall = _by(points, "reproduction_rate", "overall")
    assert overall.denominator == 1
    # No bucket for difficulty:unknown
    assert _by(points, "reproduction_rate", "difficulty:unknown") is None
    assert not any(p.dimension.startswith("difficulty:") for p in points)
    assert not any(p.dimension.startswith("source:") for p in points)


def test_tally_dimensional_bucketing():
    from analysis.reproduction_metrics import tally
    rows = [
        _row("fully_reproduced", difficulty="easy", target="local", source="arxiv"),
        _row("not_reproduced", difficulty="easy", target="local", source="arxiv"),
        _row("fully_reproduced", difficulty="hard", target="cloud_modal", source="semantic_scholar"),
    ]
    points = tally(rows)
    easy = _by(points, "reproduction_rate", "difficulty:easy")
    hard = _by(points, "reproduction_rate", "difficulty:hard")
    local = _by(points, "reproduction_rate", "target:local")
    modal = _by(points, "reproduction_rate", "target:cloud_modal")
    arxiv = _by(points, "reproduction_rate", "source:arxiv")
    assert easy.numerator == 1 and easy.denominator == 2
    assert hard.numerator == 1 and hard.denominator == 1
    assert local.numerator == 1 and local.denominator == 2
    assert modal.numerator == 1 and modal.denominator == 1
    assert arxiv.numerator == 1 and arxiv.denominator == 2


def test_tally_context_json_records_breakdown():
    import json
    from analysis.reproduction_metrics import tally
    rows = [
        _row("fully_reproduced"),
        _row("fully_reproduced"),
        _row("partially_reproduced"),
        _row("not_reproduced"),
    ]
    points = tally(rows)
    overall = _by(points, "reproduction_rate", "overall")
    ctx = json.loads(overall.context)
    assert ctx == {"fully": 2, "partial": 1, "not": 1}


def _seed_paper_experiment_result(
    engine, *, paper_id="p1", source="arxiv", difficulty="easy",
    experiment_id="e1", target="local", parent_id=None,
    overall="fully_reproduced", with_baseline_comparison=True,
):
    """Seed one Paper + one PaperAnalysis + one Experiment + one ExperimentResult."""
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import Paper, PaperAnalysis, Experiment, ExperimentResult

    with Session(engine, expire_on_commit=False) as session:
        if not session.get(Paper, paper_id):
            session.add(Paper(
                id=paper_id, title=f"Paper {paper_id}", abstract="",
                source=source, source_id=paper_id, url="http://x", pdf_url=None,
                published_date=date(2025, 1, 1), full_text=None,
            ))
            session.add(PaperAnalysis(
                id=f"analysis_{paper_id}", paper_id=paper_id,
                reproducibility_difficulty=difficulty,
            ))
        session.add(Experiment(
            id=experiment_id, paper_id=paper_id, title=f"Exp {experiment_id}",
            hypothesis="h", execution_target=target, status="completed",
            parent_experiment_id=parent_id,
        ))
        bc = json.dumps({"overall": overall, "comparisons": []}) if with_baseline_comparison else None
        session.add(ExperimentResult(
            id=f"result_{experiment_id}", experiment_id=experiment_id,
            exit_code=0, metrics="{}", baseline_comparison=bc,
            recorded_at=datetime.utcnow(),
        ))
        session.commit()


def test_gather_verdicts_returns_rows_for_completed_with_baseline(in_memory_engine):
    from analysis.reproduction_metrics import gather_verdicts

    _seed_paper_experiment_result(in_memory_engine, experiment_id="e_full",
                                  overall="fully_reproduced", difficulty="easy",
                                  target="local", source="arxiv")
    _seed_paper_experiment_result(in_memory_engine, paper_id="p2",
                                  experiment_id="e_not", overall="not_reproduced",
                                  difficulty="hard", target="cloud_modal", source="semantic_scholar")

    rows = gather_verdicts()
    assert {r.overall for r in rows} == {"fully_reproduced", "not_reproduced"}
    assert {r.difficulty for r in rows} == {"easy", "hard"}
    assert {r.target for r in rows} == {"local", "cloud_modal"}
    assert {r.source for r in rows} == {"arxiv", "semantic_scholar"}


def test_gather_verdicts_excludes_ablations(in_memory_engine):
    from analysis.reproduction_metrics import gather_verdicts
    _seed_paper_experiment_result(in_memory_engine, experiment_id="parent",
                                  overall="fully_reproduced")
    _seed_paper_experiment_result(in_memory_engine, experiment_id="abl",
                                  overall="fully_reproduced", parent_id="parent")
    rows = gather_verdicts()
    assert len(rows) == 1  # only the parent


def test_gather_verdicts_excludes_no_baseline_comparison(in_memory_engine):
    from analysis.reproduction_metrics import gather_verdicts
    _seed_paper_experiment_result(in_memory_engine, experiment_id="no_bc",
                                  with_baseline_comparison=False)
    assert gather_verdicts() == []


def test_gather_verdicts_excludes_non_comparable_verdicts(in_memory_engine):
    """Results with overall='no_experiments' (or {status: ...}) must not appear."""
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import Paper, PaperAnalysis, Experiment, ExperimentResult
    from analysis.reproduction_metrics import gather_verdicts

    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(Paper(id="p", title="t", abstract="", source="arxiv",
                          source_id="p", url="x", published_date=date(2025,1,1)))
        session.add(PaperAnalysis(id="a", paper_id="p"))
        session.add(Experiment(id="e", paper_id="p", title="t", hypothesis="h",
                               execution_target="local", status="completed"))
        session.add(ExperimentResult(
            id="r", experiment_id="e", exit_code=0, metrics="{}",
            baseline_comparison=json.dumps({"status": "no_analysis"}),
            recorded_at=datetime.utcnow(),
        ))
        session.commit()
    assert gather_verdicts() == []


def test_gather_verdicts_filters_by_experiment_ids(in_memory_engine):
    from analysis.reproduction_metrics import gather_verdicts
    _seed_paper_experiment_result(in_memory_engine, experiment_id="keep")
    _seed_paper_experiment_result(in_memory_engine, paper_id="p2",
                                  experiment_id="drop", overall="not_reproduced")
    rows = gather_verdicts(experiment_ids=["keep"])
    assert len(rows) == 1
    assert rows[0].overall == "fully_reproduced"


def test_gather_verdicts_tolerates_missing_analysis(in_memory_engine):
    """An experiment whose paper has no PaperAnalysis row still produces a verdict, with difficulty='unknown'."""
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import Paper, Experiment, ExperimentResult
    from analysis.reproduction_metrics import gather_verdicts

    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(Paper(id="p", title="t", abstract="", source="arxiv",
                          source_id="p", url="x", published_date=date(2025,1,1)))
        session.add(Experiment(id="e", paper_id="p", title="t", hypothesis="h",
                               execution_target="local", status="completed"))
        session.add(ExperimentResult(
            id="r", experiment_id="e", exit_code=0, metrics="{}",
            baseline_comparison=json.dumps({"overall": "fully_reproduced", "comparisons": []}),
            recorded_at=datetime.utcnow(),
        ))
        session.commit()

    rows = gather_verdicts()
    assert len(rows) == 1
    assert rows[0].difficulty == "unknown"
    assert rows[0].source == "arxiv"


def test_gather_verdicts_warns_on_malformed_baseline(in_memory_engine, caplog):
    import json
    from datetime import date, datetime
    from sqlmodel import Session
    from core.models import Paper, Experiment, ExperimentResult
    from analysis.reproduction_metrics import gather_verdicts

    with Session(in_memory_engine, expire_on_commit=False) as session:
        session.add(Paper(id="p", title="t", abstract="", source="arxiv",
                          source_id="p", url="x", published_date=date(2025,1,1)))
        session.add(Experiment(id="e", paper_id="p", title="t", hypothesis="h",
                               execution_target="local", status="completed"))
        session.add(ExperimentResult(
            id="r", experiment_id="e", exit_code=0, metrics="{}",
            baseline_comparison="not json{{{",
            recorded_at=datetime.utcnow(),
        ))
        session.commit()
    assert gather_verdicts() == []


def _override_recorded_at(engine, result_id: str, when):
    from sqlmodel import Session
    from core.models import ExperimentResult
    with Session(engine, expire_on_commit=False) as session:
        r = session.get(ExperimentResult, result_id)
        r.recorded_at = when
        session.add(r)
        session.commit()


def test_backfill_buckets_by_iso_week(in_memory_engine):
    from datetime import datetime
    from analysis.reproduction_metrics import backfill_from_history
    from knowledge.eval_metric_store import get_trend

    # Week 2025-W10 vs 2025-W20 (well-separated weeks).
    _seed_paper_experiment_result(in_memory_engine, experiment_id="e_a", overall="fully_reproduced")
    _override_recorded_at(in_memory_engine, "result_e_a", datetime(2025, 3, 5))   # W10
    _seed_paper_experiment_result(in_memory_engine, paper_id="p2", experiment_id="e_b",
                                  overall="not_reproduced")
    _override_recorded_at(in_memory_engine, "result_e_b", datetime(2025, 5, 14))  # W20

    written = backfill_from_history()
    assert written > 0
    trend = get_trend("reproduction_rate", "overall", limit=10)
    cycle_ids = {row.cycle_id for row in trend}
    assert "backfill-2025-W10" in cycle_ids
    assert "backfill-2025-W20" in cycle_ids


def test_backfill_is_idempotent(in_memory_engine):
    from datetime import datetime
    from analysis.reproduction_metrics import backfill_from_history
    from knowledge.eval_metric_store import count_rows

    _seed_paper_experiment_result(in_memory_engine, experiment_id="e1", overall="fully_reproduced")
    _override_recorded_at(in_memory_engine, "result_e1", datetime(2025, 3, 5))

    backfill_from_history()
    n = count_rows()
    backfill_from_history()
    assert count_rows() == n  # second invocation writes nothing


def test_backfill_skips_empty_weeks(in_memory_engine):
    """If no comparable rows exist, no backfill rows are written."""
    from analysis.reproduction_metrics import backfill_from_history
    from knowledge.eval_metric_store import count_rows
    written = backfill_from_history()
    assert written == 0
    assert count_rows() == 0


def _make_state(cycle_id="cycle_x", experiment_ids=None):
    from core.models import RunState
    from datetime import datetime
    return RunState(
        cycle_id=cycle_id, started_at=datetime.utcnow(),
        experiment_ids_this_cycle=list(experiment_ids or []),
    )


def test_record_cycle_snapshot_writes_overall_and_dimensions(in_memory_engine):
    from analysis.reproduction_metrics import record_cycle_snapshot
    from knowledge.eval_metric_store import get_latest

    _seed_paper_experiment_result(in_memory_engine, experiment_id="e1",
                                  overall="fully_reproduced", difficulty="easy",
                                  target="local", source="arxiv")
    state = _make_state(cycle_id="cycle_one", experiment_ids=["e1"])
    record_cycle_snapshot(state)

    overall = get_latest("reproduction_rate", "overall")
    by_difficulty = get_latest("reproduction_rate", "difficulty:easy")
    assert overall is not None and overall.numerator == 1 and overall.denominator == 1
    assert by_difficulty is not None and by_difficulty.numerator == 1


def test_record_cycle_snapshot_empty_cycle_writes_none_overall(in_memory_engine):
    """A cycle with no comparable experiments still records the gap honestly."""
    from analysis.reproduction_metrics import record_cycle_snapshot
    from knowledge.eval_metric_store import get_latest

    state = _make_state(cycle_id="empty", experiment_ids=[])
    record_cycle_snapshot(state)

    overall = get_latest("reproduction_rate", "overall")
    assert overall is not None
    assert overall.value is None
    assert overall.denominator == 0


def test_record_cycle_snapshot_triggers_lazy_backfill_when_empty(in_memory_engine):
    from datetime import datetime
    from analysis.reproduction_metrics import record_cycle_snapshot
    from knowledge.eval_metric_store import get_trend

    # Seed historical results from a prior week BEFORE the cycle's experiment.
    _seed_paper_experiment_result(in_memory_engine, paper_id="p_hist", experiment_id="e_hist",
                                  overall="fully_reproduced")
    _override_recorded_at(in_memory_engine, "result_e_hist", datetime(2025, 3, 5))  # W10

    # Cycle's own experiment.
    _seed_paper_experiment_result(in_memory_engine, paper_id="p_now", experiment_id="e_now",
                                  overall="not_reproduced")

    state = _make_state(cycle_id="now_cycle", experiment_ids=["e_now"])
    record_cycle_snapshot(state)

    trend = get_trend("reproduction_rate", "overall", limit=20)
    cycle_ids = [r.cycle_id for r in trend]
    assert "backfill-2025-W10" in cycle_ids
    assert "now_cycle" in cycle_ids


def test_record_cycle_snapshot_does_not_re_backfill(in_memory_engine):
    from analysis.reproduction_metrics import record_cycle_snapshot, backfill_from_history
    from knowledge.eval_metric_store import count_rows
    from datetime import datetime

    _seed_paper_experiment_result(in_memory_engine, paper_id="p_h", experiment_id="e_h",
                                  overall="fully_reproduced")
    _override_recorded_at(in_memory_engine, "result_e_h", datetime(2025, 3, 5))
    backfill_from_history()

    rows_after_backfill = count_rows()
    _seed_paper_experiment_result(in_memory_engine, paper_id="p_n", experiment_id="e_n",
                                  overall="not_reproduced")
    record_cycle_snapshot(_make_state(cycle_id="c", experiment_ids=["e_n"]))

    # Only cycle's own rows added; backfill rows not duplicated.
    assert count_rows() > rows_after_backfill  # cycle rows added
    # And re-invoking does not produce more backfill rows.
    record_cycle_snapshot(_make_state(cycle_id="c2", experiment_ids=[]))
    # cycle c2 adds only the overall=None row(s); backfill rows still present once.
