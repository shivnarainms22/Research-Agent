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
