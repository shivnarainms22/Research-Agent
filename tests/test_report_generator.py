"""Smoke test for reporting/report_generator.generate on an empty DB.

No Claude call happens (narrative is skipped when there is nothing to report),
so this exercises section building, template rendering, and persistence.
"""
from __future__ import annotations

from datetime import datetime

from core.models import RunState


def test_generate_empty_db_renders_and_saves(in_memory_engine, tmp_path, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "data_dir", tmp_path)

    from reporting.report_generator import generate

    state = RunState(cycle_id="test_report", started_at=datetime.utcnow())
    report = generate(state)

    assert report.cycle_id == "test_report"
    report_file = tmp_path / "reports" / "test_report_weekly.md"
    assert report_file.exists()
    assert "Weekly Research Digest" in report_file.read_text(encoding="utf-8")
