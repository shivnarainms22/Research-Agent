"""Tests for synthesis/experiment_extractor.py status gating."""
from __future__ import annotations

from config import settings
from synthesis.experiment_extractor import _initial_status


def test_default_is_pending_review(monkeypatch):
    monkeypatch.setattr(settings, "auto_approve_cpu_experiments", False)
    assert _initial_status("cpu_only", "local") == "pending_review"
    assert _initial_status("gpu_large", "cloud_modal") == "pending_review"


def test_auto_approve_only_cpu_local(monkeypatch):
    monkeypatch.setattr(settings, "auto_approve_cpu_experiments", True)
    assert _initial_status("cpu_only", "local") == "pending"
    # GPU / cloud always stay gated even when auto-approve is on
    assert _initial_status("gpu_large", "cloud_modal") == "pending_review"
    assert _initial_status("cpu_only", "cloud_modal") == "pending_review"
    assert _initial_status("gpu_small", "local") == "pending_review"
