"""Tests for core/token_tracker.py"""
from __future__ import annotations

import threading

import pytest


def setup_function():
    """Reset tracker before each test."""
    from core import token_tracker
    token_tracker.reset()
    token_tracker.set_cycle_id("")


def test_track_accumulates():
    from core import token_tracker
    token_tracker.track("module_a", 100, 50)
    token_tracker.track("module_a", 200, 75)
    totals = token_tracker.get_totals()
    assert totals["input_total"] == 300
    assert totals["output_total"] == 125
    assert totals["by_module"]["module_a"]["input"] == 300


def test_reset_clears_totals():
    from core import token_tracker
    token_tracker.track("module_b", 500, 300)
    token_tracker.reset()
    totals = token_tracker.get_totals()
    assert totals["input_total"] == 0
    assert totals["output_total"] == 0
    assert totals["by_module"] == {}


def test_set_cycle_id_and_no_db_error(monkeypatch):
    """set_cycle_id should work even when DB is unavailable (save_log raises)."""
    from core import token_tracker

    def _fail_save(*args, **kwargs):
        raise RuntimeError("DB unavailable")

    monkeypatch.setattr("knowledge.token_log_store.save_log", _fail_save)
    token_tracker.set_cycle_id("test_cycle_001")
    # Should not raise
    token_tracker.track("some_module", 10, 5)
    totals = token_tracker.get_totals()
    assert totals["input_total"] == 10


def test_thread_safety():
    """Concurrent track() calls should not lose counts."""
    from core import token_tracker

    n_threads = 20
    calls_per_thread = 100

    def _track():
        for _ in range(calls_per_thread):
            token_tracker.track("concurrent_module", 1, 1)

    threads = [threading.Thread(target=_track) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    totals = token_tracker.get_totals()
    expected = n_threads * calls_per_thread
    assert totals["input_total"] == expected
    assert totals["output_total"] == expected
