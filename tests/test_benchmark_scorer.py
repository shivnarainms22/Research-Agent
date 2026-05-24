"""Tests for analysis/benchmark_scorer.py."""
from __future__ import annotations


def _measurement(measured, status="ok", expected=0.90, tol=0.05,
                 ttype="relative", difficulty="medium", source="arxiv"):
    from core.models import BenchmarkItem
    from analysis.benchmark_scorer import Measurement
    item = BenchmarkItem(
        id="i1", experiment_id="e1", metric_name="accuracy",
        expected_value=expected, tolerance=tol, tolerance_type=ttype,
    )
    return Measurement(item=item, measured=measured, difficulty=difficulty,
                       source=source, status=status)


def test_score_relative_pass_within_band():
    from analysis.benchmark_scorer import score
    out = score([_measurement(0.93, expected=0.90, tol=0.05)])  # band = ±0.045
    assert out[0].status == "pass" and out[0].passed is True


def test_score_relative_fail_outside_band():
    from analysis.benchmark_scorer import score
    out = score([_measurement(0.80, expected=0.90, tol=0.05)])
    assert out[0].status == "fail" and out[0].passed is False


def test_score_absolute_mode():
    from analysis.benchmark_scorer import score
    out = score([_measurement(0.62, expected=0.60, tol=0.05, ttype="absolute")])
    assert out[0].passed is True  # |0.62-0.60|=0.02 <= 0.05


def test_score_expected_zero_falls_back_to_absolute():
    from analysis.benchmark_scorer import score
    out = score([_measurement(0.03, expected=0.0, tol=0.05, ttype="relative")])
    assert out[0].passed is True  # |0.03-0|=0.03 <= 0.05 (absolute fallback)


def test_score_carries_unscorable_status():
    from analysis.benchmark_scorer import score
    for st in ("no_result", "missing_metric", "non_numeric"):
        out = score([_measurement(None, status=st)])
        assert out[0].status == st and out[0].passed is None and out[0].measured_value is None
