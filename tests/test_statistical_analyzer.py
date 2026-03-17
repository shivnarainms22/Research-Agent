"""Tests for analysis/statistical_analyzer.py"""
from __future__ import annotations

import pytest
from analysis.statistical_analyzer import analyze, compare_groups, _describe
import numpy as np


def test_analyze_single_float():
    metrics = {"accuracy": 0.95}
    result = analyze(metrics)
    assert result["accuracy"] == {"value": 0.95}


def test_analyze_list_of_floats():
    metrics = {"loss": [0.5, 0.4, 0.3, 0.2, 0.1]}
    result = analyze(metrics)
    assert "loss" in result
    assert "mean" in result["loss"]
    assert "std" in result["loss"]
    assert result["loss"]["n"] == 5


def test_analyze_list_skips_non_numeric():
    """Lists with string values should skip non-numeric entries gracefully."""
    metrics = {"labels": ["Vehicle", "Person", 0.9, 0.8]}
    result = analyze(metrics)
    # Only 2 numeric values — should compute stats on [0.9, 0.8]
    assert "labels" in result
    assert result["labels"]["n"] == 2


def test_analyze_list_insufficient_numeric_skipped():
    """If fewer than 2 numeric values, the key should be absent."""
    metrics = {"labels": ["Vehicle", "Person", "Truck"]}
    result = analyze(metrics)
    assert "labels" not in result


def test_analyze_nested_dict_passthrough():
    metrics = {"nested": {"key": "val", "x": 1.0}}
    result = analyze(metrics)
    assert result["nested"] == {"key": "val", "x": 1.0}


def test_compare_groups_significant():
    a = [1.0, 1.1, 0.9, 1.0, 1.05]
    b = [2.0, 2.1, 1.9, 2.0, 2.05]
    result = compare_groups(a, b)
    assert result["significant_at_05"] is True
    assert result["p_value"] < 0.05
    assert result["cohens_d"] < 0  # a < b


def test_compare_groups_insufficient_samples():
    result = compare_groups([1.0], [2.0])
    assert "error" in result
    assert "insufficient" in result["error"]
