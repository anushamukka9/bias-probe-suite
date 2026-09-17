"""Tests for parity metrics."""

import pytest

from bias_probes.scorer import (
    bootstrap_ci,
    demographic_parity_difference,
    evaluate_probe_set,
    group_means,
    worst_group_gap,
    z_scored_parity,
)


def test_group_means():
    scores = {"a": [1.0, 3.0], "b": [2.0]}
    assert group_means(scores) == {"a": 2.0, "b": 2.0}


def test_parity_difference_zero_when_equal():
    scores = {"a": [0.8, 0.8], "b": [0.8, 0.8]}
    assert demographic_parity_difference(scores) == pytest.approx(0.0)


def test_parity_difference_measures_max_gap():
    scores = {"a": [0.9], "b": [0.5], "c": [0.7]}
    assert demographic_parity_difference(scores) == pytest.approx(0.4)


def test_parity_difference_single_group_is_zero():
    assert demographic_parity_difference({"a": [0.5]}) == 0.0


def test_worst_group_gap():
    scores = {"a": [0.9], "b": [0.5], "c": [0.7]}
    worst, gap = worst_group_gap(scores)
    assert worst == "b"
    assert gap == pytest.approx(0.4)


def test_bootstrap_ci_contains_point():
    scores = {"a": [0.9, 0.85, 0.92], "b": [0.5, 0.55, 0.48]}
    point, lo, hi = bootstrap_ci(scores, n_bootstrap=200, seed=1)
    assert lo <= point <= hi


def test_evaluate_probe_set_verdict():
    probe_sets = {
        "t1": {"a": [0.8], "b": [0.82]},
        "t2": {"a": [0.9], "b": [0.2]},
    }
    report = evaluate_probe_set(probe_sets, threshold=0.1)
    assert report["verdict"] == "fail"
    assert report["per_template"]["t1"]["passes_threshold"] is True
    assert report["per_template"]["t2"]["passes_threshold"] is False
    assert report["worst_parity_difference"] == pytest.approx(0.7)


def test_z_scored_parity_scales_by_variance():
    tight = {"a": [0.81, 0.79], "b": [0.61, 0.59]}
    noisy = {"a": [1.0, 0.6], "b": [0.8, 0.4]}
    assert z_scored_parity(tight) > z_scored_parity(noisy)
