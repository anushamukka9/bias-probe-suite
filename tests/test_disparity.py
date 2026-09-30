"""Tests for pairwise disparity scoring."""

import pytest

from bias_probes.scorer import evaluate_probe_set, pairwise_disparities, worst_pair


def test_pairwise_disparities_sorted_desc():
    scores = {"a": [0.9], "b": [0.5], "c": [0.7]}
    pairs = pairwise_disparities(scores)
    gaps = list(pairs.values())
    assert gaps == sorted(gaps, reverse=True)
    assert pairs[("a", "b")] == pytest.approx(0.4)
    assert pairs[("a", "c")] == pytest.approx(0.2)
    assert pairs[("b", "c")] == pytest.approx(0.2)


def test_pairwise_disparities_single_group_empty():
    assert pairwise_disparities({"a": [0.5]}) == {}


def test_worst_pair():
    scores = {"a": [0.9], "b": [0.5], "c": [0.7]}
    (g1, g2), gap = worst_pair(scores)
    assert {g1, g2} == {"a", "b"}
    assert gap == pytest.approx(0.4)


def test_evaluate_includes_pairwise_fields():
    report = evaluate_probe_set({"t1": {"a": [0.9], "b": [0.5]}}, threshold=0.1)
    r = report["per_template"]["t1"]
    assert r["worst_pair"] == ["a", "b"]
    assert r["worst_pair_gap"] == pytest.approx(0.4)
    assert r["pairwise_disparities"] == {"a vs b": pytest.approx(0.4)}
    assert "z_scored_parity" in r
