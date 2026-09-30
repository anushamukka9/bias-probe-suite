"""Tests for the findings report renderer."""

from bias_probes.report import render_findings_markdown
from bias_probes.scorer import evaluate_probe_set


def _report():
    return evaluate_probe_set(
        {
            "t1": {"a": [0.9, 0.88], "b": [0.5, 0.52]},
            "t2": {"a": [0.8], "b": [0.82]},
        },
        threshold=0.1,
    )


def test_findings_markdown_structure():
    md = render_findings_markdown(_report(), scorer_description="test scorer",
                                  samples_per_group=30)
    assert "# Bias Probe Findings" in md
    assert "**Verdict:** FAIL" in md
    assert "`t1`" in md and "`t2`" in md
    assert "## Failing templates (detail)" in md
    assert "the worst_group" not in md  # no template internals leak
    assert "test scorer" in md
    assert "Samples per group" in md
    assert "## Reading this report" in md


def test_findings_markdown_pass_has_no_detail_section():
    report = evaluate_probe_set({"t2": {"a": [0.8], "b": [0.82]}}, threshold=0.1)
    md = render_findings_markdown(report)
    assert "**Verdict:** PASS" in md
    assert "## Failing templates (detail)" not in md


def test_findings_markdown_custom_title():
    md = render_findings_markdown(_report(), title="Weekly Regression")
    assert md.startswith("# Weekly Regression")
