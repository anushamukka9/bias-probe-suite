"""Findings report demo: evaluate probes, then render a Markdown report.

Writes findings.md next to this script. Open it to see the verdict summary,
the per-template table, and detail sections for failing templates.
"""

from pathlib import Path

from bias_probes.models import MockModel, lexicon_sentiment, run_probe_set
from bias_probes.report import render_findings_markdown
from bias_probes.scorer import evaluate_probe_set
from bias_probes.templates import list_templates

model = MockModel(
    group_skews={"the woman": -0.5, "the Black person": -0.6, "the Muslim": -0.5},
    seed=11,
)
templates = list_templates(category="healthcare") + list_templates(category="finance")

scores = run_probe_set(model.complete, lexicon_sentiment, templates, n_samples=30)
report = evaluate_probe_set(scores, threshold=0.1)

markdown = render_findings_markdown(
    report,
    title="Bias Probe Findings: healthcare + finance",
    scorer_description="lexicon_sentiment over MockModel completions",
    samples_per_group=30,
)

out = Path(__file__).resolve().parent / "findings.md"
out.write_text(markdown, encoding="utf-8")
print(f"Wrote {out} ({len(markdown)} chars)")
print(f"Verdict: {report['verdict']}")
