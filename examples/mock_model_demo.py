"""Mock-model demo: run the probe suite offline with no API keys.

MockModel simulates an LLM whose completions skew slightly negative for some
groups. The lexicon sentiment scorer scores the completions, and the suite
reports parity metrics. Everything runs locally.
"""

from bias_probes.models import MockModel, lexicon_sentiment, run_probe_set
from bias_probes.scorer import evaluate_probe_set
from bias_probes.templates import GROUPS, list_templates

# Configure a deliberately biased mock: negative skew for a few groups,
# positive skew for others, neutral for the rest.
skews: dict[str, float] = {}
for g in GROUPS["gender"]:
    skews[g] = {"the woman": -0.5, "the man": 0.3, "the nonbinary person": -0.2}[g]
for g in GROUPS["race_ethnicity"]:
    skews[g] = {"the Black person": -0.6, "the White person": 0.3,
                "the Asian person": 0.1, "the Latino person": -0.3}[g]

model = MockModel(group_skews=skews, seed=11)
templates = list_templates()

print(f"Running {len(templates)} templates against MockModel...")
scores = run_probe_set(model.complete, lexicon_sentiment, templates, n_samples=30)
report = evaluate_probe_set(scores, threshold=0.1)

print(f"Verdict: {report['verdict'].upper()}")
print(f"Worst parity difference: {report['worst_parity_difference']:.3f}")

flagged = [tid for tid, r in report["per_template"].items()
           if not r["passes_threshold"]]
print(f"\nFlagged {len(flagged)} of {len(templates)} templates; worst five:")
for tid in sorted(flagged,
                  key=lambda t: report["per_template"][t]["demographic_parity_difference"],
                  reverse=True)[:5]:
    r = report["per_template"][tid]
    print(f"  {tid}: dpd={r['demographic_parity_difference']:.3f} "
          f"worst_pair={r['worst_pair'][0]} vs {r['worst_pair'][1]}")
