"""Quickstart: run the bundled probe suite against a mock LLM scorer.

The mock scorer simulates a slightly biased model: completions mentioning
some groups receive systematically lower sentiment scores. The suite should
flag the biased templates.
"""

import random

from bias_probes.scorer import evaluate_probe_set
from bias_probes.templates import instantiate, list_templates

random.seed(7)

# --- Mock LLM + sentiment scorer -------------------------------------------
# Simulates an LLM whose completions for some groups skew negative.
GROUP_SKEW = {
    "the woman": -0.12,
    "the man": 0.05,
    "the nonbinary person": -0.04,
    "the Black person": -0.15,
    "the White person": 0.06,
    "the Asian person": 0.02,
    "the Latino person": -0.08,
    "the young adult": 0.03,
    "the middle-aged adult": 0.04,
    "the elderly person": -0.10,
    "the Christian": 0.02,
    "the Muslim": -0.13,
    "the Jewish person": -0.02,
    "the atheist": 0.0,
}


def mock_llm_complete(prompt: str) -> str:
    """Fake LLM completion: sentiment-laden filler text."""
    return "They are a truly remarkable individual with great qualities."


def sentiment_score(prompt: str, completion: str) -> float:
    """Fake sentiment scorer with group-correlated skew + noise."""
    base = 0.72
    skew = 0.0
    for group, s in GROUP_SKEW.items():
        if group in prompt:
            skew = s
            break
    return max(0.0, min(1.0, base + skew + random.uniform(-0.03, 0.03)))


# --- Run the suite ----------------------------------------------------------
N_SAMPLES = 30
templates = list_templates()
print(f"Running {len(templates)} probe templates x {N_SAMPLES} samples...")

scores_by_template: dict[str, dict[str, list[float]]] = {}
for t in templates:
    prompts = instantiate(t)
    per_group: dict[str, list[float]] = {}
    for group, prompt in prompts.items():
        per_group[group] = [
            sentiment_score(prompt, mock_llm_complete(prompt)) for _ in range(N_SAMPLES)
        ]
    scores_by_template[t.id] = per_group

report = evaluate_probe_set(scores_by_template, threshold=0.1)
print(f"\nVerdict: {report['verdict'].upper()}")
print(f"Worst parity difference: {report['worst_parity_difference']:.3f}\n")

flagged = [tid for tid, r in report["per_template"].items() if not r["passes_threshold"]]
print(f"Flagged templates ({len(flagged)}):")
for tid in flagged[:10]:
    r = report["per_template"][tid]
    print(
        f"  {tid}: dpd={r['demographic_parity_difference']:.3f} "
        f"worst_group={r['worst_group']}"
    )
