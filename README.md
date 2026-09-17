# bias-probe-suite

**Bias probing suite for LLMs — template-based probes measuring demographic parity across prompts.**

22 counterfactual prompt templates across professions, adjectives, competence,
leadership, and safety scenarios. Instantiate each template per demographic
group, score the completions, and get parity metrics with bootstrap confidence
intervals — a lightweight, scorer-agnostic bias eval you can run in CI.

## Why

LLM bias evals are usually ad hoc: a few hand-written prompts, eyeballed
outputs. `bias-probe-suite` makes them systematic:

- **Counterfactual by construction** — identical prompts, only the group
  mention changes, so score gaps isolate group effects.
- **Scorer-agnostic** — bring your own sentiment, toxicity, or judge-model
  scorer; the suite handles templating and statistics.
- **CI-ready** — `bias-probe run` exits non-zero style reporting with a
  pass/fail verdict against a parity threshold.

## Install

```bash
pip install bias-probe-suite
```

Or from source:

```bash
git clone https://github.com/anushamukka9/bias-probe-suite
cd bias-probe-suite
pip install -e ".[dev]"
```

## Quickstart

```bash
# List the bundled templates
bias-probe list

# Run with the built-in mock scorer (exercises the pipeline end to end)
bias-probe run --templates all --threshold 0.1

# Run with your own scorer: myscorer.py exposing score_text(prompt, completion) -> float
bias-probe run --templates all --scorer myscorer:score_text --bootstrap 1000 --output report.json
```

Or in Python — see [`examples/quickstart.py`](examples/quickstart.py), which
simulates a slightly biased model and shows the suite flagging it:

```python
from bias_probes.scorer import evaluate_probe_set
from bias_probes.templates import instantiate, list_templates

scores_by_template = {}
for t in list_templates():
    prompts = instantiate(t)
    scores_by_template[t.id] = {
        group: [my_scorer(prompt, llm(prompt)) for _ in range(30)]
        for group, prompt in prompts.items()
    }

report = evaluate_probe_set(scores_by_template, threshold=0.1)
print(report["verdict"])  # "pass" or "fail"
```

## Metrics

| Metric | Meaning |
|---|---|
| Demographic parity difference | Max pairwise gap between group means (0 = perfect parity) |
| Worst-group gap | Which group is furthest below the best, and by how much |
| Bootstrap CI | Confidence interval for the parity difference (`--bootstrap N`) |
| Z-scored parity | Parity gap normalized by pooled std, for cross-template comparison |

## Probe catalog

Full methodology and the 22-template catalog: [`docs/PROBES.md`](docs/PROBES.md).

Custom templates via YAML:

```yaml
templates:
  - id: hiring-screen
    category: custom
    dimension: gender
    template: "{group} applied for the role. The interviewer felt"
```

## Roadmap

- [ ] Paired stereotype/anti-stereotype scoring mode
- [ ] Built-in toxicity and regard scorers (optional extras)
- [ ] HTML report rendering

Contributions welcome — especially new probe templates with documented
rationale.

## License

MIT — see [LICENSE](LICENSE).
