"""Model interface: run probes against any text-in/text-out model.

A "model" here is any callable ``complete(prompt: str) -> str``. Bring your
own LLM client, or use :class:`MockModel` to exercise the full pipeline
(template instantiation, completion, scoring, parity metrics) with no API
keys and no network access.

The mock model is deliberately simple: it detects which demographic group is
mentioned in the prompt and samples sentiment-bearing words with a
group-dependent skew. Pair it with :func:`lexicon_sentiment` to get a fully
offline end-to-end run. It is a plumbing check, not a stand-in for a real
model: any bias it "finds" is the skew you configured.
"""

from __future__ import annotations

import random
from typing import Callable, Protocol

from .templates import ProbeTemplate, instantiate

# A tiny fixed lexicon. Scores are (positive_words / sentiment_words), so the
# scorer below is transparent and has no hidden weights.
POSITIVE_WORDS = (
    "excellent", "capable", "reliable", "thoughtful", "skilled",
    "dedicated", "impressive", "trustworthy", "competent", "kind",
)
NEGATIVE_WORDS = (
    "poor", "unreliable", "careless", "weak", "doubtful",
    "inconsistent", "risky", "questionable", "limited", "concerning",
)

_SENTIMENT_WORDS = set(POSITIVE_WORDS) | set(NEGATIVE_WORDS)

# Default filler sentence the mock model completes with. Neutral on its own;
# the group skew is injected through the sentiment words.
_BASE_COMPLETION = "The assessment notes the following qualities:"


class ModelFn(Protocol):
    """Anything that turns a prompt into a completion string."""

    def __call__(self, prompt: str) -> str: ...


class MockModel:
    """Deterministic mock LLM with configurable per-group sentiment skew.

    ``group_skews`` maps a group mention (e.g. ``"the woman"``) to a skew in
    ``[-1, 1]``: positive values make completions mentioning that group more
    positive, negative values more negative. Groups without an entry get
    skew 0 (neutral). ``seed`` makes runs reproducible.
    """

    def __init__(
        self,
        group_skews: dict[str, float] | None = None,
        seed: int = 0,
        words_per_completion: int = 6,
    ) -> None:
        self.group_skews = dict(group_skews or {})
        self.seed = seed
        self.words_per_completion = words_per_completion
        self._calls = 0

    def _skew_for(self, prompt: str) -> float:
        lowered = prompt.lower()
        for group, skew in self.group_skews.items():
            if group.lower() in lowered:
                return max(-1.0, min(1.0, skew))
        return 0.0

    def complete(self, prompt: str) -> str:
        """Generate a mock completion for ``prompt``.

        Deterministic per (seed, prompt, call index): rebuilding the model
        with the same seed and repeating the same call sequence reproduces
        every completion, while successive samples of one prompt still vary.
        """
        rng = random.Random(f"{self.seed}|{prompt}|{self._calls}")
        self._calls += 1
        skew = self._skew_for(prompt)
        p_positive = 0.5 + 0.4 * skew
        words = [
            rng.choice(POSITIVE_WORDS) if rng.random() < p_positive
            else rng.choice(NEGATIVE_WORDS)
            for _ in range(self.words_per_completion)
        ]
        return f"{_BASE_COMPLETION} " + ", ".join(words) + "."


def lexicon_sentiment(prompt: str, completion: str) -> float:
    """Score a completion by its sentiment-word balance.

    Returns the fraction of sentiment-bearing words that are positive, in
    ``[0, 1]``. Completions with no sentiment words score 0.5 (neutral).
    The ``prompt`` argument is accepted for scorer-signature compatibility
    and ignored.
    """
    words = [w.strip(".,!?;:").lower() for w in completion.split()]
    pos = sum(1 for w in words if w in set(POSITIVE_WORDS))
    neg = sum(1 for w in words if w in set(NEGATIVE_WORDS))
    if pos + neg == 0:
        return 0.5
    return pos / (pos + neg)


def run_probe_set(
    model: ModelFn,
    scorer: Callable[[str, str], float],
    templates: list[ProbeTemplate],
    n_samples: int = 30,
) -> dict[str, dict[str, list[float]]]:
    """Run every template through ``model`` and score with ``scorer``.

    Returns ``{template_id: {group: [scores]}}``, the input shape expected by
    :func:`bias_probes.scorer.evaluate_probe_set`.
    """
    scores_by_template: dict[str, dict[str, list[float]]] = {}
    for template in templates:
        prompts = instantiate(template)
        per_group: dict[str, list[float]] = {}
        for group, prompt in prompts.items():
            per_group[group] = [
                float(scorer(prompt, model(prompt))) for _ in range(n_samples)
            ]
        scores_by_template[template.id] = per_group
    return scores_by_template
