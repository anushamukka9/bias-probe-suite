"""Tests for the mock model interface."""

from bias_probes.models import MockModel, lexicon_sentiment, run_probe_set
from bias_probes.templates import list_templates


def test_mock_model_is_deterministic():
    m1, m2 = MockModel(seed=3), MockModel(seed=3)
    prompts = ["the woman is here", "the man is here", "the woman is here"]
    assert [m1.complete(p) for p in prompts] == [m2.complete(p) for p in prompts]
    # Successive samples of one prompt still vary.
    assert len({MockModel(seed=3).complete("the woman is here") for _ in range(1)}
               | {m1.complete("the woman is here") for _ in range(5)}) > 1


def test_mock_model_respects_group_skew():
    biased = MockModel(group_skews={"the woman": -0.9, "the man": 0.9}, seed=1)
    plain = MockModel(seed=1)
    neg = [lexicon_sentiment("", biased.complete("the woman walks")) for _ in range(40)]
    pos = [lexicon_sentiment("", biased.complete("the man walks")) for _ in range(40)]
    neu = [lexicon_sentiment("", plain.complete("the woman walks")) for _ in range(40)]
    assert sum(neg) / len(neg) < sum(neu) / len(neu) < sum(pos) / len(pos)


def test_mock_model_unknown_group_is_neutral():
    m = MockModel(group_skews={"the woman": -0.9}, seed=2)
    scores = [lexicon_sentiment("", m.complete("the moon is bright")) for _ in range(60)]
    assert 0.4 < sum(scores) / len(scores) < 0.6


def test_lexicon_sentiment_bounds():
    assert lexicon_sentiment("", "excellent capable reliable") == 1.0
    assert lexicon_sentiment("", "poor unreliable weak") == 0.0
    assert lexicon_sentiment("", "no sentiment words here at all") == 0.5


def test_run_probe_set_shape():
    templates = list_templates()[:2]
    scores = run_probe_set(MockModel(seed=0).complete, lexicon_sentiment, templates, n_samples=5)
    assert set(scores) == {t.id for t in templates}
    for tid, per_group in scores.items():
        template = next(t for t in templates if t.id == tid)
        assert set(per_group) == set(template.groups)
        for vals in per_group.values():
            assert len(vals) == 5
            assert all(0.0 <= v <= 1.0 for v in vals)
