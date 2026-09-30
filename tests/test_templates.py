"""Tests for the probe template catalog."""

from bias_probes.templates import (
    GROUPS,
    PROBE_TEMPLATES,
    categories,
    dimensions,
    instantiate,
    list_templates,
)


def test_template_ids_unique():
    ids = [t.id for t in PROBE_TEMPLATES]
    assert len(ids) == len(set(ids))


def test_every_template_instantiates_all_groups():
    for t in PROBE_TEMPLATES:
        prompts = instantiate(t)
        assert set(prompts) == set(t.groups), t.id
        for group, prompt in prompts.items():
            assert "{group}" not in prompt
            assert group in prompt


def test_expected_categories_present():
    cats = categories()
    for expected in ("profession", "adjective", "competence", "leadership",
                     "safety", "healthcare", "finance", "education"):
        assert expected in cats, expected


def test_dimensions_cover_group_definitions():
    assert set(dimensions()) == set(GROUPS)


def test_template_count():
    assert len(PROBE_TEMPLATES) == 34


def test_no_em_dashes_in_template_text():
    for t in PROBE_TEMPLATES:
        assert "\u2014" not in t.template, t.id
        assert "\u2014" not in t.notes, t.id


def test_list_templates_filters():
    assert all(t.category == "healthcare" for t in list_templates(category="healthcare"))
    assert len(list_templates(category="healthcare")) == 4
    assert all(t.dimension == "gender" for t in list_templates(dimension="gender"))
