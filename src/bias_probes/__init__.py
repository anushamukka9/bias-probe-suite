"""bias-probe-suite: template-based bias probing for LLMs."""

from .models import MockModel, lexicon_sentiment, run_probe_set
from .report import render_findings_markdown
from .scorer import (
    bootstrap_ci,
    demographic_parity_difference,
    evaluate_probe_set,
    pairwise_disparities,
    worst_group_gap,
    worst_pair,
    z_scored_parity,
)
from .templates import (
    GROUPS,
    PROBE_TEMPLATES,
    ProbeTemplate,
    categories,
    dimensions,
    instantiate,
    list_templates,
)

__all__ = [
    "GROUPS",
    "PROBE_TEMPLATES",
    "MockModel",
    "ProbeTemplate",
    "bootstrap_ci",
    "categories",
    "demographic_parity_difference",
    "dimensions",
    "evaluate_probe_set",
    "instantiate",
    "lexicon_sentiment",
    "list_templates",
    "pairwise_disparities",
    "render_findings_markdown",
    "run_probe_set",
    "worst_group_gap",
    "worst_pair",
    "z_scored_parity",
]
__version__ = "0.1.0"
