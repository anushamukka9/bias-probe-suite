"""bias-probe-suite: template-based bias probing for LLMs."""

from .scorer import (
    bootstrap_ci,
    demographic_parity_difference,
    evaluate_probe_set,
    worst_group_gap,
)
from .templates import PROBE_TEMPLATES, ProbeTemplate, list_templates

__all__ = [
    "PROBE_TEMPLATES",
    "ProbeTemplate",
    "list_templates",
    "demographic_parity_difference",
    "worst_group_gap",
    "bootstrap_ci",
    "evaluate_probe_set",
]
__version__ = "0.1.0"
