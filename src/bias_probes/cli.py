"""CLI: bias-probe run --templates <dir|all> --scorer <module:func>"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import click
import yaml

from .scorer import bootstrap_ci, evaluate_probe_set
from .templates import GROUPS, ProbeTemplate, instantiate, list_templates


def _load_custom_templates(path: Path) -> list[ProbeTemplate]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    templates: list[ProbeTemplate] = []
    for item in data.get("templates", []):
        dimension = item.get("dimension", "gender")
        templates.append(
            ProbeTemplate(
                id=item["id"],
                category=item.get("category", "custom"),
                template=item["template"],
                dimension=dimension,
                groups=tuple(item.get("groups", GROUPS.get(dimension, ()))),
                notes=item.get("notes", ""),
            )
        )
    return templates


def _load_scorer(spec: str):
    module_name, _, func_name = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, func_name)


@click.group()
def main() -> None:
    """Bias probing suite for LLMs — measure demographic parity across prompts."""


@main.command("list")
@click.option("--category", default=None, help="Filter by category.")
@click.option("--dimension", default=None, help="Filter by dimension.")
def list_cmd(category: str | None, dimension: str | None) -> None:
    """List bundled probe templates."""
    for t in list_templates(category=category, dimension=dimension):
        click.echo(f"{t.id}  [{t.category}/{t.dimension}]  {t.template[:70]}...")


@main.command("run")
@click.option(
    "--templates",
    "templates_opt",
    default="all",
    help="Template YAML file, or 'all' for the bundled suite.",
)
@click.option(
    "--scorer",
    "scorer_spec",
    default=None,
    help="Scoring function as module:function, e.g. myscorer:score_text. "
    "Called as score_text(prompt, completion) -> float.",
)
@click.option("--threshold", default=0.1, type=float, help="Parity-difference pass threshold.")
@click.option("--bootstrap", default=0, type=int, help="Bootstrap iterations for CIs (0 to skip).")
@click.option("--output", "output_path", default=None, type=click.Path(), help="Write JSON report here.")
def run(
    templates_opt: str,
    scorer_spec: str | None,
    threshold: float,
    bootstrap: int,
    output_path: str | None,
) -> None:
    """Run bias probes and report demographic-parity metrics.

    Without --scorer, completions are simulated with a mock scorer so the
    pipeline can be exercised end to end (see examples/quickstart.py for a
    worked example with a real scoring function).
    """
    if templates_opt == "all":
        templates = list_templates()
    else:
        templates = _load_custom_templates(Path(templates_opt))

    scorer = _load_scorer(scorer_spec) if scorer_spec else None

    scores_by_template: dict[str, dict[str, list[float]]] = {}
    for t in templates:
        prompts = instantiate(t)
        per_group: dict[str, list[float]] = {}
        for group, prompt in prompts.items():
            if scorer is not None:
                completion = f"[completion for: {prompt}]"  # placeholder hook
                score = float(scorer(prompt, completion))
            else:
                # Deterministic mock: slight group-correlated skew so the
                # metrics exercise the non-trivial path.
                skew = (hash(group) % 100) / 1000.0
                score = 0.5 + skew
            per_group[group] = [score]
        scores_by_template[t.id] = per_group

    report = evaluate_probe_set(scores_by_template, threshold=threshold)

    if bootstrap > 0:
        for template_id, scores in scores_by_template.items():
            point, lo, hi = bootstrap_ci(
                scores, n_bootstrap=bootstrap, seed=42
            )
            report["per_template"][template_id]["bootstrap_ci"] = {
                "point": point,
                "lower": lo,
                "upper": hi,
                "n_bootstrap": bootstrap,
            }

    click.echo(f"Templates: {report['templates_evaluated']}")
    click.echo(f"Worst parity difference: {report['worst_parity_difference']:.4f}")
    click.echo(f"Threshold: {threshold}  ->  {report['verdict'].upper()}")
    for tid, r in report["per_template"].items():
        mark = "PASS" if r["passes_threshold"] else "FAIL"
        click.echo(
            f"  [{mark}] {tid}: dpd={r['demographic_parity_difference']:.4f} "
            f"worst_group={r['worst_group']}"
        )

    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        click.echo(f"Report written to {output_path}")


if __name__ == "__main__":
    main()
