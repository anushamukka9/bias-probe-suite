"""Findings reports: turn an evaluation dict into a readable Markdown report.

:func:`render_findings_markdown` takes the output of
:func:`bias_probes.scorer.evaluate_probe_set` and produces a Markdown
document with a verdict summary, a per-template table, and a detail section
for every template that failed the threshold. The report states its own
limits plainly: which scorer produced the numbers, how many samples backed
each mean, and what the threshold means.
"""

from __future__ import annotations


def render_findings_markdown(
    report: dict,
    *,
    title: str = "Bias Probe Findings",
    scorer_description: str = "user-supplied scorer",
    samples_per_group: int | None = None,
) -> str:
    """Render ``report`` (from ``evaluate_probe_set``) as Markdown."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"**Verdict:** {report['verdict'].upper()}")
    lines.append(f"**Parity threshold:** {report['threshold']}")
    lines.append(f"**Templates evaluated:** {report['templates_evaluated']}")
    lines.append(
        f"**Worst parity difference:** {report['worst_parity_difference']:.4f}"
    )
    lines.append(f"**Scorer:** {scorer_description}")
    if samples_per_group is not None:
        lines.append(f"**Samples per group:** {samples_per_group}")
    lines.append("")

    per_template = report["per_template"]
    failed = [tid for tid, r in per_template.items() if not r["passes_threshold"]]
    lines.append(f"**Templates failing threshold:** {len(failed)} of {len(per_template)}")
    lines.append("")

    lines.append("## Summary by template")
    lines.append("")
    lines.append("| Template | Parity diff | Worst group | Worst pair | Verdict |")
    lines.append("|---|---|---|---|---|")
    for tid, r in per_template.items():
        mark = "PASS" if r["passes_threshold"] else "FAIL"
        pair = f"{r['worst_pair'][0]} vs {r['worst_pair'][1]}" if r["worst_pair"][0] else "-"
        lines.append(
            f"| `{tid}` | {r['demographic_parity_difference']:.4f} "
            f"| {r['worst_group']} ({r['worst_group_gap']:.4f}) "
            f"| {pair} ({r['worst_pair_gap']:.4f}) | {mark} |"
        )
    lines.append("")

    if failed:
        lines.append("## Failing templates (detail)")
        lines.append("")
        for tid in failed:
            r = per_template[tid]
            lines.append(f"### `{tid}`")
            lines.append("")
            lines.append(
                f"Parity difference {r['demographic_parity_difference']:.4f} "
                f"exceeds the {report['threshold']} threshold."
            )
            lines.append("")
            lines.append("| Group | Mean score |")
            lines.append("|---|---|")
            for group, mean in sorted(
                r["group_means"].items(), key=lambda kv: kv[1], reverse=True
            ):
                lines.append(f"| {group} | {mean:.4f} |")
            lines.append("")
            lines.append("Largest pairwise disparities:")
            lines.append("")
            for pair, gap in list(r["pairwise_disparities"].items())[:3]:
                lines.append(f"- {pair}: {gap:.4f}")
            lines.append("")

    lines.append("## Reading this report")
    lines.append("")
    lines.append(
        "- Parity difference is the max pairwise gap between group mean scores "
        "(0 means perfect parity). It measures the *prompted* behavior under "
        "this scorer, not bias in any absolute sense."
    )
    lines.append(
        "- A failing template is a signal to investigate the prompts, the "
        "scorer, and the model, not a verdict on the model alone."
    )
    lines.append(
        "- Threshold choice is a policy decision. The default 0.1 is a "
        "starting point; calibrate it against a known-good baseline before "
        "gating releases on it."
    )
    lines.append("")
    return "\n".join(lines)
