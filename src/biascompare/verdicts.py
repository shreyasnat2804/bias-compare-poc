"""Plain-English verdict text. See DESIGN.md §6.

Every verdict here follows DESIGN §6's rule: a non-significant result is
always phrased as "insufficient evidence" (or "no detectable difference"),
never as a claim that there is "no bias": with small samples the study is
underpowered, and absence of evidence is not evidence of absence.

Wording follows the run's axis ("gender swaps", "disability swaps") and its
group labels ("versions referencing women than versions referencing men").
Comparison verdicts use the Holm-adjusted p-value when one is present.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from biascompare.axes import BUILTIN_AXES, AxisSpec


def _axis(axis: AxisSpec | None) -> AxisSpec:
    return axis or BUILTIN_AXES["gender"]


def _is_significant(p: float, alpha: float) -> bool:
    return p is not None and not math.isnan(p) and p < alpha


def _ci_excludes_zero(ci: Sequence[float]) -> bool:
    lower, upper = ci
    if lower is None or upper is None or math.isnan(lower) or math.isnan(upper):
        return False
    return lower > 0 or upper < 0


def _comparison_p(diff: dict) -> float:
    return diff.get("p_holm", diff["p"])


def _direction_word(mean_direction: float) -> str:
    if math.isnan(mean_direction) or mean_direction == 0:
        return "about the same for"
    return "higher for" if mean_direction > 0 else "lower for"


def model_verdict(
    p_delta_bonferroni: float,
    mean_direction: float,
    p_direction: float,
    n: int,
    alpha: float,
    axis: AxisSpec | None = None,
) -> str:
    """Per-model verdict (DESIGN §6): a sentence on `p_delta_bonferroni`
    against `alpha`, then a sentence on the sign and significance of
    `mean_direction`."""
    axis = _axis(axis)
    return (
        f"{_delta_sentence(p_delta_bonferroni, n, alpha, axis)} "
        f"{_direction_sentence(mean_direction, p_direction, alpha, axis)}"
    )


def _delta_sentence(p_delta_bonferroni: float, n: int, alpha: float, axis: AxisSpec) -> str:
    if _is_significant(p_delta_bonferroni, alpha):
        return (
            f"{axis.name.capitalize()} swaps move this model's output more than matched wording changes do "
            f"(Bonferroni-corrected p = {p_delta_bonferroni:.4f}, n = {n})."
        )
    return f"Insufficient evidence of {axis.name} bias beyond wording sensitivity (n = {n})."


def _direction_sentence(mean_direction: float, p_direction: float, alpha: float, axis: AxisSpec) -> str:
    word = _direction_word(mean_direction)
    target, other = axis.label(axis.target_group), axis.label(axis.other_group)
    if _is_significant(p_direction, alpha):
        return (
            f"On average, toxicity scores are {word} versions referencing {target} than versions "
            f"referencing {other}, and the difference is statistically significant (p = {p_direction:.4f})."
        )
    return (
        f"On average, toxicity scores are {word} versions referencing {target} than versions "
        f"referencing {other}, but the difference is not statistically significant."
    )


def comparison_verdict(
    reference_label: str,
    model_label: str,
    delta_diff: dict,
    direction_diff: dict,
    alpha: float,
    n: int,
    axis: AxisSpec | None = None,
) -> str:
    """Model-vs-model verdict (DESIGN §6): one sentence each for the
    difference in the effect beyond wording sensitivity and in direction."""
    axis = _axis(axis)
    delta_sentence = _diff_sentence(
        model_label, reference_label, f"{axis.name} effect beyond wording sensitivity", delta_diff, alpha, n
    )
    direction_sentence = _diff_sentence(
        model_label, reference_label, f"direction of the {axis.name} effect", direction_diff, alpha, n
    )
    return f"{delta_sentence} {direction_sentence}"


def _diff_sentence(model_label: str, reference_label: str, metric: str, diff: dict, alpha: float, n: int) -> str:
    p = _comparison_p(diff)
    if _is_significant(p, alpha) and _ci_excludes_zero(diff["ci"]):
        comparative = "more" if diff["mean"] > 0 else "less"
        return f"{model_label} shows {comparative} {metric} than {reference_label} (p = {p:.4f}, n = {n})."
    return (
        f"There is no detectable difference in {metric} between {model_label} and "
        f"{reference_label} (insufficient evidence at n = {n})."
    )


def build_summary_verdict(
    comparisons: Sequence[dict], n: int, alpha: float, axis: AxisSpec | None = None
) -> str:
    """Top-level `summary_verdict`: plain-English sentences covering every
    comparison, always naming `n`, and only ever describing a
    non-significant result as "insufficient evidence"."""
    axis = _axis(axis)
    if not comparisons:
        return (
            "Fewer than two models were successfully scored, so no model-vs-model "
            f"comparison could be made (n = {n})."
        )

    sentences = [_summary_sentence(comparison, alpha, n, axis) for comparison in comparisons]
    if not _any_significant(comparisons, alpha):
        sentences.append(
            f"With only n = {n} posts, statistical power is low, so this result is inconclusive "
            "rather than a confirmation that the models behave identically."
        )
    return " ".join(sentences)


def _any_significant(comparisons: Sequence[dict], alpha: float) -> bool:
    return any(
        _is_significant(_comparison_p(comparison["delta_diff"]), alpha)
        and _ci_excludes_zero(comparison["delta_diff"]["ci"])
        for comparison in comparisons
    )


def _summary_sentence(comparison: dict, alpha: float, n: int, axis: AxisSpec) -> str:
    delta_diff = comparison["delta_diff"]
    p = _comparison_p(delta_diff)
    if _is_significant(p, alpha) and _ci_excludes_zero(delta_diff["ci"]):
        comparative = "more" if delta_diff["mean"] > 0 else "less"
        return (
            f"{comparison['model']} shows {comparative} {axis.name} bias beyond wording sensitivity than "
            f"{comparison['reference']} (p = {p:.4f}, n = {n})."
        )
    return (
        f"There is insufficient evidence that {comparison['model']} differs from "
        f"{comparison['reference']} in {axis.name} bias beyond wording sensitivity (n = {n})."
    )
