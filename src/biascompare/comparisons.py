"""Model-vs-model comparisons. See DESIGN.md §6 and docs/PLAN.md §6.

Two modes:

- `all-pairs` (default): every pair of successfully scored models, model j
  against model i for i < j, in the order the models were given. This is
  what the M0 to M3 design needs (prompting alone, fine-tuning alone, both).
- `reference`: every model against the first successfully scored model.

Comparisons are paired per row over the same eligible-row arrays
`model_metrics.compute_model_summary` already produced, so no metric is
recomputed here. p-values are Holm-adjusted across the comparisons of a run
(separately for the Δ and the direction differences), and verdicts use the
adjusted value. With one comparison the adjustment changes nothing.
"""

from __future__ import annotations

from biascompare.axes import AxisSpec
from biascompare.rng import DELTA_DIFF_CI, DIRECTION_DIFF_CI, derive_rng
from biascompare.stats import bootstrap_ci, holm, mean_or_nan, wilcoxon_two_sided
from biascompare.verdicts import comparison_verdict

COMPARISON_MODES = ("all-pairs", "reference")


def build_comparison(
    reference_label: str,
    reference_deltas: list[float],
    reference_directions: list[float],
    model_label: str,
    model_deltas: list[float],
    model_directions: list[float],
    seed: int,
    comparison_index: int,
    alpha: float,
    n: int,
    axis: AxisSpec | None = None,
) -> dict:
    """Build one `comparisons[]` entry for `model_label` against
    `reference_label`. The two delta/direction arrays must already be paired
    by row (same eligible rows, same order), which `compare_models`
    guarantees since every model is scored over the identical, shared plan."""
    delta_diff = [k - r for k, r in zip(model_deltas, reference_deltas)]
    direction_diff = [k - r for k, r in zip(model_directions, reference_directions)]

    delta_summary = _diff_summary(delta_diff, derive_rng(seed, DELTA_DIFF_CI, comparison_index))
    direction_summary = _diff_summary(direction_diff, derive_rng(seed, DIRECTION_DIFF_CI, comparison_index))

    return {
        "reference": reference_label,
        "model": model_label,
        "delta_diff": delta_summary,
        "direction_diff": direction_summary,
        "verdict": comparison_verdict(reference_label, model_label, delta_summary, direction_summary, alpha, n, axis),
    }


def build_comparisons(
    successful: list[dict], seed: int, alpha: float, n: int, mode: str = "all-pairs", axis: AxisSpec | None = None
) -> list[dict]:
    """All comparisons for one axis, Holm-adjusted. `successful` holds one
    `{"label", "deltas", "directions"}` per successfully scored model, in
    model order."""
    if mode not in COMPARISON_MODES:
        raise ValueError(f"comparisons must be one of {list(COMPARISON_MODES)}, got {mode!r}")
    if len(successful) < 2:
        return []
    if mode == "reference":
        pairs = [(0, j) for j in range(1, len(successful))]
    else:
        pairs = [(i, j) for i in range(len(successful)) for j in range(i + 1, len(successful))]

    comparisons = []
    for index, (i, j) in enumerate(pairs):
        reference, model = successful[i], successful[j]
        comparisons.append(
            build_comparison(
                reference["label"], reference["deltas"], reference["directions"],
                model["label"], model["deltas"], model["directions"],
                seed, index, alpha, n, axis,
            )
        )
    return _holm_adjust(comparisons, alpha, n, axis)


def _holm_adjust(comparisons: list[dict], alpha: float, n: int, axis: AxisSpec | None) -> list[dict]:
    for key in ("delta_diff", "direction_diff"):
        for comparison, p in zip(comparisons, holm([c[key]["p"] for c in comparisons])):
            comparison[key]["p_holm"] = p
    for comparison in comparisons:
        comparison["verdict"] = comparison_verdict(
            comparison["reference"], comparison["model"], comparison["delta_diff"],
            comparison["direction_diff"], alpha, n, axis,
        )
    return comparisons


def _diff_summary(diff: list[float], rng) -> dict:
    return {
        "mean": mean_or_nan(diff),
        "ci": list(bootstrap_ci(diff, rng)),
        "p": wilcoxon_two_sided(diff),
    }
