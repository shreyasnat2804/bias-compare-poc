"""Per-post metrics and per-model summary statistics. See DESIGN.md §6.

`compute_per_post_metrics` turns one post's scored probability pairs into
the `per_post[]` entry from DESIGN §7. `compute_model_summary` aggregates
those entries (restricted to posts eligible for statistics, DESIGN §5) into
the `summary` object, and also returns the raw per-post delta/direction
arrays so `compare.py` can pair them up for the model-vs-reference
comparisons without recomputing anything.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

from biascompare.metrics import Distribution, js_divergence
from biascompare.perturbation_plan import PostPlan, TextLayout
from biascompare.rng import DIRECTION_CI, derive_rng
from biascompare.stats import bonferroni, bootstrap_ci, mean_or_nan, paired_ttest_greater, ttest_two_sided
from biascompare.tasks import DEFAULT_TASK, TaskSpec
from biascompare.verdicts import model_verdict

_NAN = float("nan")


def compute_per_post_metrics(
    plan: PostPlan,
    layout: TextLayout,
    post_index: int,
    scores: Sequence[Distribution],
    task: TaskSpec = DEFAULT_TASK,
    groups: tuple[str, str] = ("male", "female"),
    axis: str = "gender",
) -> dict:
    """Build one `per_post[]` entry (DESIGN §7) for `plan` from `scores`,
    the full flattened score list for this model (see `TextLayout`).

    JSD uses the full distribution over the task's levels; flips compare
    the most probable level; `direction` is measured on severity (equal to
    P(toxic) for a 2-level task, see `biascompare.tasks`).
    """
    p_original = scores[layout.original_index[post_index]]
    p_gender = scores[layout.gender_index[post_index]]
    p_controls = [scores[i] for i in layout.control_indices[post_index]]

    jsd_gender = js_divergence(p_original, p_gender)
    jsd_controls = [js_divergence(p_original, p_control) for p_control in p_controls]
    mean_jsd_controls = mean_or_nan(jsd_controls)
    delta = jsd_gender - mean_jsd_controls if jsd_controls else _NAN

    level_original = task.predicted_level(p_original)
    flip_gender = task.predicted_level(p_gender) != level_original
    control_flips = [task.predicted_level(p_control) != level_original for p_control in p_controls]
    flip_controls_rate = mean_or_nan([float(f) for f in control_flips])

    direction = _direction(
        plan.post.group(axis), groups, task.severity(p_original), task.severity(p_gender)
    )
    return {
        "post_id": plan.post.id,
        "p_toxic_original": task.p_toxic(p_original),
        "p_toxic_gender_swap": task.p_toxic(p_gender),
        "p_toxic_controls": [task.p_toxic(p) for p in p_controls],
        "p_original": list(p_original),
        "p_gender_swap": list(p_gender),
        "p_controls": [list(p) for p in p_controls],
        "predicted_original": task.levels[level_original],
        "jsd_gender": jsd_gender,
        "jsd_controls": jsd_controls,
        "delta": delta,
        "direction": direction,
        "flip_gender": bool(flip_gender),
        "flip_controls_rate": flip_controls_rate,
    }


def _direction(
    group: str, groups: tuple[str, str], severity_original: float, severity_swap: float
) -> float:
    """Severity of the second group's version minus the first group's
    (for gender: female-referencing minus male-referencing, DESIGN §6)."""
    if group == groups[1]:
        return severity_original - severity_swap
    return severity_swap - severity_original


def compute_model_summary(
    per_post: Sequence[dict],
    eligible_post_ids: set[str],
    n_success_models: int,
    seed: int,
    model_index: int,
    alpha: float,
) -> tuple[dict, list[float], list[float]]:
    """Aggregate `per_post` (restricted to eligible posts) into the
    `summary` object from DESIGN §7. Returns `(summary, deltas, directions)`
    — the eligible per-post arrays, *not* further filtered — so
    `compare.py` can pair them up by position for the model-vs-reference
    comparisons; every model shares the same `eligible_post_ids` (DESIGN
    §5), so the two arrays line up across models without recomputing
    anything.

    The summary's own hypothesis tests, however, are computed only over the
    finite subset of those arrays: `eligible_post_ids` already excludes
    posts with zero usable controls (the `no_controls` flag), but this is a
    defensive second layer so one non-finite value (e.g. from a scorer
    emitting a non-finite probability) can never NaN-poison every other
    post's contribution to `p_delta`/`p_direction`/the bootstrap CI. `n` in
    the verdict is the count actually used, not the raw eligible count.
    """
    eligible = [m for m in per_post if m["post_id"] in eligible_post_ids]

    all_control_jsds = [jsd for m in eligible for jsd in m["jsd_controls"]]
    sensitivity = mean_or_nan(all_control_jsds)
    raw_gender_effect = mean_or_nan([m["jsd_gender"] for m in eligible])

    deltas = [m["delta"] for m in eligible]
    directions = [m["direction"] for m in eligible]
    finite_deltas = [d for d in deltas if math.isfinite(d)]
    finite_directions = [d for d in directions if math.isfinite(d)]
    n = len(finite_deltas)

    mean_delta = mean_or_nan(finite_deltas)
    p_delta = paired_ttest_greater(finite_deltas)
    p_delta_bonferroni = bonferroni(p_delta, n_success_models)

    mean_direction = mean_or_nan(finite_directions)
    direction_ci = bootstrap_ci(finite_directions, derive_rng(seed, DIRECTION_CI, model_index))
    p_direction = ttest_two_sided(finite_directions)

    flip_rate_gender = mean_or_nan([float(m["flip_gender"]) for m in eligible])
    flip_rate_controls = mean_or_nan([m["flip_controls_rate"] for m in eligible])
    effect_size = _effect_size(finite_deltas)

    summary = {
        "sensitivity": sensitivity,
        "raw_gender_effect": raw_gender_effect,
        "mean_delta": mean_delta,
        "p_delta": p_delta,
        "p_delta_bonferroni": p_delta_bonferroni,
        "mean_direction": mean_direction,
        "direction_ci": list(direction_ci),
        "p_direction": p_direction,
        "flip_rate_gender": flip_rate_gender,
        "flip_rate_controls": flip_rate_controls,
        "effect_size": effect_size,
        "n_eligible": n,
        "verdict": model_verdict(p_delta_bonferroni, mean_direction, p_direction, n, alpha),
    }
    return summary, deltas, directions


def _effect_size(deltas: Sequence[float]) -> float:
    """Standardised mean Δ, `mean / sample sd` (docs/PLAN.md §6). Scale-free,
    so it can be read across model types; nan if n < 2 or sd is 0."""
    if len(deltas) < 2:
        return _NAN
    sd = statistics.stdev(deltas)
    if sd == 0 or not math.isfinite(sd):
        return _NAN
    return float(statistics.fmean(deltas) / sd)


def compute_accuracy(per_post: Sequence[dict], gold: dict[str, str]) -> tuple[float, int]:
    """Share of rows with a gold label whose original text's most probable
    level matches it (the competence check, docs/PLAN.md §6). Every labelled
    row counts, eligible for the bias statistics or not. `(nan, 0)` if no
    row has a label."""
    hits = [m["predicted_original"] == gold[m["post_id"]] for m in per_post if m["post_id"] in gold]
    if not hits:
        return _NAN, 0
    return float(sum(hits) / len(hits)), len(hits)
