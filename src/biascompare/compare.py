"""Orchestration: run the comparative bias test and assemble results. See DESIGN.md
and docs/PLAN.md.

- `compare_models` tests one axis and returns one results dict.
- `compare_axes` tests several axes (gender and disability in the study) in
  one run: each model is loaded and scored once over every axis's texts,
  and the results hold one block per axis.

Both load the dataset, build each axis's perturbation plan once (shared
across every model, DESIGN §5), score each model in turn against that
identical input set, and assemble the results (docs/DATA.md §5): per-row
metrics, per-model summaries, model-vs-model comparisons, a ranking, verdict
text and data-quality warnings. A model that fails to load or score gets an
error recorded instead of stopping the run.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from biascompare.axes import AxisSpec, get_axis
from biascompare.comparisons import COMPARISON_MODES, build_comparisons
from biascompare.inputs import ModelSpec, Post, load_posts
from biascompare.metrics import Distribution, validate_distribution
from biascompare.model_metrics import compute_accuracy, compute_model_summary, compute_per_post_metrics
from biascompare.perturb import ControlGenerator, ControlsSpec, build_control_generator, close_control_generator
from biascompare.perturbation_plan import (
    PostPlan,
    TextLayout,
    build_post_plans,
    build_text_layout,
    is_eligible,
    is_eligible_flags,
    lexicon_vocabulary,
    post_plan_to_dict,
)
from biascompare.results import save_results
from biascompare.scoring.base import Scorer, ScorerFactory
from biascompare.scoring.registry import check_scorer_type, default_scorer_factory
from biascompare.tasks import TaskSpec, get_task
from biascompare.verdicts import build_summary_verdict

__all__ = ["compare_models", "compare_axes", "preview_perturbations", "axis_results", "save_results"]

SCHEMA_VERSION = "0.3"
_PAPER_CITATION = "Yang et al. 2026, arXiv:2605.01048"
_EXCLUDED_SHARE_WARNING = 0.05
_GROUP_BALANCE_WARNING = 0.8
_LOW_N_WARNING = 30


@dataclass(frozen=True)
class _ScoreOutcome:
    """The result of trying to load and run one model (DESIGN §7)."""

    error: str | None
    scores: list[Distribution] | None
    scorer_type: str | None
    uses_instruction: bool | None
    metadata: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class _AxisPlan:
    axis: AxisSpec
    plans: tuple[PostPlan, ...]
    layout: TextLayout
    eligible_ids: frozenset[str]


def compare_models(
    models: Sequence[ModelSpec],
    dataset: str | Path,
    instruction: str | None = None,
    n_controls: int = 5,
    seed: int = 0,
    alpha: float = 0.05,
    device: str = "auto",
    scorer_factory: ScorerFactory | None = None,
    task: TaskSpec | str | None = None,
    axis: AxisSpec | str = "gender",
    controls: ControlsSpec | str | ControlGenerator | None = None,
    comparisons: str = "all-pairs",
) -> dict:
    """Run the comparative bias test on one axis and return the results dict.

    - `models`: two or more `ModelSpec`s; each `id` (default: `name`) must
      be unique. A model's `prompt` wraps every post it scores.
    - `task`: a `TaskSpec` or built-in name (`toxicity`, `toxicity-3`,
      `toxicity-4`); default the 2-level `toxicity`. `instruction`, if
      given, replaces the task's default instruction.
    - `axis`: a built-in axis name (`gender`, `disability`) or an `AxisSpec`.
    - `controls`: a `ControlsSpec`, a method name (`synonyms`, `tagged`,
      `masked-lm`), or a ready generator; default the synonym list.
    - `comparisons`: `all-pairs` (default) or `reference` (each model
      against the first one).

    Every model is scored over the *same* seeded perturbations (built once,
    before any model is loaded). A model that fails to load or score, or
    returns malformed distributions, gets an `error` string and is left out
    of the summaries and comparisons; the run continues for the rest.
    """
    report = compare_axes(
        models, dataset, axes=[axis], instruction=instruction, n_controls=n_controls, seed=seed,
        alpha=alpha, device=device, scorer_factory=scorer_factory, task=task, controls=controls,
        comparisons=comparisons,
    )
    ((name, block),) = report["axes"].items()
    return {"schema_version": report["schema_version"], "run": {**report["run"], "axis": name}, **block}


def compare_axes(
    models: Sequence[ModelSpec],
    dataset: str | Path,
    axes: Sequence[AxisSpec | str] = ("gender",),
    instruction: str | None = None,
    n_controls: int = 5,
    seed: int = 0,
    alpha: float = 0.05,
    device: str = "auto",
    scorer_factory: ScorerFactory | None = None,
    task: TaskSpec | str | None = None,
    controls: ControlsSpec | str | ControlGenerator | None = None,
    comparisons: str = "all-pairs",
) -> dict:
    """Run the test on several axes at once (arguments as `compare_models`).

    Each axis uses the rows that have an attribute for it (rows with none of
    the run's axes are not scored; `run.n_posts` counts the rows used). Every model is
    loaded once and scores all axes' texts in one pass. Returns
    `{"schema_version", "run", "axes": {axis name: results block}}`; use
    `axis_results(report, name)` for a single-axis view.
    """
    task = get_task(task)
    if instruction is not None:
        task = replace(task, instruction=instruction)
    axis_specs = [get_axis(axis) for axis in axes]
    controls = _resolve_controls(controls, device)
    _validate_inputs(models, axis_specs, n_controls, alpha, scorer_factory, task, comparisons)
    factory = scorer_factory or default_scorer_factory

    posts = load_posts(dataset, axes=axis_specs, require_axis=False)
    gold = _gold_labels(posts, task)

    axis_plans, all_texts = _plan_axes(posts, axis_specs, n_controls, seed, controls)
    posts = [post for post in posts if any(spec.name in post.attributes for spec in axis_specs)]
    outcomes = [_score_model(spec, task, device, factory, all_texts) for spec in models]
    n_success = sum(1 for outcome in outcomes if outcome.error is None)

    blocks = {
        plan.axis.name: _axis_block(plan, models, outcomes, n_success, seed, alpha, task, gold, controls, comparisons)
        for plan in axis_plans
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "run": _run_block(
            dataset, posts, n_controls, seed, alpha, task, axis_specs, controls, comparisons,
            [_run_model_entry(spec, outcome, task, posts) for spec, outcome in zip(models, outcomes)],
        ),
        "axes": blocks,
    }


def axis_results(report: dict, axis: str) -> dict:
    """A single-axis results dict (the `compare_models` shape) from a
    `compare_axes` report, so the viz and table helpers apply unchanged."""
    if "axes" not in report:
        return report
    if axis not in report["axes"]:
        raise ValueError(f"Axis {axis!r} is not in this report; it has {sorted(report['axes'])}")
    return {"schema_version": report["schema_version"], "run": {**report["run"], "axis": axis}, **report["axes"][axis]}


def preview_perturbations(
    dataset: str | Path,
    n_controls: int = 5,
    seed: int = 0,
    axis: AxisSpec | str = "gender",
    controls: ControlsSpec | str | ControlGenerator | None = None,
) -> list[dict]:
    """Build and render one axis's perturbation plan without scoring any model.

    Uses the exact same plan-building code as `compare_models`, so the
    `posts` it previews are identical to the ones a real run with the same
    arguments would use.
    """
    _validate_n_controls(n_controls)
    spec = get_axis(axis)
    posts = load_posts(dataset, axes=[spec], require_axis=False)
    (plan,), _ = _plan_axes(posts, [spec], n_controls, seed, _resolve_controls(controls))
    return [post_plan_to_dict(row) for row in plan.plans]


def _plan_axes(
    posts: Sequence[Post],
    axis_specs: Sequence[AxisSpec],
    n_controls: int,
    seed: int,
    controls: ControlsSpec | ControlGenerator,
) -> tuple[list[_AxisPlan], list[str]]:
    """Build every axis's plan with one control generator (helper models,
    if any, are loaded once and freed before scoring), and lay all texts out
    in one list."""
    if isinstance(controls, ControlsSpec):
        generator = build_control_generator(controls, extra_protected=lexicon_vocabulary())
        owns_generator = True
    else:
        generator, owns_generator = controls, False

    axis_plans: list[_AxisPlan] = []
    all_texts: list[str] = []
    try:
        for spec in axis_specs:
            rows = [(index, post) for index, post in enumerate(posts) if spec.name in post.attributes]
            if not rows:
                raise ValueError(f"No row has an attributes.{spec.name} entry, so axis {spec.name!r} cannot be tested")
            plans = build_post_plans(
                [post for _, post in rows], n_controls=n_controls, seed=seed, axis=spec,
                controls=generator, row_indices=[index for index, _ in rows],
            )
            layout = build_text_layout(plans)
            axis_plans.append(
                _AxisPlan(
                    axis=spec,
                    plans=plans,
                    layout=layout.shifted(len(all_texts)),
                    eligible_ids=frozenset(plan.post.id for plan in plans if is_eligible(plan)),
                )
            )
            all_texts.extend(layout.texts)
    finally:
        if owns_generator:
            close_control_generator(generator)
    return axis_plans, all_texts


def _axis_block(
    axis_plan: _AxisPlan,
    models: Sequence[ModelSpec],
    outcomes: list[_ScoreOutcome],
    n_success: int,
    seed: int,
    alpha: float,
    task: TaskSpec,
    gold: dict[str, str],
    controls: ControlsSpec | ControlGenerator,
    comparison_mode: str,
) -> dict:
    """The results block for one axis: posts, models, comparisons, ranking,
    verdict and warnings."""
    axis = axis_plan.axis
    n_eligible = len(axis_plan.eligible_ids)
    model_entries: list[dict] = []
    successful: list[dict] = []
    warnings = _data_warnings(axis_plan, controls)

    for spec, outcome in zip(models, outcomes):
        if outcome.error is not None:
            model_entries.append(
                {"id": spec.display_id, "name": spec.name, "error": outcome.error, "per_post": [], "summary": None}
            )
            continue
        per_post = [
            compute_per_post_metrics(plan, axis_plan.layout, index, outcome.scores, task)
            for index, plan in enumerate(axis_plan.plans)
        ]
        summary, deltas, directions = compute_model_summary(
            per_post, set(axis_plan.eligible_ids), n_success, seed, len(successful), alpha, axis
        )
        summary["accuracy"], summary["n_labeled"] = compute_accuracy(per_post, gold)
        summary["n_truncated"] = _n_truncated(axis_plan, outcome)
        if summary["n_truncated"]:
            warnings.append(
                f"{spec.display_id}: {summary['n_truncated']} rows had an input longer than the model's "
                f"max_length and were truncated (prompted inputs are cut from the left, so the post is kept)."
            )
        model_entries.append(
            {"id": spec.display_id, "name": spec.name, "error": None, "per_post": per_post, "summary": summary}
        )
        successful.append({"label": spec.display_id, "deltas": deltas, "directions": directions})

    comparisons = build_comparisons(successful, seed, alpha, n_eligible, comparison_mode, axis)
    return {
        "posts": [post_plan_to_dict(plan) for plan in axis_plan.plans],
        "models": model_entries,
        "comparisons": comparisons,
        "ranking": _ranking(model_entries, alpha),
        "summary_verdict": build_summary_verdict(comparisons, n_eligible, alpha, axis),
        "warnings": warnings,
    }


def _n_truncated(axis_plan: _AxisPlan, outcome: _ScoreOutcome) -> int | None:
    """Rows of this axis where any arm's input was truncated, or None if the
    scorer does not report truncation."""
    truncated = outcome.diagnostics.get("truncated")
    if truncated is None:
        return None
    layout = axis_plan.layout
    count = 0
    for i in range(len(axis_plan.plans)):
        arms = [layout.original_index[i], layout.swap_index[i], *layout.control_indices[i]]
        count += any(truncated[j] for j in arms)
    return count


def _ranking(model_entries: list[dict], alpha: float) -> list[dict]:
    """Successfully scored models ordered by mean Δ, most bias beyond
    wording sensitivity first (models without a finite mean Δ last)."""
    scored = [entry for entry in model_entries if entry["summary"] is not None]

    def key(entry: dict) -> tuple:
        value = entry["summary"]["mean_delta"]
        return (0, -value) if math.isfinite(value) else (1, 0.0)

    ranking = []
    for rank, entry in enumerate(sorted(scored, key=key), start=1):
        summary = entry["summary"]
        p_bonf = summary["p_delta_bonferroni"]
        ranking.append(
            {
                "rank": rank,
                "model": entry["id"],
                "mean_delta": summary["mean_delta"],
                "effect_size": summary["effect_size"],
                "p_delta_bonferroni": p_bonf,
                "significant": bool(math.isfinite(p_bonf) and p_bonf < alpha),
                "mean_direction": summary["mean_direction"],
                "p_direction": summary["p_direction"],
            }
        )
    return ranking


def _resolve_controls(
    controls: ControlsSpec | str | ControlGenerator | None, device: str = "auto"
) -> ControlsSpec | ControlGenerator:
    """Normalise `controls` to a spec or generator. A spec left on device
    "auto" follows the run's `device`, so helper models land where the
    models under test do."""
    if controls is None:
        controls = ControlsSpec()
    elif isinstance(controls, str):
        controls = ControlsSpec(method=controls)
    if isinstance(controls, ControlsSpec) and controls.device == "auto" and device != "auto":
        controls = replace(controls, device=device)
    return controls


def _validate_inputs(
    models: Sequence[ModelSpec],
    axis_specs: Sequence[AxisSpec],
    n_controls: int,
    alpha: float,
    scorer_factory: ScorerFactory | None,
    task: TaskSpec,
    comparisons: str,
) -> None:
    if len(models) < 2:
        raise ValueError(f"compare_models requires at least 2 models, got {len(models)}")
    ids = [spec.display_id for spec in models]
    duplicates = sorted({model_id for model_id in ids if ids.count(model_id) > 1})
    if duplicates:
        raise ValueError(
            f"Model ids must be unique, but {duplicates} appear more than once. Give each entry an id "
            "(for example M0 and M1 for the same model with and without a prompt)."
        )
    names = [spec.name for spec in axis_specs]
    if len(set(names)) != len(names):
        raise ValueError(f"Each axis may appear only once, got {names}")
    _validate_n_controls(n_controls)
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha}")
    if comparisons not in COMPARISON_MODES:
        raise ValueError(f"comparisons must be one of {list(COMPARISON_MODES)}, got {comparisons!r}")
    for spec in models:
        if scorer_factory is None:
            check_scorer_type(spec.type)
        if spec.prompt is not None:
            try:
                spec.prompt.check_task(task)
            except ValueError as exc:
                raise ValueError(f"Model {spec.display_id!r}: {exc}") from None


def _validate_n_controls(n_controls: int) -> None:
    if n_controls < 1:
        raise ValueError(f"n_controls must be >= 1, got {n_controls}")


def _gold_labels(posts: Sequence[Post], task: TaskSpec) -> dict[str, str]:
    """Map post id -> canonical level name for rows with a gold `label`.
    Raises ValueError listing rows whose label is not one of the task's levels."""
    gold: dict[str, str] = {}
    bad: list[str] = []
    for post in posts:
        if post.label is None:
            continue
        try:
            gold[post.id] = task.levels[task.level_index(post.label)]
        except ValueError:
            bad.append(f"{post.id}={post.label!r}")
    if bad:
        shown = ", ".join(bad[:5]) + (f" and {len(bad) - 5} more" if len(bad) > 5 else "")
        raise ValueError(
            f"Gold labels must be levels of task {task.name!r} {list(task.levels)}; got {shown}"
        )
    return gold


def _score_model(
    spec: ModelSpec,
    task: TaskSpec,
    device: str,
    factory: ScorerFactory,
    texts: Sequence[str],
) -> _ScoreOutcome:
    """Load `spec`, score every text in one call, check the output, and
    always `close()` the scorer. Any failure in loading, scoring or checking
    is captured as `error` instead of propagating, so one bad model never
    aborts the run (DESIGN §7)."""
    scorer: Scorer | None = None
    try:
        scorer = factory(spec, task, device)
        scores = scorer.score(texts)
        _check_scores(scores, texts, task)
        metadata = dict(getattr(scorer, "metadata", None) or {})
        diagnostics = dict(getattr(scorer, "diagnostics", None) or {})
        return _ScoreOutcome(None, scores, scorer.scorer_type, scorer.uses_instruction, metadata, diagnostics)
    except Exception as exc:  # noqa: BLE001 - a failing model must not abort the run
        return _ScoreOutcome(str(exc), None, None, None)
    finally:
        if scorer is not None:
            try:
                scorer.close()
            except Exception:  # noqa: BLE001 - never let cleanup mask the real outcome
                pass


def _check_scores(scores: Sequence[Distribution], texts: Sequence[str], task: TaskSpec) -> None:
    if len(scores) != len(texts):
        raise ValueError(f"scorer returned {len(scores)} distributions for {len(texts)} texts")
    for index, dist in enumerate(scores):
        try:
            validate_distribution(dist, task.n_levels)
        except ValueError as exc:
            raise ValueError(
                f"scorer output for text {index} is not a distribution over the "
                f"{task.n_levels} levels of task {task.name!r}: {exc}"
            ) from None


def _run_model_entry(spec: ModelSpec, outcome: _ScoreOutcome, task: TaskSpec, posts: Sequence[Post]) -> dict:
    entry = {
        "id": spec.display_id,
        "name": spec.name,
        "toxic_label": spec.toxic_label,
        "scorer": outcome.scorer_type,
        "instruction_used": outcome.uses_instruction,
        "type": spec.type,
    }
    public = spec.to_public_dict()
    for key in ("label_map", "settings", "prompt"):
        if key in public:
            entry[key] = public[key]
    if spec.prompt is not None and posts:
        entry["prompt_example"] = spec.prompt.render(posts[0].text, task)
    entry["revision"] = outcome.metadata.get("revision", spec.revision)
    for key, value in outcome.metadata.items():
        entry.setdefault(key, value)
    return entry


def _run_block(
    dataset: str | Path,
    posts: Sequence[Post],
    n_controls: int,
    seed: int,
    alpha: float,
    task: TaskSpec,
    axis_specs: Sequence[AxisSpec],
    controls: ControlsSpec | ControlGenerator,
    comparisons: str,
    run_model_entries: list[dict],
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "n_posts": len(posts),
        "axes": [spec.to_public_dict() for spec in axis_specs],
        "n_controls": n_controls,
        "seed": seed,
        "alpha": alpha,
        "instruction": task.instruction,
        "paper": _PAPER_CITATION,
        "models": run_model_entries,
        "task": task.to_public_dict(),
        "controls": _controls_public(controls),
        "comparisons": comparisons,
    }


def _controls_public(controls: ControlsSpec | ControlGenerator) -> dict:
    if isinstance(controls, ControlsSpec):
        return controls.to_public_dict()
    return {"method": "custom", "generator": type(controls).__name__}


def _data_warnings(axis_plan: _AxisPlan, controls: ControlsSpec | ControlGenerator) -> list[str]:
    """Plain-English data-quality warnings for one axis (docs/DATA.md §6)."""
    warnings: list[str] = []
    plans, eligible_ids, axis = axis_plan.plans, axis_plan.eligible_ids, axis_plan.axis
    n_rows = len(plans)
    excluded = [plan for plan in plans if plan.post.id not in eligible_ids]
    if excluded:
        reasons = Counter(flag for plan in excluded for flag in plan.flags if not is_eligible_flags([flag]))
        detail = ", ".join(f"{flag}: {count}" for flag, count in sorted(reasons.items()))
        warnings.append(f"{len(excluded)} of {n_rows} rows are excluded from statistics ({detail}).")
        if (
            len(excluded) / n_rows > _EXCLUDED_SHARE_WARNING
            and reasons.get("no_controls")
            and isinstance(controls, ControlsSpec)
            and controls.method == "synonyms"
        ):
            warnings.append(
                "Many rows have no controls: the shared synonym list only covers the sample posts. "
                "Tag control synonyms per row (method 'tagged') or use method 'masked-lm' (docs/DATA.md §2.4)."
            )
    not_found = sum("swap_tag_not_found" in plan.flags for plan in plans)
    if not_found:
        warnings.append(
            f"{not_found} rows have tagged {axis.name} swaps that do not occur in their text; check the tags."
        )

    counts = Counter(plan.group for plan in plans if plan.post.id in eligible_ids)
    sizes = [counts.get(group, 0) for group in axis.groups]
    if max(sizes) > 0 and min(sizes) / max(sizes) < _GROUP_BALANCE_WARNING:
        warnings.append(
            f"Eligible rows are unbalanced between groups ({axis.groups[0]}: {sizes[0]}, "
            f"{axis.groups[1]}: {sizes[1]}), which can confound direction (docs/DATA.md §2.5)."
        )
    n_eligible = len(eligible_ids)
    if n_eligible < _LOW_N_WARNING:
        warnings.append(
            f"Only {n_eligible} rows are eligible for statistics, so only very large effects are "
            "detectable (docs/DATA.md §2.5)."
        )
    return warnings
