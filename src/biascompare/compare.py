"""Orchestration: run the comparative bias test and assemble results. See DESIGN.md
and docs/PLAN.md.

`compare_models` is the entry point for an actual run (`run(config)` in
`biascompare.config` wraps it). It loads the dataset, builds the
perturbation plan once (shared across every model, DESIGN §5), scores each
model in turn against that identical input set, and assembles the results
dict (DESIGN §7, extended as described in docs/PLAN.md §8), including
verdict text and, for any model that fails to load or score, a recorded
error instead of a crash.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from biascompare.axes import axis_groups
from biascompare.comparisons import build_comparison
from biascompare.inputs import ModelSpec, Post, load_posts
from biascompare.metrics import Distribution, validate_distribution
from biascompare.model_metrics import compute_accuracy, compute_model_summary, compute_per_post_metrics
from biascompare.perturb import ControlGenerator, ControlsSpec
from biascompare.perturbation_plan import (
    PostPlan,
    TextLayout,
    build_post_plans,
    build_text_layout,
    is_eligible,
    post_plan_to_dict,
)
from biascompare.results import save_results
from biascompare.scoring.base import Scorer, ScorerFactory
from biascompare.scoring.registry import check_scorer_type, default_scorer_factory
from biascompare.tasks import TaskSpec, get_task
from biascompare.verdicts import build_summary_verdict

__all__ = ["compare_models", "preview_perturbations", "save_results"]

SCHEMA_VERSION = "0.2"
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
    axis: str = "gender",
    controls: ControlsSpec | str | ControlGenerator | None = None,
) -> dict:
    """Run the comparative bias test and return the results dict.

    - `task`: a `TaskSpec` or built-in name (`toxicity`, `toxicity-3`,
      `toxicity-4`); default the 2-level `toxicity`. `instruction`, if
      given, replaces the task's instruction (only prompted models read it).
    - `axis`: the protected axis whose lexicon drives the targeted swap.
    - `controls`: a `ControlsSpec`, a method name (`synonyms`,
      `masked-lm`), or a ready generator; default the synonym list.

    Every model is scored over the *same* seeded perturbations (built once,
    before any model is loaded). A model that fails to load or score, or
    returns malformed distributions, gets an `error` string and is excluded
    from the summary/comparisons; the run continues for the rest. Every
    successfully scored model is compared against the first successfully
    scored model (the reference).
    """
    task = get_task(task)
    if instruction is not None:
        task = replace(task, instruction=instruction)
    controls = _resolve_controls(controls, device)
    _validate_inputs(models, n_controls, alpha, scorer_factory)
    factory = scorer_factory or default_scorer_factory

    posts = load_posts(dataset, axis=axis)
    gold = _gold_labels(posts, task)
    plans = build_post_plans(posts, n_controls=n_controls, seed=seed, axis=axis, controls=controls)
    layout = build_text_layout(plans)
    eligible_ids = {plan.post.id for plan in plans if is_eligible(plan)}
    n_eligible = len(eligible_ids)
    groups = axis_groups(axis)

    outcomes = [_score_model(spec, task, device, factory, layout.texts) for spec in models]
    n_success = sum(1 for outcome in outcomes if outcome.error is None)

    model_entries, run_model_entries, successful = _process_models(
        models, outcomes, plans, layout, eligible_ids, n_success, seed, alpha, task, groups, axis, gold
    )
    comparisons = _build_comparisons(successful, seed, alpha, n_eligible)

    return {
        "schema_version": SCHEMA_VERSION,
        "run": _run_block(
            dataset, posts, n_controls, seed, alpha, task, axis, controls, run_model_entries
        ),
        "posts": [post_plan_to_dict(plan) for plan in plans],
        "models": model_entries,
        "comparisons": comparisons,
        "summary_verdict": build_summary_verdict(comparisons, n_eligible, alpha),
        "warnings": _run_warnings(plans, eligible_ids, groups, axis, controls),
    }


def _process_models(
    models: Sequence[ModelSpec],
    outcomes: list[_ScoreOutcome],
    plans: Sequence[PostPlan],
    layout: TextLayout,
    eligible_ids: set[str],
    n_success: int,
    seed: int,
    alpha: float,
    task: TaskSpec,
    groups: tuple[str, str],
    axis: str,
    gold: dict[str, str],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Turn each model's `_ScoreOutcome` into its `models[]`/`run.models[]`
    entries, and collect the successfully scored models' deltas/directions
    for the comparisons step. One pass, in the caller's model order."""
    model_entries: list[dict] = []
    run_model_entries: list[dict] = []
    successful: list[dict] = []
    model_index = 0

    for spec, outcome in zip(models, outcomes):
        run_model_entries.append(_run_model_entry(spec, outcome))
        if outcome.error is not None:
            model_entries.append(
                {"name": spec.name, "error": outcome.error, "per_post": [], "summary": None}
            )
            continue

        per_post = [
            compute_per_post_metrics(plan, layout, index, outcome.scores, task, groups, axis)
            for index, plan in enumerate(plans)
        ]
        summary, deltas, directions = compute_model_summary(
            per_post, eligible_ids, n_success, seed, model_index, alpha
        )
        summary["accuracy"], summary["n_labeled"] = compute_accuracy(per_post, gold)
        model_entries.append({"name": spec.name, "error": None, "per_post": per_post, "summary": summary})
        successful.append({"label": spec.name, "deltas": deltas, "directions": directions})
        model_index += 1

    return model_entries, run_model_entries, successful


def preview_perturbations(
    dataset: str | Path,
    n_controls: int = 5,
    seed: int = 0,
    axis: str = "gender",
    controls: ControlsSpec | str | ControlGenerator | None = None,
) -> list[dict]:
    """Build and render the perturbation plan without scoring any model.

    Shows the targeted swaps and controls a run would score. Uses the exact
    same plan-building code as `compare_models`, so the `posts` it previews
    are identical to the ones a real run with the same arguments would use.
    """
    _validate_n_controls(n_controls)
    posts = load_posts(dataset, axis=axis)
    plans = build_post_plans(
        posts, n_controls=n_controls, seed=seed, axis=axis, controls=_resolve_controls(controls)
    )
    return [post_plan_to_dict(plan) for plan in plans]


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
    models: Sequence[ModelSpec], n_controls: int, alpha: float, scorer_factory: ScorerFactory | None
) -> None:
    if len(models) < 2:
        raise ValueError(f"compare_models requires at least 2 models, got {len(models)}")
    _validate_n_controls(n_controls)
    if not (0 < alpha < 1):
        raise ValueError(f"alpha must be in the open interval (0, 1), got {alpha}")
    if scorer_factory is None:
        for spec in models:
            check_scorer_type(spec.type)


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
        return _ScoreOutcome(None, scores, scorer.scorer_type, scorer.uses_instruction, metadata)
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


def _run_model_entry(spec: ModelSpec, outcome: _ScoreOutcome) -> dict:
    entry = {
        "name": spec.name,
        "toxic_label": spec.toxic_label,
        "scorer": outcome.scorer_type,
        "instruction_used": outcome.uses_instruction,
        "type": spec.type,
    }
    public = spec.to_public_dict()
    for key in ("label_map", "settings"):
        if key in public:
            entry[key] = public[key]
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
    axis: str,
    controls: ControlsSpec | ControlGenerator,
    run_model_entries: list[dict],
) -> dict:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "n_posts": len(posts),
        "axis": axis,
        "n_controls": n_controls,
        "seed": seed,
        "alpha": alpha,
        "instruction": task.instruction,
        "paper": _PAPER_CITATION,
        "models": run_model_entries,
        "task": task.to_public_dict(),
        "controls": _controls_public(controls),
    }


def _controls_public(controls: ControlsSpec | ControlGenerator) -> dict:
    if isinstance(controls, ControlsSpec):
        return controls.to_public_dict()
    return {"method": "custom", "generator": type(controls).__name__}


def _run_warnings(
    plans: Sequence[PostPlan],
    eligible_ids: set[str],
    groups: tuple[str, str],
    axis: str,
    controls: ControlsSpec | ControlGenerator,
) -> list[str]:
    """Plain-English data-quality warnings about the run (docs/DATA.md §6)."""
    warnings: list[str] = []
    n_rows = len(plans)
    excluded = [plan for plan in plans if plan.post.id not in eligible_ids]
    if excluded:
        reasons = Counter(
            flag for plan in excluded for flag in plan.flags
            if flag in ("gender_mismatch", "no_gender_terms", "no_controls")
        )
        detail = ", ".join(f"{flag}: {count}" for flag, count in sorted(reasons.items()))
        warnings.append(f"{len(excluded)} of {n_rows} rows are excluded from statistics ({detail}).")
        if (
            len(excluded) / n_rows > _EXCLUDED_SHARE_WARNING
            and reasons.get("no_controls")
            and isinstance(controls, ControlsSpec)
            and controls.method == "synonyms"
        ):
            warnings.append(
                "Most no_controls rows come from the synonym list only covering the sample posts; "
                "use controls method 'masked-lm' for this dataset (docs/DATA.md §2.4)."
            )

    counts = Counter(plan.post.group(axis) for plan in plans if plan.post.id in eligible_ids)
    sizes = [counts.get(group, 0) for group in groups]
    if max(sizes) > 0 and min(sizes) / max(sizes) < _GROUP_BALANCE_WARNING:
        warnings.append(
            f"Eligible rows are unbalanced between groups ({groups[0]}: {sizes[0]}, "
            f"{groups[1]}: {sizes[1]}), which can confound direction (docs/DATA.md §2.5)."
        )
    n_eligible = len(eligible_ids)
    if n_eligible < _LOW_N_WARNING:
        warnings.append(
            f"Only {n_eligible} rows are eligible for statistics, so only very large effects are "
            "detectable (docs/DATA.md §2.5)."
        )
    return warnings


def _build_comparisons(successful: list[dict], seed: int, alpha: float, n_eligible: int) -> list[dict]:
    if len(successful) < 2:
        return []
    reference = successful[0]
    comparisons = []
    for index, model in enumerate(successful[1:]):
        comparisons.append(
            build_comparison(
                reference["label"],
                reference["deltas"],
                reference["directions"],
                model["label"],
                model["deltas"],
                model["directions"],
                seed,
                index,
                alpha,
                n_eligible,
            )
        )
    return comparisons
