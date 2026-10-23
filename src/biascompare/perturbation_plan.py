"""Build the shared perturbation plan for a dataset. See DESIGN.md §5, §7.

Both perturbations (targeted gender swap, baseline controls) are
generated **once per post** here and reused for every model, so every model
sees identical inputs (DESIGN §5). This module also flattens those texts
into a single list for one `Scorer.score()` call per model, and renders a
post's plan into the JSON-serialisable shape used by both `posts` in
`compare_models`'s results and by `preview_perturbations`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from biascompare.axes import load_swapper
from biascompare.inputs import Post
from biascompare.perturb import (
    ControlGenerator,
    ControlSet,
    ControlsSpec,
    GenderSwapper,
    Perturbation,
    build_control_generator,
    close_control_generator,
)

_GENDER_MISMATCH = "gender_mismatch"
_NO_GENDER_TERMS = "no_gender_terms"
_NO_CONTROLS = "no_controls"
_EXCLUDING_FLAGS = frozenset({_GENDER_MISMATCH, _NO_GENDER_TERMS, _NO_CONTROLS})


@dataclass(frozen=True)
class PostPlan:
    """The perturbations generated for one post, plus its exclusion flags."""

    post: Post
    gender_swap: Perturbation
    control_set: ControlSet
    flags: tuple[str, ...]


@dataclass(frozen=True)
class TextLayout:
    """Maps a flattened text list back to each post's original/swap/controls.

    `texts[original_index[i]]` is post `i`'s original text,
    `texts[gender_index[i]]` its gender swap, and
    `[texts[j] for j in control_indices[i]]` its controls.
    """

    texts: tuple[str, ...]
    original_index: tuple[int, ...]
    gender_index: tuple[int, ...]
    control_indices: tuple[tuple[int, ...], ...]


def build_post_plans(
    posts: Sequence[Post],
    n_controls: int,
    seed: int,
    axis: str = "gender",
    controls: ControlsSpec | ControlGenerator | None = None,
) -> tuple[PostPlan, ...]:
    """Build the shared, seeded perturbation plan for every post (DESIGN §5).

    The targeted swap is deterministic (a pure lexicon lookup for `axis`);
    the per-post `np.random.default_rng([seed, post_index])` only drives
    control sampling. `controls` is a `ControlsSpec` (default: the synonym
    list), or an already built generator. A generator built here from a
    spec is closed before returning, so masked-LM helper models never stay
    loaded while the models under test are scored.
    """
    swapper = load_swapper(axis)
    if controls is None or isinstance(controls, ControlsSpec):
        generator = build_control_generator(controls or ControlsSpec(), extra_protected=swapper.vocabulary)
        owns_generator = True
    else:
        generator = controls
        owns_generator = False

    try:
        plans: list[PostPlan] = []
        for index, post in enumerate(posts):
            rng = np.random.default_rng([seed, index])
            gender_swap = swapper.swap(post.text)
            control_set = generator.generate(
                post.text, n_words=gender_swap.n_changed, k=n_controls, rng=rng
            )
            flags = _post_flags(swapper, post.text, post.group(axis), gender_swap, control_set)
            plans.append(
                PostPlan(post=post, gender_swap=gender_swap, control_set=control_set, flags=flags)
            )
    finally:
        if owns_generator:
            close_control_generator(generator)
    return tuple(plans)


def _post_flags(
    swapper: GenderSwapper,
    text: str,
    group: str,
    gender_swap: Perturbation,
    control_set: ControlSet,
) -> tuple[str, ...]:
    flags: list[str] = []
    if swapper.detect_group(text) != group:
        flags.append(_GENDER_MISMATCH)
    if gender_swap.n_changed == 0:
        flags.append(_NO_GENDER_TERMS)
    if control_set.effective_k == 0:
        flags.append(_NO_CONTROLS)
    flags.extend(control_set.flags)
    return tuple(flags)


def is_eligible(plan: PostPlan) -> bool:
    """False if `plan` must be excluded from statistics (DESIGN §5: a
    detected-gender mismatch, a gender swap that changed no words, or zero
    usable control perturbations)."""
    return is_eligible_flags(plan.flags)


def is_eligible_flags(flags: Sequence[str]) -> bool:
    """The same eligibility rule as `is_eligible`, applied directly to a
    `flags` sequence (e.g. from a serialised `posts[]` entry) instead of a
    `PostPlan`. Lets callers working from a results dict — like `viz.py` —
    apply identical filtering without reconstructing a `PostPlan`."""
    return not (_EXCLUDING_FLAGS & set(flags))


def build_text_layout(plans: Sequence[PostPlan]) -> TextLayout:
    """Flatten every post's original/gender-swap/control texts into one
    list, so a model can be scored with a single `score()` call."""
    texts: list[str] = []
    original_index: list[int] = []
    gender_index: list[int] = []
    control_indices: list[tuple[int, ...]] = []

    for plan in plans:
        original_index.append(len(texts))
        texts.append(plan.post.text)
        gender_index.append(len(texts))
        texts.append(plan.gender_swap.text)

        indices: list[int] = []
        for perturbation in plan.control_set.perturbations:
            indices.append(len(texts))
            texts.append(perturbation.text)
        control_indices.append(tuple(indices))

    return TextLayout(
        texts=tuple(texts),
        original_index=tuple(original_index),
        gender_index=tuple(gender_index),
        control_indices=tuple(control_indices),
    )


def post_plan_to_dict(plan: PostPlan) -> dict:
    """Render `plan` into the `posts[]` shape from DESIGN.md §7."""
    return {
        "id": plan.post.id,
        "platform": plan.post.platform,
        "gender": plan.post.gender,
        "intended_toxicity": plan.post.intended_toxicity,
        "attributes": dict(plan.post.attributes),
        "label": plan.post.label,
        "meta": dict(plan.post.meta),
        "original": plan.post.text,
        "gender_swap": _perturbation_to_dict(plan.gender_swap),
        "controls": [_perturbation_to_dict(p) for p in plan.control_set.perturbations],
        "n_words_changed": plan.gender_swap.n_changed,
        "effective_k": plan.control_set.effective_k,
        "flags": list(plan.flags),
    }


def _perturbation_to_dict(perturbation: Perturbation) -> dict:
    rendered = {
        "text": perturbation.text,
        "changes": [[change.original, change.replacement] for change in perturbation.changes],
    }
    if perturbation.kind == "control":
        rendered["similarity"] = perturbation.similarity
    return rendered
