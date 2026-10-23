"""Build the shared perturbation plan for a dataset. See DESIGN.md §5, §7 and
docs/DATA.md §2.4.

Both perturbations (the targeted swap on the run's axis, and the baseline
controls) are generated **once per row** here and reused for every model,
so every model sees identical inputs (DESIGN §5). The targeted swap comes
from the row's tags when it has them for the axis, otherwise from the
axis's lexicon. This module also flattens all texts into a single list for
one `Scorer.score()` call per model, and renders a row's plan into the
JSON-serialisable `posts[]` shape.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from biascompare.axes import BUILTIN_AXES, AxisSpec, get_axis
from biascompare.inputs import Post
from biascompare.perturb import (
    ControlGenerator,
    ControlSet,
    ControlsSpec,
    Perturbation,
    build_control_generator,
    close_control_generator,
)
from biascompare.perturb.tagged import apply_tagged_swap

_NO_CONTROLS = "no_controls"
_MISSING_SWAP_TAGS = "missing_swap_tags"
_SWAP_TAG_NOT_FOUND = "swap_tag_not_found"
_EXCLUDING_PATTERN = re.compile(r"^(no_controls|missing_swap_tags|no_\w+_terms|\w+_mismatch)$")


@dataclass(frozen=True)
class PostPlan:
    """The perturbations generated for one row on one axis, plus its flags.

    `swap_source` is "tags" (the row's tagged swaps), "lexicon" (the axis
    lexicon) or None (no swap could be made)."""

    post: Post
    axis: AxisSpec
    swap: Perturbation
    swap_source: str | None
    control_set: ControlSet
    flags: tuple[str, ...]

    @property
    def group(self) -> str:
        return self.post.group(self.axis.name)


@dataclass(frozen=True)
class TextLayout:
    """Maps a flattened text list back to each row's original/swap/controls.

    `texts[original_index[i]]` is row `i`'s original text,
    `texts[swap_index[i]]` its targeted swap, and
    `[texts[j] for j in control_indices[i]]` its controls.
    """

    texts: tuple[str, ...]
    original_index: tuple[int, ...]
    swap_index: tuple[int, ...]
    control_indices: tuple[tuple[int, ...], ...]

    def shifted(self, offset: int) -> "TextLayout":
        """The same layout with every index moved by `offset` (for scoring
        several axes' texts in one call)."""
        return TextLayout(
            texts=self.texts,
            original_index=tuple(i + offset for i in self.original_index),
            swap_index=tuple(i + offset for i in self.swap_index),
            control_indices=tuple(tuple(i + offset for i in row) for row in self.control_indices),
        )


def lexicon_vocabulary() -> frozenset[str]:
    """Every word of every built-in axis lexicon. Controls never change or
    introduce these, whichever axis a run tests."""
    words: set[str] = set()
    for spec in BUILTIN_AXES.values():
        swapper = spec.swapper()
        if swapper is not None:
            words |= swapper.vocabulary
    return frozenset(words)


def build_post_plans(
    posts: Sequence[Post],
    n_controls: int,
    seed: int,
    axis: AxisSpec | str = "gender",
    controls: ControlsSpec | ControlGenerator | None = None,
    row_indices: Sequence[int] | None = None,
) -> tuple[PostPlan, ...]:
    """Build the shared, seeded perturbation plan for every row (DESIGN §5).

    The targeted swap is deterministic (tags or lexicon); the per-row
    `np.random.default_rng([seed, row_index])` only drives control sampling.
    `row_indices` are the rows' positions in the dataset file (default:
    their position in `posts`), so a row gets the same controls whether or
    not other rows are filtered out. `controls` is a `ControlsSpec`
    (default: the synonym list) or an already built generator; a generator
    built here is closed before returning, so masked-LM helper models never
    stay loaded while the models under test are scored.
    """
    spec = get_axis(axis)
    swapper = spec.swapper()
    if controls is None or isinstance(controls, ControlsSpec):
        generator = build_control_generator(controls or ControlsSpec(), extra_protected=lexicon_vocabulary())
        owns_generator = True
    else:
        generator = controls
        owns_generator = False
    indices = list(row_indices) if row_indices is not None else list(range(len(posts)))

    try:
        plans: list[PostPlan] = []
        for index, post in zip(indices, posts):
            rng = np.random.default_rng([seed, index])
            swap, source, missing = _targeted_swap(post, spec, swapper)
            control_set = generator.generate(
                post.text,
                n_words=swap.n_changed,
                k=n_controls,
                rng=rng,
                row_synonyms=post.synonyms or None,
                row_protected=post.swap_words(),
            )
            flags = _post_flags(spec, swapper, post, source, missing, swap, control_set)
            plans.append(
                PostPlan(post=post, axis=spec, swap=swap, swap_source=source, control_set=control_set, flags=flags)
            )
    finally:
        if owns_generator:
            close_control_generator(generator)
    return tuple(plans)


def _targeted_swap(post: Post, spec: AxisSpec, swapper) -> tuple[Perturbation, str | None, list[str]]:
    pairs = post.swaps.get(spec.name)
    if pairs:
        swap, missing = apply_tagged_swap(post.text, pairs, kind=spec.name)
        return swap, "tags", missing
    if swapper is not None:
        swap = swapper.swap(post.text)
        return Perturbation(kind=spec.name, text=swap.text, changes=swap.changes), "lexicon", []
    return Perturbation(kind=spec.name, text=post.text, changes=()), None, []


def _post_flags(
    spec: AxisSpec,
    swapper,
    post: Post,
    source: str | None,
    missing: Iterable[str],
    swap: Perturbation,
    control_set: ControlSet,
) -> tuple[str, ...]:
    flags: list[str] = []
    if source is None:
        flags.append(_MISSING_SWAP_TAGS)
    else:
        if source == "lexicon" and swapper.detect_group(post.text) != post.group(spec.name):
            flags.append(f"{spec.name}_mismatch")
        if swap.n_changed == 0:
            flags.append(f"no_{spec.name}_terms")
        if source == "tags" and list(missing):
            flags.append(_SWAP_TAG_NOT_FOUND)
    if control_set.effective_k == 0:
        flags.append(_NO_CONTROLS)
    flags.extend(control_set.flags)
    return tuple(flags)


def is_eligible(plan: PostPlan) -> bool:
    """False if `plan` must be excluded from statistics (docs/DATA.md §2.4:
    a group mismatch, a swap that changed no words or could not be made, or
    zero usable controls)."""
    return is_eligible_flags(plan.flags)


def is_eligible_flags(flags: Sequence[str]) -> bool:
    """The same eligibility rule as `is_eligible`, applied directly to a
    `flags` sequence (e.g. from a serialised `posts[]` entry), so callers
    working from a results dict, like `viz.py`, filter identically."""
    return not any(_EXCLUDING_PATTERN.match(flag) for flag in flags)


def build_text_layout(plans: Sequence[PostPlan]) -> TextLayout:
    """Flatten every row's original/swap/control texts into one list, so a
    model can be scored with a single `score()` call."""
    texts: list[str] = []
    original_index: list[int] = []
    swap_index: list[int] = []
    control_indices: list[tuple[int, ...]] = []

    for plan in plans:
        original_index.append(len(texts))
        texts.append(plan.post.text)
        swap_index.append(len(texts))
        texts.append(plan.swap.text)

        indices: list[int] = []
        for perturbation in plan.control_set.perturbations:
            indices.append(len(texts))
            texts.append(perturbation.text)
        control_indices.append(tuple(indices))

    return TextLayout(
        texts=tuple(texts),
        original_index=tuple(original_index),
        swap_index=tuple(swap_index),
        control_indices=tuple(control_indices),
    )


def post_plan_to_dict(plan: PostPlan) -> dict:
    """Render `plan` into the `posts[]` shape (docs/DATA.md §5)."""
    return {
        "id": plan.post.id,
        "group": plan.group,
        "attributes": dict(plan.post.attributes),
        "label": plan.post.label,
        "meta": dict(plan.post.meta),
        "original": plan.post.text,
        "swap": {**_perturbation_to_dict(plan.swap), "source": plan.swap_source},
        "controls": [_perturbation_to_dict(p) for p in plan.control_set.perturbations],
        "n_words_changed": plan.swap.n_changed,
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
