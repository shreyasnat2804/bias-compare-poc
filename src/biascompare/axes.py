"""Registry of protected axes. See docs/PLAN.md §3.3.

An axis is a lexicon file of word pairs between exactly two groups (see
`resources/gender_lexicon.json`). Adding a new word-level axis means adding
a lexicon file and an entry here; no other code changes. Only `gender`
exists today.
"""

from __future__ import annotations

from functools import lru_cache

from biascompare.perturb import GenderSwapper

AXES: dict[str, str] = {"gender": "gender_lexicon.json"}


def _require_known(axis: str) -> None:
    if axis not in AXES:
        raise ValueError(f"Unknown axis {axis!r}; available axes are {sorted(AXES)}")


@lru_cache(maxsize=None)
def load_swapper(axis: str) -> GenderSwapper:
    """The lexicon swapper for `axis` (cached: lexicons are read-only)."""
    _require_known(axis)
    return GenderSwapper.from_resource(AXES[axis])


def axis_groups(axis: str) -> tuple[str, str]:
    """The axis's two groups, in lexicon order. `direction` is the second
    group's version minus the first group's version."""
    return tuple(load_swapper(axis).groups)
