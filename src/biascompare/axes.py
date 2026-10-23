"""Protected axes. See docs/PLAN.md §3.3 and docs/DATA.md §2.

An axis names two groups, says which one is the **target** group (the one
`direction` asks about: "does the version referencing this group score as
more toxic?"), and optionally points to a lexicon of word pairs for
generating swaps automatically. Without a lexicon, every row must carry its
own tagged swaps for the axis (docs/DATA.md §2.3).

Built in:

- `gender`: male / female, target female, swaps from `gender_lexicon.json`
  or from row tags.
- `disability`: non-disabled / disabled, target disabled, swaps from row
  tags only (disability references are not word pairs like he/she).

Other axes can be defined with an `AxisSpec` (or an `axes` block in a run
config) without code changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType

from biascompare.perturb import GenderSwapper


@dataclass(frozen=True)
class AxisSpec:
    """One protected axis: two groups, the target group, display labels for
    verdict text, and an optional lexicon (a resource name or a file path)."""

    name: str
    groups: tuple[str, str]
    target_group: str
    group_labels: Mapping[str, str] | None = None
    lexicon: str | None = None
    _label_items: tuple = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or not self.name.replace("_", "").isalnum():
            raise ValueError(f"Axis name must be a non-empty word (letters, digits, _), got {self.name!r}")
        groups = tuple(self.groups)
        if len(groups) != 2 or len(set(groups)) != 2 or not all(isinstance(g, str) and g.strip() for g in groups):
            raise ValueError(f"Axis {self.name!r} needs exactly two different group names, got {list(groups)}")
        object.__setattr__(self, "groups", groups)
        if self.target_group not in groups:
            raise ValueError(f"Axis {self.name!r} target_group {self.target_group!r} is not one of {list(groups)}")
        labels = dict(self.group_labels or {})
        unknown = sorted(set(labels) - set(groups))
        if unknown:
            raise ValueError(f"Axis {self.name!r} group_labels name unknown group(s) {unknown}")
        object.__setattr__(self, "group_labels", MappingProxyType(labels))
        object.__setattr__(self, "_label_items", tuple(sorted(labels.items())))

    def __hash__(self) -> int:
        return hash((self.name, self.groups, self.target_group, self._label_items, self.lexicon))

    @property
    def other_group(self) -> str:
        return self.groups[0] if self.target_group == self.groups[1] else self.groups[1]

    def label(self, group: str) -> str:
        """How verdicts refer to people in `group` ("women", "disabled people")."""
        return self.group_labels.get(group, f"{group} people")

    def swapper(self) -> GenderSwapper | None:
        """The lexicon swapper, or None for a tag-only axis."""
        if self.lexicon is None:
            return None
        swapper = _load_lexicon(self.lexicon)
        if set(swapper.groups) != set(self.groups):
            raise ValueError(
                f"Lexicon {self.lexicon!r} has groups {list(swapper.groups)} but axis {self.name!r} "
                f"has {list(self.groups)}"
            )
        return swapper

    def to_public_dict(self) -> dict:
        return {
            "name": self.name,
            "groups": list(self.groups),
            "target_group": self.target_group,
            "group_labels": dict(self.group_labels),
            "lexicon": self.lexicon,
        }


BUILTIN_AXES: dict[str, AxisSpec] = {
    "gender": AxisSpec(
        name="gender",
        groups=("male", "female"),
        target_group="female",
        group_labels={"male": "men", "female": "women"},
        lexicon="gender_lexicon.json",
    ),
    "disability": AxisSpec(
        name="disability",
        groups=("non-disabled", "disabled"),
        target_group="disabled",
        group_labels={"non-disabled": "non-disabled people", "disabled": "disabled people"},
    ),
}

# Kept for callers that predate AxisSpec.
AXES: dict[str, str | None] = {name: spec.lexicon for name, spec in BUILTIN_AXES.items()}


def get_axis(axis: AxisSpec | str) -> AxisSpec:
    """Resolve a built-in axis name, or pass an AxisSpec through."""
    if isinstance(axis, AxisSpec):
        return axis
    if axis in BUILTIN_AXES:
        return BUILTIN_AXES[axis]
    raise ValueError(f"Unknown axis {axis!r}; available axes are {sorted(BUILTIN_AXES)}")


def load_swapper(axis: AxisSpec | str) -> GenderSwapper:
    """The lexicon swapper for `axis`. Raises ValueError for a tag-only axis."""
    spec = get_axis(axis)
    swapper = spec.swapper()
    if swapper is None:
        raise ValueError(f"Axis {spec.name!r} has no lexicon; its swaps must come from row tags")
    return swapper


def axis_groups(axis: AxisSpec | str) -> tuple[str, str]:
    """The axis's two groups, in definition order."""
    return get_axis(axis).groups


@lru_cache(maxsize=None)
def _load_lexicon(lexicon: str) -> GenderSwapper:
    path = Path(lexicon)
    if path.suffix == ".json" and path.is_file():
        return GenderSwapper.from_file(path)
    return GenderSwapper.from_resource(lexicon)
