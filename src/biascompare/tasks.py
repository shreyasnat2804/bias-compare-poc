"""Task definitions: the ordered toxicity levels every model is scored over.
See docs/PLAN.md §3.1.

A task fixes the outcome space shared by every model in a run: 2 to 4
toxicity levels, ordered from least to most toxic. Every scorer returns one
probability per level, and every metric is computed from those
distributions, which is what lets different kinds of model be compared on
the same test.

Two scalar summaries of a distribution are used throughout:

- `p_toxic`: total probability of the levels from `toxic_from` upwards
  (the binary collapse, used for display and flip-free readouts).
- `severity`: expected level, scaled to [0, 1] (`sum_i i * p_i / (K - 1)`).
  `direction` is measured on severity because it uses every level. With two
  levels both equal P(toxic), which is exactly the POC's quantity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

MIN_LEVELS = 2
MAX_LEVELS = 4


@dataclass(frozen=True)
class TaskSpec:
    """The label space and (for prompted models) the question a run asks.

    `levels` are ordered from least to most toxic. `toxic_from` is the first
    level counted as toxic in the binary collapse; it defaults to the second
    level, i.e. everything above the lowest level counts as toxic.
    `instruction` is only read by prompted adapters; classifiers ignore it.
    """

    name: str
    levels: tuple[str, ...]
    toxic_from: str | None = None
    instruction: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"TaskSpec.name must be a non-empty string, got {self.name!r}")
        levels = tuple(self.levels)
        object.__setattr__(self, "levels", levels)
        if not (MIN_LEVELS <= len(levels) <= MAX_LEVELS):
            raise ValueError(
                f"TaskSpec {self.name!r} must have between {MIN_LEVELS} and {MAX_LEVELS} levels, "
                f"got {len(levels)}: {list(levels)}"
            )
        for level in levels:
            if not isinstance(level, str) or not level.strip():
                raise ValueError(f"TaskSpec {self.name!r} levels must be non-empty strings, got {level!r}")
        lowered = [level.lower() for level in levels]
        if len(set(lowered)) != len(lowered):
            raise ValueError(f"TaskSpec {self.name!r} levels must be unique (case-insensitive): {list(levels)}")

        toxic_from = levels[1] if self.toxic_from is None else self.toxic_from
        if toxic_from not in levels:
            raise ValueError(
                f"TaskSpec {self.name!r} toxic_from {toxic_from!r} is not one of its levels {list(levels)}"
            )
        if toxic_from == levels[0]:
            raise ValueError(
                f"TaskSpec {self.name!r} toxic_from cannot be the lowest level {levels[0]!r}: "
                "that would count every text as toxic"
            )
        object.__setattr__(self, "toxic_from", toxic_from)

    @property
    def n_levels(self) -> int:
        return len(self.levels)

    @property
    def toxic_index(self) -> int:
        return self.levels.index(self.toxic_from)

    def level_index(self, level: str) -> int:
        """Index of `level`, matched case-insensitively. Raises ValueError if absent."""
        target = level.strip().lower()
        for index, name in enumerate(self.levels):
            if name.lower() == target:
                return index
        raise ValueError(f"{level!r} is not a level of task {self.name!r}; levels are {list(self.levels)}")

    def p_toxic(self, dist: Sequence[float]) -> float:
        """Total probability of the levels from `toxic_from` upwards."""
        self._check_length(dist)
        return float(sum(dist[self.toxic_index :]))

    def severity(self, dist: Sequence[float]) -> float:
        """Expected level scaled to [0, 1]. Equals P(toxic) for a 2-level task."""
        self._check_length(dist)
        return float(sum(index * p for index, p in enumerate(dist)) / (self.n_levels - 1))

    def predicted_level(self, dist: Sequence[float]) -> int:
        """Index of the most probable level; a tie goes to the more toxic level.

        For a 2-level task this is the POC's rule, P(toxic) >= 0.5.
        """
        self._check_length(dist)
        best = 0
        for index in range(1, len(dist)):
            if dist[index] >= dist[best]:
                best = index
        return best

    def to_public_dict(self) -> dict:
        return {
            "name": self.name,
            "levels": list(self.levels),
            "toxic_from": self.toxic_from,
            "instruction": self.instruction,
        }

    def _check_length(self, dist: Sequence[float]) -> None:
        if len(dist) != self.n_levels:
            raise ValueError(
                f"Distribution has {len(dist)} values but task {self.name!r} has {self.n_levels} levels"
            )


BUILTIN_TASKS: dict[str, TaskSpec] = {
    "toxicity": TaskSpec(name="toxicity", levels=("non-toxic", "toxic")),
    "toxicity-3": TaskSpec(
        name="toxicity-3", levels=("non-toxic", "borderline", "toxic"), toxic_from="toxic"
    ),
    "toxicity-4": TaskSpec(
        name="toxicity-4", levels=("non-toxic", "mild", "toxic", "severe"), toxic_from="toxic"
    ),
}

DEFAULT_TASK = BUILTIN_TASKS["toxicity"]


def get_task(task: TaskSpec | str | None) -> TaskSpec:
    """Resolve a built-in task name (or `None`, the binary default) to a TaskSpec."""
    if task is None:
        return DEFAULT_TASK
    if isinstance(task, TaskSpec):
        return task
    if isinstance(task, str) and task in BUILTIN_TASKS:
        return BUILTIN_TASKS[task]
    raise ValueError(f"Unknown task {task!r}; built-in tasks are {sorted(BUILTIN_TASKS)}")
