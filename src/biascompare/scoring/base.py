"""Scorer protocol and shared probability-conversion helpers. See DESIGN.md §4
and docs/PLAN.md §3.2.

This module has no heavy dependencies (no torch/transformers) so it, and
everything that only imports it, can be imported without either installed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from biascompare.inputs import ModelSpec
from biascompare.metrics import Distribution, ProbPair
from biascompare.tasks import TaskSpec


class Scorer(Protocol):
    """A model that turns a batch of texts into one distribution over the
    task's levels per text (least to most toxic).

    Optionally, a scorer may expose `metadata: dict` (resolved revision,
    label mapping, ...); `compare_models` copies it into `run.models[]`.
    """

    name: str
    scorer_type: str
    uses_instruction: bool

    def score(self, texts: Sequence[str]) -> list[Distribution]: ...

    def close(self) -> None: ...


ScorerFactory = Callable[[ModelSpec, TaskSpec, str], Scorer]
"""(spec, task, device) -> Scorer. `scoring.registry.default_scorer_factory`
dispatches on `spec.type` and is the default."""


def load_error(spec: ModelSpec, exc: BaseException, what: str = "model") -> RuntimeError:
    """A `RuntimeError` for a failed Hub load that says how to fix the usual
    causes (gated model not yet granted, no token, wrong id) before the
    original error. Chain it with `raise load_error(...) from exc`."""
    chain, seen = [], exc
    while seen is not None and len(chain) < 10:
        chain.append(seen)
        seen = seen.__cause__ or seen.__context__
    names = {type(e).__name__ for e in chain}
    text = " ".join(str(e) for e in chain)
    page = f"https://huggingface.co/{spec.name}"
    if "not in the authorized list" in text:
        hint = (
            f"{spec.name!r} is gated and the Hugging Face account that owns your token has not been granted "
            f"access yet. Request it at {page} while logged in as that account, wait for the approval "
            "(Hugging Face settings, Gated repositories), then rerun."
        )
    elif "fine-grained" in text:
        hint = (
            f"{spec.name!r} is gated and your fine-grained token cannot read gated repos. Enable "
            "'Read access to contents of all public gated repos you can access' on the token, or use a read token."
        )
    elif "GatedRepoError" in names or "gated repo" in text:
        hint = (
            f"{spec.name!r} is gated and no valid token was sent. Accept its licence at {page} and set "
            "HF_TOKEN (a Colab secret on Colab) or name an environment variable with `token_env`."
        )
    elif "RepositoryNotFoundError" in names or "not a valid model identifier" in text:
        hint = (
            f"{spec.name!r} is neither a Hub model id nor a local folder. Check the spelling, or for a private "
            "model set HF_TOKEN or `token_env`."
        )
    else:
        hint = "If this is a gated or private model, set HF_TOKEN or name an environment variable with `token_env`."
    return RuntimeError(f"Failed to load {what} {spec.name!r}: {hint}\nOriginal error: {exc}")


@dataclass(frozen=True)
class LevelMapping:
    """How a classification head's outputs become a distribution over task levels.

    - `multi_label=True`: a single sigmoid head (`head_index`) gives
      P(top level); only valid for a 2-level task.
    - `multi_label=False`: softmax over every logit, then the probabilities
      of the model labels in `groups[i]` are summed into level `i`.

    `description` maps each level to the model label names feeding it, for
    the results JSON.
    """

    multi_label: bool
    groups: tuple[tuple[int, ...], ...]
    head_index: int | None
    description: Mapping[str, list[str]]


def resolve_label_index(id2label: Mapping[int | str, str], toxic_label: str) -> int:
    """Find the label index matching `toxic_label`, case-insensitively.

    `id2label` keys may be `int` or `str` — Hugging Face configs are
    inconsistent about this after a JSON round-trip, since JSON object keys
    are always strings. Raises `ValueError` listing the available labels if
    no case-insensitive match is found.
    """
    target = toxic_label.strip().lower()
    for key, label in id2label.items():
        if str(label).strip().lower() == target:
            return int(key)

    available = [str(label) for label in id2label.values()]
    raise ValueError(
        f"toxic_label {toxic_label!r} not found among model labels {available}. "
        "Check the ModelSpec.toxic_label value against the model's config.id2label."
    )


def binary_from_logits(logits: Sequence[float], toxic_index: int, multi_label: bool) -> ProbPair:
    """Convert raw classification logits into a `(P(non-toxic), P(toxic))` pair.

    - `multi_label=True`: `p = sigmoid(logits[toxic_index])`; other heads
      (severe_toxic, insult, ...) are ignored.
    - `multi_label=False`: `p = softmax(logits)[toxic_index]`.

    Implemented with numpy for numerical stability (shifted softmax,
    branch-stable sigmoid).
    """
    logits_arr = np.asarray(logits, dtype=np.float64)
    if logits_arr.ndim != 1:
        raise ValueError(
            f"binary_from_logits expects a 1-D sequence of logits, got shape {logits_arr.shape}"
        )
    if not (0 <= toxic_index < logits_arr.shape[0]):
        raise ValueError(
            f"toxic_index {toxic_index} out of range for {logits_arr.shape[0]} logits"
        )

    if multi_label:
        p_toxic = _sigmoid(float(logits_arr[toxic_index]))
    else:
        p_toxic = float(_softmax(logits_arr)[toxic_index])

    return (1.0 - p_toxic, p_toxic)


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def resolve_level_mapping(
    id2label: Mapping[int | str, str],
    task: TaskSpec,
    multi_label: bool,
    toxic_label: str = "toxic",
    label_map: Mapping[str, Sequence[str]] | None = None,
    model_name: str = "model",
) -> LevelMapping:
    """Decide how a classifier's labels feed the task's levels (docs/DATA.md §3).

    Rules, all matched case-insensitively:

    - **Multi-label (sigmoid) heads**: the task must have 2 levels. The head
      is `label_map[<top level>]` (exactly one label) if given, else
      `toxic_label`.
    - **Softmax heads with `label_map`**: each level lists the model labels
      whose probabilities are summed into it. The lowest level may be left
      out, in which case it receives every label not mapped elsewhere. Every
      model label must end up in exactly one level, and every level must
      receive at least one.
    - **Softmax heads without `label_map`**: on a 2-level task,
      `toxic_label` feeds the top level and every other label the bottom
      one (the POC rule). On a 3- or 4-level task, model labels must match
      the level names exactly.

    Raises ValueError naming the model labels and task levels on any mismatch.
    """
    labels = {int(key): str(name) for key, name in id2label.items()}
    available = [labels[index] for index in sorted(labels)]

    if multi_label:
        return _multi_label_mapping(labels, available, task, toxic_label, label_map, model_name)
    if label_map is None:
        label_map = _default_label_map(available, task, toxic_label, model_name)
    return _softmax_mapping(labels, available, task, label_map, model_name)


def distribution_from_logits(logits: Sequence[float], mapping: LevelMapping) -> Distribution:
    """Turn one row of raw logits into a distribution over task levels."""
    logits_arr = np.asarray(logits, dtype=np.float64)
    if logits_arr.ndim != 1:
        raise ValueError(f"expected a 1-D sequence of logits, got shape {logits_arr.shape}")
    if mapping.multi_label:
        p_top = _sigmoid(float(logits_arr[mapping.head_index]))
        return (1.0 - p_top, p_top)
    probs = _softmax(logits_arr)
    return tuple(float(probs[list(group)].sum()) for group in mapping.groups)


def _index_of(labels: Mapping[int, str], name: str) -> int | None:
    target = name.strip().lower()
    for index, label in labels.items():
        if label.strip().lower() == target:
            return index
    return None


def _multi_label_mapping(
    labels: Mapping[int, str],
    available: list[str],
    task: TaskSpec,
    toxic_label: str,
    label_map: Mapping[str, Sequence[str]] | None,
    model_name: str,
) -> LevelMapping:
    top_level = task.levels[-1]
    if task.n_levels != 2:
        raise ValueError(
            f"{model_name} is a multi-label (sigmoid) classifier, which gives one independent "
            f"probability per head, so it can only be scored on a 2-level task. Task "
            f"{task.name!r} has {task.n_levels} levels {list(task.levels)}."
        )
    head_name = toxic_label
    if label_map is not None:
        entry = {level.lower(): heads for level, heads in label_map.items()}
        unknown = [level for level in label_map if level.lower() != top_level.lower()]
        if unknown or len(entry.get(top_level.lower(), ())) != 1:
            raise ValueError(
                f"{model_name} is a multi-label classifier: label_map must name exactly one head "
                f"for the top level, e.g. {{{top_level!r}: [{toxic_label!r}]}}; got {dict(label_map)}"
            )
        head_name = entry[top_level.lower()][0]
    head_index = _index_of(labels, head_name)
    if head_index is None:
        raise ValueError(
            f"toxic_label {head_name!r} not found among model labels {available} of {model_name}. "
            "Check the ModelSpec.toxic_label (or label_map) value against the model's config.id2label."
        )
    return LevelMapping(
        multi_label=True,
        groups=(),
        head_index=head_index,
        description={task.levels[0]: [f"1 - sigmoid({labels[head_index]})"], top_level: [labels[head_index]]},
    )


def _default_label_map(
    available: list[str], task: TaskSpec, toxic_label: str, model_name: str
) -> Mapping[str, Sequence[str]]:
    if task.n_levels == 2:
        return {task.levels[1]: [toxic_label]}
    by_name = {label.strip().lower(): label for label in available}
    if set(by_name) == {level.lower() for level in task.levels}:
        return {level: [by_name[level.lower()]] for level in task.levels}
    raise ValueError(
        f"{model_name} has labels {available}, which do not match the levels {list(task.levels)} of "
        f"task {task.name!r}. Pass a label_map that assigns every model label to a level."
    )


def _softmax_mapping(
    labels: Mapping[int, str],
    available: list[str],
    task: TaskSpec,
    label_map: Mapping[str, Sequence[str]],
    model_name: str,
) -> LevelMapping:
    groups: list[list[int]] = [[] for _ in task.levels]
    owner: dict[int, str] = {}
    for level, model_labels in label_map.items():
        try:
            level_index = task.level_index(level)
        except ValueError:
            raise ValueError(
                f"label_map for {model_name} names level {level!r}, but task {task.name!r} has "
                f"levels {list(task.levels)}"
            ) from None
        for model_label in model_labels:
            index = _index_of(labels, model_label)
            if index is None:
                raise ValueError(
                    f"toxic_label {model_label!r} not found among model labels {available} of "
                    f"{model_name}. Check the ModelSpec.toxic_label (or label_map) value against "
                    "the model's config.id2label."
                )
            if index in owner and owner[index] != task.levels[level_index]:
                raise ValueError(
                    f"label_map for {model_name} puts model label {labels[index]!r} under both "
                    f"{owner[index]!r} and {task.levels[level_index]!r}"
                )
            owner[index] = task.levels[level_index]
            if index not in groups[level_index]:
                groups[level_index].append(index)

    unmapped = [index for index in sorted(labels) if index not in owner]
    if unmapped and not groups[0]:
        groups[0] = unmapped
        unmapped = []
    if unmapped:
        raise ValueError(
            f"label_map for {model_name} leaves model label(s) {[labels[i] for i in unmapped]} "
            f"unassigned; add them to one of the levels {list(task.levels)}"
        )
    empty = [task.levels[i] for i, group in enumerate(groups) if not group]
    if empty:
        raise ValueError(
            f"label_map for {model_name} gives no model label to level(s) {empty}; every level of "
            f"task {task.name!r} needs at least one (model labels: {available})"
        )
    return LevelMapping(
        multi_label=False,
        groups=tuple(tuple(sorted(group)) for group in groups),
        head_index=None,
        description={task.levels[i]: [labels[j] for j in sorted(group)] for i, group in enumerate(groups)},
    )
