"""Input data types and loading/validation. See docs/DATA.md.

`ModelSpec` describes a model to score with, and `Post`/`load_posts`
describe and load the dataset rows to perturb and score. Both dataclasses
are frozen and copy any mapping they are given into a read-only view, so
nothing here is mutated after construction.

Two row formats are accepted, decided per line (docs/DATA.md §2):

- **v1**: `{"id", "text", "attributes": {axis: group}, "label"?, "meta"?, "lang"?}`.
- **v0.1** (the POC format): `{"id", "platform", "text", "gender",
  "intended_toxicity"?}`. `gender` becomes `attributes.gender`; `platform`,
  `intended_toxicity` and any other extra fields move into `meta`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from biascompare.axes import AxisSpec, get_axis
from biascompare.prompts import PromptSpec

_LEGACY_REQUIRED_FIELDS = ("id", "platform", "text", "gender")
_LEGACY_GENDERS = frozenset({"male", "female"})
_V1_REQUIRED_FIELDS = ("id", "text", "attributes")
_V1_FIELDS = frozenset({"id", "text", "attributes", "label", "meta", "lang", "swaps", "synonyms"})
_SUPPORTED_LANGS = frozenset({"en"})
_SETTING_TYPES = (str, int, float, bool, type(None))

_EMPTY: Mapping = MappingProxyType({})
_WORD = re.compile(r"[A-Za-z]+")
_SINGLE_WORD = re.compile(r"^[A-Za-z]+$")


@dataclass(frozen=True)
class ModelSpec:
    """A model to score with. See docs/DATA.md §3.

    - `type` selects the adapter (`hf-classifier` is the only one today).
    - `toxic_label` / `label_map` say how the model's own labels map onto
      the task's levels (see `scoring.base.resolve_level_mapping`).
    - `revision` pins a Hugging Face Hub commit.
    - `token` is an optional per-model access token; `token_env` names an
      environment variable holding one instead. The token is excluded from
      `repr()`, equality and `to_public_dict()`, so it never reaches logs or
      results.
    - `settings` holds adapter-specific scalars (`batch_size`, `max_length`).
    - `id` is the model's name in the results (`M0`, `M1`, ...); it defaults
      to `name` and must be unique in a run, so one Hugging Face model can
      appear twice with different prompts.
    - `prompt` wraps each post before the model sees it (instruction and
      few-shot examples, see `biascompare.prompts`); `None` scores the raw post.
    """

    name: str
    toxic_label: str = "toxic"
    token: str | None = field(default=None, repr=False, compare=False)
    type: str = "hf-classifier"
    label_map: Mapping[str, tuple[str, ...]] | None = None
    revision: str | None = None
    token_env: str | None = None
    settings: Mapping[str, object] = _EMPTY
    id: str | None = None
    prompt: PromptSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"ModelSpec.name must be a non-empty string, got {self.name!r}")
        if not isinstance(self.toxic_label, str) or not self.toxic_label.strip():
            raise ValueError(
                f"ModelSpec.toxic_label must be a non-empty string, got {self.toxic_label!r}"
            )
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError(f"ModelSpec.type must be a non-empty string, got {self.type!r}")
        if self.id is not None and (not isinstance(self.id, str) or not self.id.strip()):
            raise ValueError(f"ModelSpec.id must be a non-empty string when given, got {self.id!r}")
        if self.prompt is not None and not isinstance(self.prompt, PromptSpec):
            raise ValueError(f"ModelSpec.prompt must be a PromptSpec, got {type(self.prompt).__name__}")
        if self.label_map is not None:
            object.__setattr__(self, "label_map", _freeze_label_map(self.name, self.label_map))
        object.__setattr__(self, "settings", _freeze_settings(self.name, self.settings))

    def __hash__(self) -> int:
        label_map = None if self.label_map is None else tuple(self.label_map.items())
        return hash(
            (self.name, self.toxic_label, self.type, label_map, self.revision, self.token_env,
             tuple(sorted(self.settings.items())), self.id, self.prompt)
        )

    @property
    def display_id(self) -> str:
        """The model's identifier in results: `id` if set, else `name`."""
        return self.id or self.name

    @property
    def label(self) -> str:
        """Short display name: the last path component of the HF repo id."""
        return self.name.rstrip("/").split("/")[-1]

    def resolved_token(self) -> str | None:
        """The explicit token, else the `token_env` variable, else `HF_TOKEN`, else None.

        `None` tells `from_pretrained` to fall back to a cached
        `huggingface-cli login` session, if any.
        """
        if self.token is not None:
            return self.token
        if self.token_env:
            return os.environ.get(self.token_env) or None
        return os.environ.get("HF_TOKEN") or None

    def to_public_dict(self) -> dict:
        """Serialisable view for results/logs. Never includes the token."""
        public: dict = {"name": self.name, "toxic_label": self.toxic_label}
        if self.id is not None:
            public["id"] = self.id
        if self.prompt is not None:
            public["prompt"] = self.prompt.to_public_dict()
        if self.type != "hf-classifier":
            public["type"] = self.type
        if self.label_map is not None:
            public["label_map"] = {level: list(labels) for level, labels in self.label_map.items()}
        if self.revision is not None:
            public["revision"] = self.revision
        if self.settings:
            public["settings"] = dict(self.settings)
        return public


def _freeze_label_map(model_name: str, label_map: Mapping) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(label_map, Mapping) or not label_map:
        raise ValueError(f"ModelSpec {model_name!r}: label_map must be a non-empty mapping, got {label_map!r}")
    frozen: dict[str, tuple[str, ...]] = {}
    for level, labels in label_map.items():
        if isinstance(labels, str):
            labels = (labels,)
        labels = tuple(labels)
        if not isinstance(level, str) or not level.strip() or not labels:
            raise ValueError(
                f"ModelSpec {model_name!r}: label_map entries must map a level name to one or more "
                f"model labels, got {level!r}: {labels!r}"
            )
        if not all(isinstance(label, str) and label.strip() for label in labels):
            raise ValueError(f"ModelSpec {model_name!r}: label_map[{level!r}] must list non-empty strings")
        frozen[level] = labels
    return MappingProxyType(frozen)


def _freeze_settings(model_name: str, settings: Mapping) -> Mapping[str, object]:
    if not isinstance(settings, Mapping):
        raise ValueError(f"ModelSpec {model_name!r}: settings must be a mapping, got {settings!r}")
    for key, value in settings.items():
        if not isinstance(key, str) or not isinstance(value, _SETTING_TYPES):
            raise ValueError(
                f"ModelSpec {model_name!r}: settings must map names to scalar values, got {key!r}: {value!r}"
            )
    return MappingProxyType(dict(settings))


@dataclass(frozen=True)
class Post:
    """One dataset row: the text to perturb and score, plus its metadata.

    `attributes` maps each axis to the group the text originally refers to
    (`{"gender": "female"}`); values are checked against the axis lexicon
    by `load_posts`. `label` is an optional gold level for the competence
    check. `meta` is free-form and copied into the results.
    """

    id: str
    text: str
    attributes: Mapping[str, str]
    label: str | None = None
    meta: Mapping[str, object] = _EMPTY
    lang: str = "en"
    swaps: Mapping[str, tuple[tuple[str, str], ...]] = _EMPTY
    synonyms: Mapping[str, tuple[str, ...]] = _EMPTY

    def __post_init__(self) -> None:
        _require_nonempty_str("id", self.id)
        _require_nonempty_str("text", self.text)
        if not isinstance(self.attributes, Mapping):
            raise ValueError(f"Post.attributes must be an object, got {self.attributes!r} (id={self.id!r})")
        for axis, group in self.attributes.items():
            if not (isinstance(axis, str) and axis.strip() and isinstance(group, str) and group.strip()):
                raise ValueError(
                    f"Post.attributes must map axis names to non-empty group names, "
                    f"got {axis!r}: {group!r} (id={self.id!r})"
                )
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        if self.label is not None:
            _require_nonempty_str("label", self.label)
        if not isinstance(self.meta, Mapping):
            raise ValueError(f"Post.meta must be an object, got {self.meta!r} (id={self.id!r})")
        object.__setattr__(self, "meta", MappingProxyType(dict(self.meta)))
        if self.lang not in _SUPPORTED_LANGS:
            raise ValueError(
                f"Post.lang {self.lang!r} is not supported (id={self.id!r}); only English lexicons "
                f"exist, so lang must be one of {sorted(_SUPPORTED_LANGS)}"
            )
        object.__setattr__(self, "swaps", _freeze_swaps(self.id, self.swaps))
        object.__setattr__(self, "synonyms", _freeze_synonyms(self.id, self.synonyms))

    def swap_words(self) -> frozenset[str]:
        """Every word in this row's tagged swaps (both sides, every axis).
        Controls never change or introduce these."""
        return frozenset(
            word.lower()
            for pairs in self.swaps.values()
            for pair in pairs
            for phrase in pair
            for word in _WORD.findall(phrase)
        )

    def __hash__(self) -> int:
        # `meta` may hold unhashable JSON values; id and text are enough to
        # stay consistent with equality.
        return hash((self.id, self.text))

    @classmethod
    def from_legacy(
        cls,
        id: str,
        platform: str,
        text: str,
        gender: str,
        intended_toxicity: str | None = None,
        **extra: object,
    ) -> "Post":
        """Build a row from the v0.1 POC fields (DESIGN §3)."""
        _require_nonempty_str("id", id)
        _require_nonempty_str("platform", platform)
        _require_nonempty_str("text", text)
        _require_nonempty_str("gender", gender)
        if gender not in _LEGACY_GENDERS:
            raise ValueError(
                f"Post.gender must be one of {sorted(_LEGACY_GENDERS)}, got {gender!r} (id={id!r})"
            )
        meta = {"platform": platform, "intended_toxicity": intended_toxicity, **extra}
        return cls(id=id, text=text, attributes={"gender": gender}, meta=meta)

    @property
    def gender(self) -> str | None:
        return self.attributes.get("gender")

    @property
    def platform(self) -> str | None:
        return self.meta.get("platform")

    @property
    def intended_toxicity(self) -> str | None:
        return self.meta.get("intended_toxicity")

    def group(self, axis: str) -> str:
        """The group this row originally refers to on `axis`."""
        try:
            return self.attributes[axis]
        except KeyError:
            raise ValueError(f"Post {self.id!r} has no attribute for axis {axis!r}") from None


def _freeze_swaps(post_id: str, swaps: object) -> Mapping[str, tuple[tuple[str, str], ...]]:
    if not isinstance(swaps, Mapping):
        raise ValueError(f"Post.swaps must be an object of axis -> [[phrase, replacement], ...] (id={post_id!r})")
    frozen: dict[str, tuple[tuple[str, str], ...]] = {}
    for axis, pairs in swaps.items():
        if not isinstance(pairs, Sequence) or isinstance(pairs, str):
            raise ValueError(f"Post.swaps[{axis!r}] must be a list of [phrase, replacement] pairs (id={post_id!r})")
        checked = []
        for pair in pairs:
            if (
                not isinstance(pair, Sequence) or isinstance(pair, str) or len(pair) != 2
                or not all(isinstance(side, str) and _WORD.search(side) for side in pair)
            ):
                raise ValueError(
                    f"Post.swaps[{axis!r}] entries must be [phrase, replacement] with words on both "
                    f"sides, got {pair!r} (id={post_id!r})"
                )
            if pair[0].strip().lower() == pair[1].strip().lower():
                raise ValueError(f"Post.swaps[{axis!r}] pair {list(pair)!r} changes nothing (id={post_id!r})")
            checked.append((pair[0], pair[1]))
        frozen[axis] = tuple(checked)
    return MappingProxyType(frozen)


def _freeze_synonyms(post_id: str, synonyms: object) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(synonyms, Mapping):
        raise ValueError(f"Post.synonyms must be an object of word -> [synonyms] (id={post_id!r})")
    frozen: dict[str, tuple[str, ...]] = {}
    for word, options in synonyms.items():
        if not isinstance(word, str) or not _SINGLE_WORD.match(word):
            raise ValueError(f"Post.synonyms keys must be single words, got {word!r} (id={post_id!r})")
        if isinstance(options, str):
            options = [options]
        if not isinstance(options, Sequence) or not options or not all(
            isinstance(option, str) and _WORD.search(option) for option in options
        ):
            raise ValueError(f"Post.synonyms[{word!r}] must be a non-empty list of words (id={post_id!r})")
        frozen[word] = tuple(options)
    return MappingProxyType(frozen)


def _require_nonempty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Post.{field_name} must be a non-empty string, got {value!r}")


def load_posts(
    path: str | Path,
    axis: AxisSpec | str = "gender",
    axes: Sequence[AxisSpec | str] | None = None,
    require_axis: bool = True,
) -> tuple[Post, ...]:
    """Load and validate dataset rows from a JSONL file (docs/DATA.md §2).

    One JSON object per non-blank line, in either the v1 or the v0.1 format
    (decided per line by the presence of `attributes`). Validates: valid
    JSON objects (errors name the line number), required fields present,
    no unknown top-level v1 fields, tagged swaps and synonyms well formed,
    unique ids, and at least one row.

    With `axis` (the default), every row must have that axis in its
    `attributes`. With `axes`, a row needs at least one of them. With
    `require_axis=False` (what a run uses), rows need none of them: a run
    then uses, per axis, the rows that have it. Any attribute that is
    present must name one of its axis's groups.

    Raises ValueError with an actionable message for any violation.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"Posts file not found: {file_path}")

    specs = [get_axis(name) for name in (axes if axes is not None else [axis])]
    posts: list[Post] = []
    seen_ids: set[str] = set()

    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = _parse_json_line(line, line_number, file_path)
            _require_dict_record(record, line_number, file_path)
            try:
                post = _post_from_record(
                    record, specs, require="none" if not require_axis else ("each" if axes is None else "any")
                )
            except ValueError as exc:
                raise ValueError(f"Line {line_number} of {file_path}: {exc}") from exc

            if post.id in seen_ids:
                raise ValueError(f"Duplicate post id {post.id!r} on line {line_number} of {file_path}")
            seen_ids.add(post.id)
            posts.append(post)

    if not posts:
        raise ValueError(f"No posts found in {file_path}: file is empty or contains only blank lines")

    return tuple(posts)


def _post_from_record(record: dict, specs: list[AxisSpec], require: str) -> Post:
    if "attributes" in record:
        post = _post_from_v1(record)
    else:
        post = _post_from_legacy(record)

    present = 0
    for spec in specs:
        if spec.name not in post.attributes:
            if require == "each":
                raise ValueError(f"row {post.id!r} has no {spec.name!r} entry in attributes")
            continue
        present += 1
        if post.attributes[spec.name] not in spec.groups:
            raise ValueError(
                f"row {post.id!r} attributes.{spec.name} must be one of {list(spec.groups)}, "
                f"got {post.attributes[spec.name]!r}"
            )
    if present == 0 and require == "any":
        raise ValueError(
            f"row {post.id!r} has no attributes for any of the axes {[spec.name for spec in specs]}"
        )
    return post


def _post_from_v1(record: dict) -> Post:
    missing = [name for name in _V1_REQUIRED_FIELDS if name not in record]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")
    unknown = sorted(set(record) - _V1_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown field(s) {unknown}; allowed fields are {sorted(_V1_FIELDS)}. "
            "Put any extra information under 'meta'"
        )
    return Post(
        id=record["id"],
        text=record["text"],
        attributes=record["attributes"],
        label=record.get("label"),
        meta=record.get("meta") or {},
        lang=record.get("lang", "en"),
        swaps=record.get("swaps") or {},
        synonyms=record.get("synonyms") or {},
    )


def _post_from_legacy(record: dict) -> Post:
    missing = [name for name in _LEGACY_REQUIRED_FIELDS if name not in record]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")
    return Post.from_legacy(**record)


def _parse_json_line(line: str, line_number: int, file_path: Path) -> object:
    try:
        return json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON on line {line_number} of {file_path}: {exc}") from exc


def _require_dict_record(record: object, line_number: int, file_path: Path) -> None:
    if not isinstance(record, dict):
        raise ValueError(
            f"Line {line_number} of {file_path} must be a JSON object, got {type(record).__name__}"
        )
