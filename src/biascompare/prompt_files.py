"""Reading the prompts and few-shot examples delivered by the data side.
See docs/DATA.md §3.4.

The testing framework does not write prompts or choose examples. The data
side delivers a prompts file (JSON or YAML) naming one prompt per variant,
and the few-shot examples as data rows in the same format as the evaluated
posts (`id`, `text`, `label`, `attributes`, `swaps`, `meta`), so each example
carries its bias signature. When an example is loaded, its counterfactual
text is built from its tagged swaps (or, for gender rows without tags, the
gender lexicon), so a template can show the example next to its swap.

    prompts:
      instruction:
        instruction: "..."
      few_shot:
        instruction: "..."
        examples: few_shot_examples.jsonl     # relative to the prompts file
        example_template: "Post: {text}\\nAnswer: {answer}\\n\\n"
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

from biascompare.axes import BUILTIN_AXES
from biascompare.fileio import read_mapping
from biascompare.inputs import Post
from biascompare.perturb.tagged import apply_tagged_swap
from biascompare.prompts import DEFAULT_EXAMPLE_TEMPLATE, DEFAULT_TEMPLATE, Example, PromptSpec

_PROMPT_KEYS = frozenset({"name", "instruction", "examples", "template", "example_template", "options", "system"})
# `synonyms` is accepted (and ignored) so evaluated-data rows can be reused as examples.
_EXAMPLE_KEYS = frozenset({"id", "text", "label", "attributes", "swaps", "meta", "synonyms"})


def load_prompts(path: str | Path) -> dict[str, PromptSpec]:
    """Read a prompts file: `{"prompts": {name: prompt}}` (or the mapping of
    named prompts directly). Example files are resolved against its folder."""
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"Prompts file not found: {file_path}")
    raw = read_mapping(file_path)
    if isinstance(raw, Mapping) and "prompts" in raw:
        raw = raw["prompts"]
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError(f"{file_path} must hold a non-empty mapping of prompt names to prompts")
    try:
        return prompts_from_dict(raw, base_dir=file_path.parent)
    except ValueError as exc:
        raise ValueError(f"{file_path}: {exc}") from exc


def prompts_from_dict(raw: Mapping, base_dir: str | Path = ".") -> dict[str, PromptSpec]:
    prompts = {}
    for name, value in raw.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"prompt {name!r} must be an object")
        prompts[name] = prompt_from_dict(value, name=name, base_dir=base_dir)
    return prompts


def prompt_from_dict(raw: Mapping, name: str = "prompt", base_dir: str | Path = ".") -> PromptSpec:
    """Build one PromptSpec. `examples` is a path to a JSONL file of example
    rows, or a list of example rows."""
    unknown = sorted(set(raw) - _PROMPT_KEYS)
    if unknown:
        raise ValueError(f"prompt {name!r}: unknown key(s) {unknown}; allowed keys are {sorted(_PROMPT_KEYS)}")
    examples = raw.get("examples") or ()
    if isinstance(examples, str):
        examples = load_examples(Path(base_dir) / examples)
    elif isinstance(examples, Sequence):
        examples = tuple(example_from_record(record, f"prompt {name!r} examples[{i}]") for i, record in enumerate(examples))
    else:
        raise ValueError(f"prompt {name!r}: examples must be a path to a JSONL file or a list of example rows")
    return PromptSpec(
        name=raw.get("name", name),
        instruction=raw.get("instruction"),
        examples=examples,
        template=raw.get("template", DEFAULT_TEMPLATE),
        example_template=raw.get("example_template", DEFAULT_EXAMPLE_TEMPLATE),
        options=raw.get("options"),
        system=raw.get("system"),
    )


def load_examples(path: str | Path) -> tuple[Example, ...]:
    """Read few-shot examples from a JSONL file of example rows."""
    import json

    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"Few-shot examples file not found: {file_path}")
    examples = []
    for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {file_path}: {exc}") from exc
        examples.append(example_from_record(record, f"line {line_number} of {file_path}"))
    if not examples:
        raise ValueError(f"No examples found in {file_path}")
    return tuple(examples)


def example_from_record(record: object, where: str) -> Example:
    """One example row -> Example, with its counterfactual text per axis."""
    if not isinstance(record, Mapping):
        raise ValueError(f"{where}: an example must be an object, got {record!r}")
    unknown = sorted(set(record) - _EXAMPLE_KEYS)
    if unknown:
        raise ValueError(f"{where}: unknown field(s) {unknown}; allowed fields are {sorted(_EXAMPLE_KEYS)}")
    for required in ("text", "label"):
        if required not in record:
            raise ValueError(f"{where}: an example needs {required!r}")
    try:
        row = Post(
            id=str(record.get("id") or "example"),
            text=record["text"],
            attributes=record.get("attributes") or {},
            swaps=record.get("swaps") or {},
            meta=record.get("meta") or {},
        )
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
    counterfactuals = {axis: apply_tagged_swap(row.text, pairs, axis)[0].text for axis, pairs in row.swaps.items()}
    for axis in row.attributes:
        spec = BUILTIN_AXES.get(axis)
        swapper = spec.swapper() if spec is not None else None
        if axis not in counterfactuals and swapper is not None:
            counterfactuals[axis] = swapper.swap(row.text).text
    return Example(
        text=row.text,
        label=record["label"],
        id=record.get("id"),
        attributes=row.attributes,
        swaps=counterfactuals,
        meta=row.meta,
    )
