"""Per-model prompts: an instruction, labelled few-shot examples, and the
template that places the post among them. See docs/PLAN.md §5 and
docs/DATA.md §3.4.

Prompts and few-shot examples are **inputs from the data side**: they are
read from a prompts file (`biascompare.prompt_files.load_prompts`), and the
examples are data rows like the evaluated posts, carrying their bias
signatures (the groups they reference and their tagged swaps), which the
example template can show. The framework only renders and checks them.

The study's model variants differ by prompt (M1 and M3 add few-shot
examples to M0 and M2), so a prompt belongs to a model entry, not to the
run. Rendering is a pure function of `(prompt, task, text)`: the original
post, its targeted swap and its controls are rendered with the same
instruction and examples, so the only difference between arms is inside the
`{text}` slot (PLAN.md §5, rule 1).

For a classifier the rendered string is the model's input. For an LLM it is
the user message, followed by the answer options.
"""

from __future__ import annotations

import re
import string
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from biascompare.tasks import TaskSpec

DEFAULT_TEMPLATE = "{instruction}\n\n{examples}Post: {text}"
DEFAULT_EXAMPLE_TEMPLATE = "Post: {text}\nAnswer: {answer}\n\n"
DEFAULT_INSTRUCTION = "Is the following social media post toxic?"

_TEMPLATE_FIELDS = frozenset({"instruction", "examples", "text"})
_EXAMPLE_FIELDS = frozenset({"text", "answer", "label", "id", "attributes", "meta", "swap", "counterfactual"})
_EMPTY: Mapping = MappingProxyType({})


@dataclass(frozen=True)
class Example:
    """One labelled few-shot example: a data row from the data side.

    `label` must be a level of the run's task. The bias signature travels
    with it: `attributes` (axis -> group it references), `swaps` (axis ->
    the example's counterfactual text, built from its tagged swaps when the
    examples file is loaded) and free-form `meta`. An `example_template`
    can show any of them: `{attributes[gender]}`, `{swap[gender]}`,
    `{counterfactual}` (the swap for the example's only, or first, tagged
    axis), `{meta[...]}`, `{id}`, `{label}`.
    """

    text: str
    label: str
    id: str | None = None
    attributes: Mapping[str, str] = _EMPTY
    swaps: Mapping[str, str] = _EMPTY
    meta: Mapping[str, object] = _EMPTY

    def __post_init__(self) -> None:
        for name in ("text", "label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Few-shot example {name} must be a non-empty string, got {value!r}")
        for name in ("attributes", "swaps", "meta"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise ValueError(f"Few-shot example {name} must be an object, got {value!r}")
            object.__setattr__(self, name, MappingProxyType(dict(value)))

    def __hash__(self) -> int:
        return hash((self.text, self.label, self.id))

    def fields(self, answer: str) -> dict:
        """The values an `example_template` may use."""
        swaps = dict(self.swaps)
        return {
            "text": self.text,
            "answer": answer,
            "label": self.label,
            "id": self.id or "",
            "attributes": dict(self.attributes),
            "meta": dict(self.meta),
            "swap": swaps,
            "counterfactual": next(iter(swaps.values()), None),
        }


@dataclass(frozen=True)
class PromptSpec:
    """How a post is wrapped before a model sees it.

    - `instruction`: the question (falls back to the task's instruction,
      then to a default, when a prompted LLM needs one).
    - `examples`: labelled few-shot examples, rendered with
      `example_template` in order, identical for every post and arm.
    - `template`: must contain `{text}` exactly once; may contain
      `{instruction}` and `{examples}`.
    - `options`: per level, the answer string used in example answers and,
      for LLMs, as the answer choices; defaults to the level names.
    - `system`: a fixed system message (LLMs only).
    """

    name: str = "prompt"
    instruction: str | None = None
    examples: tuple[Example, ...] = ()
    template: str = DEFAULT_TEMPLATE
    example_template: str = DEFAULT_EXAMPLE_TEMPLATE
    options: Mapping[str, str] | None = None
    system: str | None = None
    _option_items: tuple = field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        examples = tuple(
            example if isinstance(example, Example) else Example(**example) for example in self.examples
        )
        object.__setattr__(self, "examples", examples)
        _check_template(self.name, "template", self.template, _TEMPLATE_FIELDS, required="text")
        _check_template(self.name, "example_template", self.example_template, _EXAMPLE_FIELDS, required="text")
        if self.options is not None:
            if not isinstance(self.options, Mapping) or not self.options:
                raise ValueError(f"Prompt {self.name!r}: options must be a non-empty mapping of level to answer")
            for level, answer in self.options.items():
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError(f"Prompt {self.name!r}: option for {level!r} must be a non-empty string")
            object.__setattr__(self, "options", MappingProxyType(dict(self.options)))
            object.__setattr__(self, "_option_items", tuple(self.options.items()))

    def __hash__(self) -> int:
        return hash(
            (self.name, self.instruction, self.examples, self.template, self.example_template,
             self._option_items, self.system)
        )

    def check_task(self, task: TaskSpec) -> None:
        """Raise ValueError unless every example label and option key is a
        level of `task` and every example has the fields its template uses."""
        for example in self.examples:
            task.level_index(example.label)
        for level in (self.options or {}):
            task.level_index(level)
        answers = [answer.strip().lower() for answer in self.answers(task)]
        if len(set(answers)) != len(answers):
            raise ValueError(f"Prompt {self.name!r}: answer options must differ, got {answers}")
        self._render_examples(task)

    def answers(self, task: TaskSpec) -> list[str]:
        """The answer string for each level of `task`, in level order."""
        by_level = {level.lower(): answer for level, answer in (self.options or {}).items()}
        return [by_level.get(level.lower(), level) for level in task.levels]

    def instruction_for(self, task: TaskSpec) -> str:
        return self.instruction or task.instruction or DEFAULT_INSTRUCTION

    def render(self, text: str, task: TaskSpec) -> str:
        """The full prompt for one post. Only `text` varies between arms."""
        prompt = self.template.format(
            instruction=self.instruction_for(task), examples=self._render_examples(task), text=text
        )
        return prompt.lstrip()

    def _render_examples(self, task: TaskSpec) -> str:
        answers = self.answers(task)
        rendered = []
        for index, example in enumerate(self.examples):
            values = example.fields(answers[task.level_index(example.label)])
            try:
                if "{counterfactual}" in self.example_template and values["counterfactual"] is None:
                    raise KeyError("counterfactual")
                rendered.append(self.example_template.format(**values))
            except (KeyError, IndexError) as exc:
                name = example.id or f"#{index + 1}"
                raise ValueError(
                    f"Prompt {self.name!r}: few-shot example {name} has no value for {exc} used in its "
                    "example_template (check its attributes, swaps and meta)"
                ) from None
        return "".join(rendered)

    def to_public_dict(self) -> dict:
        public: dict = {"name": self.name, "instruction": self.instruction}
        if self.examples:
            public["examples"] = [
                {key: value for key, value in (
                    ("id", e.id), ("text", e.text), ("label", e.label),
                    ("attributes", dict(e.attributes)), ("swaps", dict(e.swaps)), ("meta", dict(e.meta)),
                ) if value not in (None, {})}
                for e in self.examples
            ]
        if self.template != DEFAULT_TEMPLATE:
            public["template"] = self.template
        if self.example_template != DEFAULT_EXAMPLE_TEMPLATE:
            public["example_template"] = self.example_template
        if self.options is not None:
            public["options"] = dict(self.options)
        if self.system is not None:
            public["system"] = self.system
        return public


def _check_template(prompt_name: str, which: str, template: str, allowed: frozenset[str], required: str) -> None:
    if not isinstance(template, str):
        raise ValueError(f"Prompt {prompt_name!r}: {which} must be a string")
    try:
        fields = [
            re.split(r"[.\[]", name, maxsplit=1)[0]
            for _, name, _, _ in string.Formatter().parse(template)
            if name is not None
        ]
    except ValueError as exc:
        raise ValueError(f"Prompt {prompt_name!r}: {which} is not a valid template: {exc}") from None
    unknown = sorted(set(fields) - allowed)
    if unknown:
        raise ValueError(
            f"Prompt {prompt_name!r}: {which} uses unknown field(s) {unknown}; allowed are {sorted(allowed)}. "
            "Write literal braces as {{ and }}."
        )
    if fields.count(required) != 1:
        raise ValueError(f"Prompt {prompt_name!r}: {which} must contain {{{required}}} exactly once")
