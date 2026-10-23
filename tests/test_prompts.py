"""Tests for biascompare.prompts and per-model prompts in a run."""

from __future__ import annotations

import pytest

from biascompare import Example, ModelSpec, PromptSpec, compare_models
from biascompare.prompts import prompt_from_dict
from biascompare.tasks import BUILTIN_TASKS, DEFAULT_TASK
from test_compare import DATASET, FakeScorer, _biased_toxicity, _unbiased_toxicity

FEW_SHOT = PromptSpec(
    name="few_shot",
    instruction="Is the following post toxic? Answer toxic or non-toxic.",
    examples=(Example("You are an idiot.", "toxic"), Example("Have a nice day!", "non-toxic")),
)


def test_render_places_instruction_examples_and_post():
    rendered = FEW_SHOT.render("He said hi.", DEFAULT_TASK)
    assert rendered == (
        "Is the following post toxic? Answer toxic or non-toxic.\n\n"
        "Post: You are an idiot.\nAnswer: toxic\n\n"
        "Post: Have a nice day!\nAnswer: non-toxic\n\n"
        "Post: He said hi."
    )


def test_only_the_text_slot_differs_between_arms():
    original = FEW_SHOT.render("He said hi.", DEFAULT_TASK)
    swapped = FEW_SHOT.render("She said hi.", DEFAULT_TASK)
    prefix = original[: original.index("Post: He")]
    assert swapped.startswith(prefix) and swapped[len(prefix):] == "Post: She said hi."


def test_post_text_with_braces_is_inserted_verbatim():
    assert PromptSpec(instruction="Q?").render("a {weird} post", DEFAULT_TASK).endswith("Post: a {weird} post")


def test_instruction_falls_back_to_task_then_default():
    assert PromptSpec().render("x", DEFAULT_TASK).startswith("Is the following social media post toxic?")
    task = BUILTIN_TASKS["toxicity-3"].__class__(name="t", levels=("a", "b"), instruction="Custom?")
    assert PromptSpec().render("x", task).startswith("Custom?")


def test_options_change_example_answers():
    prompt = PromptSpec(examples=(Example("bad", "toxic"),), options={"toxic": "Yes", "non-toxic": "No"})
    assert "Answer: Yes" in prompt.render("x", DEFAULT_TASK)
    assert prompt.answers(DEFAULT_TASK) == ["No", "Yes"]


def test_template_validation():
    with pytest.raises(ValueError, match="exactly once"):
        PromptSpec(template="{instruction} no text slot")
    with pytest.raises(ValueError, match="unknown field"):
        PromptSpec(template="{instruction} {post}")
    with pytest.raises(ValueError, match="exactly once"):
        PromptSpec(template="{text} and {text}")


def test_check_task_rejects_labels_that_are_not_levels():
    prompt = PromptSpec(examples=(Example("meh", "borderline"),))
    with pytest.raises(ValueError, match="not a level"):
        prompt.check_task(DEFAULT_TASK)
    prompt.check_task(BUILTIN_TASKS["toxicity-3"])
    with pytest.raises(ValueError, match="must differ"):
        PromptSpec(options={"toxic": "x", "non-toxic": "X"}).check_task(DEFAULT_TASK)


def test_prompt_from_dict_validates_shape():
    prompt = prompt_from_dict({"instruction": "Q?", "examples": [{"text": "a", "label": "toxic"}]}, name="p")
    assert prompt.examples == (Example("a", "toxic"),)
    with pytest.raises(ValueError, match="unknown key"):
        prompt_from_dict({"instructions": "Q?"})
    with pytest.raises(ValueError, match="exactly 'text' and 'label'"):
        prompt_from_dict({"examples": [{"text": "a"}]})


def test_prompts_are_hashable_and_comparable():
    assert hash(FEW_SHOT) == hash(PromptSpec(
        name="few_shot",
        instruction="Is the following post toxic? Answer toxic or non-toxic.",
        examples=(Example("You are an idiot.", "toxic"), Example("Have a nice day!", "non-toxic")),
    ))


# ---------------------------------------------------------------------------
# In a run: the same model twice, with and without a prompt
# ---------------------------------------------------------------------------


class _PromptAwareScorer(FakeScorer):
    """Scores the rendered prompt, so a prompt changes the output."""

    def __init__(self, spec, task):
        fn = _biased_toxicity if spec.name == "biased" else _unbiased_toxicity
        super().__init__(spec.display_id, fn)
        self._spec, self._task = spec, task
        self.seen: list[str] = []
        self.uses_instruction = spec.prompt is not None

    def score(self, texts):
        rendered = [t if self._spec.prompt is None else self._spec.prompt.render(t, self._task) for t in texts]
        self.seen.extend(rendered)
        return super().score(rendered)


def test_same_model_with_and_without_prompt_are_separate_variants():
    scorers = {}

    def factory(spec, task, device):
        scorers[spec.display_id] = _PromptAwareScorer(spec, task)
        return scorers[spec.display_id]

    models = [ModelSpec("unbiased", id="M0"), ModelSpec("unbiased", id="M1", prompt=FEW_SHOT)]
    results = compare_models(models, DATASET, seed=0, scorer_factory=factory)

    run_models = {entry["id"]: entry for entry in results["run"]["models"]}
    assert run_models["M1"]["prompt"]["name"] == "few_shot"
    assert run_models["M1"]["prompt_example"].startswith("Is the following post toxic?")
    assert run_models["M1"]["instruction_used"] is True and run_models["M0"]["instruction_used"] is False
    assert "prompt" not in run_models["M0"]
    assert [c["model"] for c in results["comparisons"]] == ["M1"]
    assert all(text.startswith("Is the following post toxic?") for text in scorers["M1"].seen)
    # the prompt changes the inputs, so the two variants' scores differ
    p0 = [m["p_toxic_original"] for m in results["models"][0]["per_post"]]
    p1 = [m["p_toxic_original"] for m in results["models"][1]["per_post"]]
    assert p0 != p1


def test_duplicate_model_ids_are_rejected():
    with pytest.raises(ValueError, match="ids must be unique"):
        compare_models([ModelSpec("m"), ModelSpec("m")], DATASET, scorer_factory=lambda *a: None)


def test_prompt_with_a_label_outside_the_task_fails_before_scoring():
    bad = PromptSpec(examples=(Example("x", "severe"),))
    with pytest.raises(ValueError, match="'M1'.*not a level"):
        compare_models(
            [ModelSpec("a", id="M0"), ModelSpec("a", id="M1", prompt=bad)], DATASET, scorer_factory=lambda *a: None
        )
