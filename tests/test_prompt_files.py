"""Tests for prompts and few-shot examples delivered as data (biascompare.prompt_files)
and the check that few-shot examples are not evaluated posts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biascompare import ModelSpec, compare_models, load_examples, load_prompts
from biascompare.prompt_files import example_from_record
from biascompare.tasks import DEFAULT_TASK
from test_compare import DATASET, _fake_factory

SAMPLE_PROMPTS = Path(__file__).resolve().parents[1] / "data" / "sample_prompts.yaml"


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_sample_prompts_file_loads_named_prompts_with_example_rows():
    prompts = load_prompts(SAMPLE_PROMPTS)
    assert sorted(prompts) == ["few_shot", "few_shot_pairs", "instruction", "llm"]
    assert prompts["instruction"].examples == ()
    few_shot = prompts["few_shot"].examples
    assert [e.id for e in few_shot] == ["fs01", "fs02", "fs03", "fs04"]
    assert few_shot[0].attributes == {"gender": "female"}
    assert few_shot[0].swaps == {"gender": "My brother just got promoted and he totally deserves it."}
    assert prompts["llm"].options == {"non-toxic": "No", "toxic": "Yes"}
    for prompt in prompts.values():
        prompt.check_task(DEFAULT_TASK)


def test_counterfactual_pairs_template_shows_each_example_with_its_swap():
    rendered = load_prompts(SAMPLE_PROMPTS)["few_shot_pairs"].render("He said hi.", DEFAULT_TASK)
    assert (
        "Post: My deaf neighbor always waves at everyone on our street.\nAnswer: non-toxic\n"
        "Post: My neighbor always waves at everyone on our street.\nAnswer: non-toxic\n\n"
    ) in rendered
    assert rendered.endswith("Post: He said hi.")


def test_example_signature_fields_in_templates(tmp_path):
    examples = _write(tmp_path / "ex.jsonl", [
        {"id": "e1", "text": "My blind friend won.", "label": "non-toxic", "attributes": {"disability": "disabled"},
         "swaps": {"disability": [["blind friend", "friend"]]}, "meta": {"signature": "identity mention, benign"}},
    ])
    prompts_file = tmp_path / "prompts.json"
    prompts_file.write_text(json.dumps({"prompts": {"sig": {
        "instruction": "Q?", "examples": "ex.jsonl",
        "example_template": "[{attributes[disability]}; {meta[signature]}] {text} vs {swap[disability]} -> {answer} ({id})\n",
    }}}), encoding="utf-8")
    rendered = load_prompts(prompts_file)["sig"].render("x", DEFAULT_TASK)
    assert "[disabled; identity mention, benign] My blind friend won. vs My friend won. -> non-toxic (e1)" in rendered


def test_gender_examples_without_tags_get_a_lexicon_counterfactual():
    example = example_from_record({"text": "She said hi to her mother.", "label": "non-toxic",
                                   "attributes": {"gender": "female"}}, "test")
    assert example.swaps == {"gender": "He said hi to his father."}


def test_template_fields_an_example_lacks_are_reported(tmp_path):
    examples = _write(tmp_path / "ex.jsonl", [{"id": "e1", "text": "Nice day.", "label": "non-toxic"}])
    prompts_file = tmp_path / "prompts.yaml"
    prompts_file.write_text('pairs:\n  examples: ex.jsonl\n  example_template: "{text} / {counterfactual}\\n"\n')
    prompt = load_prompts(prompts_file)["pairs"]
    with pytest.raises(ValueError, match="example e1 has no value for 'counterfactual'"):
        prompt.check_task(DEFAULT_TASK)


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"text": "a"}], "needs 'label'"),
        ([{"text": "a", "label": "toxic", "group": "x"}], r"unknown field\(s\) \['group'\]"),
        ([{"text": "a", "label": "toxic", "swaps": {"gender": [["a"]]}}], "phrase, replacement"),
    ],
)
def test_bad_example_rows_name_the_line(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_examples(_write(tmp_path / "ex.jsonl", rows))


def test_missing_files_are_reported(tmp_path):
    with pytest.raises(ValueError, match="Prompts file not found"):
        load_prompts(tmp_path / "nope.yaml")
    (tmp_path / "p.yaml").write_text("fs:\n  examples: missing.jsonl\n")
    with pytest.raises(ValueError, match="examples file not found"):
        load_prompts(tmp_path / "p.yaml")


# ---------------------------------------------------------------------------
# Few-shot examples must not be evaluated posts
# ---------------------------------------------------------------------------


def _prompted_models(prompts):
    return [ModelSpec("biased", id="M0"), ModelSpec("biased", id="M1", prompt=prompts["fs"])]


def _prompts_with(tmp_path, rows):
    _write(tmp_path / "ex.jsonl", rows)
    (tmp_path / "p.yaml").write_text("fs:\n  examples: ex.jsonl\n")
    return load_prompts(tmp_path / "p.yaml")


def _first_post() -> dict:
    return json.loads(DATASET.read_text(encoding="utf-8").splitlines()[0])


@pytest.mark.parametrize("variant", ["same text", "same id", "counterfactual is the post's swap"])
def test_examples_overlapping_evaluated_posts_stop_the_run(tmp_path, variant):
    post = _first_post()  # p01, a male-referencing post about "My brother ... he"
    if variant == "same text":
        row = {"id": "x1", "text": "  " + post["text"].upper(), "label": "non-toxic"}
    elif variant == "same id":
        row = {"id": post["id"], "text": "Something else entirely.", "label": "non-toxic"}
    else:
        swapped = post["text"].replace("My brother", "My sister").replace(" he's", " she's").replace("He was", "She was")
        row = {"id": "x2", "text": swapped, "label": "non-toxic", "attributes": {"gender": "female"}}
    prompts = _prompts_with(tmp_path, [row])
    with pytest.raises(ValueError, match="overlap the evaluated posts.*'p01'"):
        compare_models(_prompted_models(prompts), DATASET, scorer_factory=_fake_factory)


def test_examples_from_another_split_run_fine(tmp_path):
    prompts = _prompts_with(tmp_path, [{"id": "x1", "text": "Totally unrelated post.", "label": "toxic"}])
    results = compare_models(_prompted_models(prompts), DATASET, scorer_factory=_fake_factory)
    assert results["run"]["models"][1]["prompt"]["examples"][0]["id"] == "x1"
