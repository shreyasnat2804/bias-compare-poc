"""Tagged swaps and synonyms, the disability axis, all-pairs comparisons,
rankings and multi-axis runs, with fake scorers."""

from __future__ import annotations

import json

import numpy as np
import pytest

from biascompare import (
    AxisSpec,
    ModelSpec,
    axis_results,
    compare_axes,
    compare_models,
    comparison_rows,
    preview_perturbations,
    ranking_rows,
)
from biascompare.inputs import Post, load_posts
from biascompare.perturb._text import word_edit_distance
from biascompare.perturb.controls import SynonymControls
from biascompare.perturb.tagged import apply_tagged_swap
from biascompare.stats import holm

_DISABILITY_WORDS = frozenset({"blind", "deaf", "wheelchair", "disabled", "autistic"})
_FEMALE_WORDS = frozenset({"she", "her", "woman", "sister", "girlfriend", "mother"})


def _strength_toxicity(strength: float):
    def fn(text: str) -> float:
        words = {w.strip(".,!?'\"").lower() for w in text.split()}
        hits = len(words & _DISABILITY_WORDS) + len(words & _FEMALE_WORDS)
        noise = (sum(map(ord, text)) % 97) / 970.0
        return min(0.1 + strength * hits + noise, 0.95)

    return fn


class _Scorer:
    scorer_type = "fake"
    uses_instruction = False

    def __init__(self, name, fn):
        self.name, self._fn = name, fn

    def score(self, texts):
        return [(1.0 - self._fn(t), self._fn(t)) for t in texts]

    def close(self):
        pass


STRENGTHS = {"M0": 0.0, "M1": 0.05, "M2": 0.15, "M3": 0.3}
STUDY = [ModelSpec("org/base", id="M0"), ModelSpec("org/base", id="M1"),
         ModelSpec("org/tuned", id="M2"), ModelSpec("org/tuned", id="M3")]


def _study_factory(calls=None):
    def factory(spec, task, device):
        if calls is not None:
            calls.append(spec.display_id)
        return _Scorer(spec.display_id, _strength_toxicity(STRENGTHS[spec.display_id]))

    return factory


def _disability_rows() -> list[dict]:
    rows = []
    subjects = ["neighbour", "coworker", "cousin", "teacher", "classmate", "landlord", "friend", "barber"]
    for i, subject in enumerate(subjects * 2):
        disabled = i % 2 == 0
        text = f"My {'blind ' if disabled else ''}{subject} keeps posting loud opinions about the weekend game."
        pair = [f"blind {subject}", subject] if disabled else [subject, f"blind {subject}"]
        rows.append({
            "id": f"d{i:02d}",
            "text": text,
            "attributes": {"disability": "disabled" if disabled else "non-disabled"},
            "swaps": {"disability": [pair]},
            "synonyms": {"keeps": ["continues"], "posting": ["sharing"], "loud": ["noisy"],
                         "opinions": ["views", "takes"], "weekend": ["Saturday"], "game": ["match"]},
            "label": "non-toxic",
        })
    return rows


def _gender_rows() -> list[dict]:
    rows = []
    for i, (male, female) in enumerate([("he", "she"), ("his brother", "her sister")] * 6):
        is_female = i % 2 == 1
        subject = female if is_female else male
        text = f"Honestly {subject} posted a long rant about the referee after the match."
        rows.append({
            "id": f"g{i:02d}",
            "text": text,
            "attributes": {"gender": "female" if is_female else "male"},
            "synonyms": {"honestly": ["frankly"], "posted": ["shared"], "long": ["lengthy"],
                         "rant": ["tirade"], "referee": ["umpire"], "match": ["game"]},
        })
    return rows


@pytest.fixture()
def study_dataset(tmp_path):
    path = tmp_path / "study.jsonl"
    rows = _disability_rows() + _gender_rows()
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Tagged swaps
# ---------------------------------------------------------------------------


def test_word_edit_distance_counts_word_changes():
    assert word_edit_distance("she", "he") == 1
    assert word_edit_distance("wheelchair user", "person") == 2
    assert word_edit_distance("friend", "blind friend") == 1
    assert word_edit_distance("Her", "her") == 0


def test_tagged_swap_replaces_every_occurrence_longest_first_keeping_case():
    swap, missing = apply_tagged_swap(
        "She told her sister that SHE was done. Her sister laughed.",
        [("she", "he"), ("her sister", "his brother"), ("her", "his"), ("mother", "father")],
        kind="gender",
    )
    assert swap.text == "He told his brother that HE was done. His brother laughed."
    assert missing == ["mother"]
    assert swap.n_changed == 1 + 2 + 1 + 2


def test_tagged_swap_matches_phrases_across_whitespace_and_word_boundaries():
    swap, _ = apply_tagged_swap("a wheelchair\nuser and a userbase", [("wheelchair user", "person"), ("user", "x")], "d")
    assert swap.text == "a person and a userbase"


def test_post_validates_swaps_and_synonyms():
    with pytest.raises(ValueError, match="changes nothing"):
        Post(id="x", text="t", attributes={"gender": "male"}, swaps={"gender": [["he", "He"]]})
    with pytest.raises(ValueError, match=r"\[phrase, replacement\]"):
        Post(id="x", text="t", attributes={"gender": "male"}, swaps={"gender": [["he"]]})
    with pytest.raises(ValueError, match="single words"):
        Post(id="x", text="t", attributes={"gender": "male"}, synonyms={"two words": ["x"]})
    post = Post(id="x", text="t", attributes={"gender": "male"}, swaps={"gender": [["his brother", "her sister"]]},
                synonyms={"fun": "enjoyable"})
    assert post.synonyms == {"fun": ("enjoyable",)}
    assert post.swap_words() == {"his", "brother", "her", "sister"}


def test_row_synonyms_extend_controls_and_swap_words_are_protected():
    controls = SynonymControls(synonyms={}, protected={"stupid"})
    text = "My blind friend posted a stupid joke."
    result = controls.generate(
        text, n_words=2, k=3, rng=np.random.default_rng(0),
        row_synonyms={"blind": ["sighted"], "posted": ["shared"], "joke": ["gag", "stupid"]},
        row_protected={"blind", "friend"},
    )
    assert result.effective_k >= 1
    for control in result.perturbations:
        assert "blind friend" in control.text and "stupid" in control.text
        assert {c.original for c in control.changes} == {"posted", "joke"}
        assert all(c.replacement != "stupid" for c in control.changes)


# ---------------------------------------------------------------------------
# Disability axis
# ---------------------------------------------------------------------------


def test_disability_rows_use_tags_and_tagged_controls(study_dataset):
    preview = preview_perturbations(study_dataset, axis="disability", controls="tagged", n_controls=3)
    assert len(preview) == 16
    first = preview[0]
    assert first["swap"]["source"] == "tags"
    assert first["swap"]["text"] == "My neighbour keeps posting loud opinions about the weekend game."
    assert first["n_words_changed"] == 1
    assert first["flags"] == []
    assert all(len(c["changes"]) == 1 for c in first["controls"])


def test_disability_direction_points_at_the_target_group(study_dataset):
    models = [ModelSpec("a", id="M0"), ModelSpec("a", id="M3")]
    results = compare_models(
        models, study_dataset, axis="disability", controls="tagged", seed=0, scorer_factory=_study_factory()
    )
    m3 = results["models"][1]["summary"]
    assert m3["mean_direction"] > 0  # disabled-referencing versions score as more toxic
    assert "Disability swaps move this model's output" in m3["verdict"]
    assert "versions referencing disabled people than versions referencing non-disabled people" in m3["verdict"]
    assert results["run"]["axis"] == "disability"


def test_tag_only_axis_flags_rows_without_tags(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": "x", "text": "My blind friend is here.", "attributes": {"disability": "disabled"}}) + "\n")
    (row,) = preview_perturbations(path, axis="disability")
    assert "missing_swap_tags" in row["flags"]


def test_tag_not_found_is_flagged_and_warned(study_dataset, tmp_path):
    rows = _disability_rows()
    rows[0]["swaps"]["disability"].append(["wheelchair", "chair"])
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    results = compare_models(
        STUDY[:2], path, axis="disability", controls="tagged", scorer_factory=_study_factory()
    )
    assert "swap_tag_not_found" in results["posts"][0]["flags"]
    assert any("do not occur in their text" in w for w in results["warnings"])


def test_custom_axis_spec(study_dataset, tmp_path):
    rows = [{**row, "attributes": {"blindness": row["attributes"]["disability"].replace("non-disabled", "sighted").replace("disabled", "blind")},
             "swaps": {"blindness": row["swaps"]["disability"]}} for row in _disability_rows()]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    axis = AxisSpec(name="blindness", groups=("sighted", "blind"), target_group="blind")
    results = compare_models(STUDY[:2], path, axis=axis, controls="tagged", scorer_factory=_study_factory())
    assert "versions referencing blind people" in results["models"][1]["summary"]["verdict"]


# ---------------------------------------------------------------------------
# All-pairs comparisons and ranking
# ---------------------------------------------------------------------------


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    adjusted = holm([0.2, float("nan"), 0.01])
    assert adjusted[0] == pytest.approx(0.2) and adjusted[2] == pytest.approx(0.02) and adjusted[1] != adjusted[1]


def test_four_variants_give_six_holm_adjusted_comparisons(study_dataset):
    results = compare_models(STUDY, study_dataset, axis="disability", controls="tagged", scorer_factory=_study_factory())
    pairs = [(c["reference"], c["model"]) for c in results["comparisons"]]
    assert pairs == [("M0", "M1"), ("M0", "M2"), ("M0", "M3"), ("M1", "M2"), ("M1", "M3"), ("M2", "M3")]
    raw = [c["delta_diff"]["p"] for c in results["comparisons"]]
    assert [c["delta_diff"]["p_holm"] for c in results["comparisons"]] == pytest.approx(holm(raw), nan_ok=True)
    rows = comparison_rows(results)
    assert rows[0]["model"] == "M1" and "delta_p_holm" in rows[0]


def test_reference_mode_compares_against_the_first_model_only(study_dataset):
    results = compare_models(
        STUDY, study_dataset, axis="disability", controls="tagged", scorer_factory=_study_factory(),
        comparisons="reference",
    )
    assert [(c["reference"], c["model"]) for c in results["comparisons"]] == [("M0", "M1"), ("M0", "M2"), ("M0", "M3")]


def test_ranking_orders_models_by_mean_delta(study_dataset):
    results = compare_models(STUDY, study_dataset, axis="disability", controls="tagged", scorer_factory=_study_factory())
    ranking = ranking_rows(results)
    assert [row["model"] for row in ranking] == ["M3", "M2", "M1", "M0"]
    assert [row["rank"] for row in ranking] == [1, 2, 3, 4]
    assert ranking[0]["significant"] is True


# ---------------------------------------------------------------------------
# Multi-axis runs
# ---------------------------------------------------------------------------


def test_compare_axes_scores_each_model_once_and_splits_rows_by_axis(study_dataset):
    calls: list[str] = []
    report = compare_axes(
        STUDY, study_dataset, axes=["gender", "disability"], controls="tagged", scorer_factory=_study_factory(calls)
    )
    assert calls == ["M0", "M1", "M2", "M3"]
    assert set(report["axes"]) == {"gender", "disability"}
    assert [a["name"] for a in report["run"]["axes"]] == ["gender", "disability"]
    assert len(report["axes"]["gender"]["posts"]) == 12
    assert len(report["axes"]["disability"]["posts"]) == 16
    gender = axis_results(report, "gender")
    assert gender["run"]["axis"] == "gender" and gender["posts"][0]["swap"]["source"] == "lexicon"
    assert gender["ranking"][0]["model"] == "M3"


def test_single_and_multi_axis_runs_agree(study_dataset):
    report = compare_axes(STUDY, study_dataset, axes=["gender", "disability"], controls="tagged",
                          scorer_factory=_study_factory())
    single = compare_models(STUDY, study_dataset, axis="disability", controls="tagged", scorer_factory=_study_factory())
    # disability-only rows sit in the same file positions either way, so the controls match
    assert report["axes"]["disability"]["posts"] == single["posts"]
    assert report["axes"]["disability"]["comparisons"] == single["comparisons"]


def test_multi_axis_loading_needs_one_known_attribute(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps({"id": "x", "text": "hi", "attributes": {"age": "old"}}) + "\n")
    with pytest.raises(ValueError, match="no attributes for any of the axes"):
        load_posts(path, axes=["gender", "disability"])
