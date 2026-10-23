"""Multi-level runs: compare_models with fake scorers over 3- and 4-level tasks,
distribution checks, gold labels and accuracy, and run warnings."""

from __future__ import annotations

import json
import math
import zlib

import pytest

from biascompare import ModelSpec, TaskSpec, compare_models
from test_compare import BIASED_VS_UNBIASED, DATASET, NO_CONTROLS_RECORD, _fake_factory

_FEMALE_WORDS = frozenset({"she", "her", "herself", "sister", "girlfriend", "woman", "girl"})


def _three_level_dist(text: str, biased: bool) -> tuple[float, float, float]:
    if biased:
        hits = sum(word.strip(".,!?'\"").lower() in _FEMALE_WORDS for word in text.split())
        top = min(0.1 + 0.15 * hits, 0.8)
    else:
        top = 0.1 + 0.2 * ((zlib.crc32(text.encode()) % 1000) / 1000.0)
    middle = 0.3
    return (1.0 - middle - top, middle, top)


class _ThreeLevelScorer:
    scorer_type = "fake-3"
    uses_instruction = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.metadata = {"label_mapping": {"note": "fake"}}

    def score(self, texts):
        return [_three_level_dist(text, self.name == "biased") for text in texts]

    def close(self) -> None:
        pass


def _three_level_factory(spec, task, device):
    assert task.n_levels == 3
    return _ThreeLevelScorer(spec.name)


def test_three_level_run_records_full_distributions():
    results = compare_models(
        BIASED_VS_UNBIASED, DATASET, task="toxicity-3", seed=0, scorer_factory=_three_level_factory
    )
    assert results["run"]["task"]["levels"] == ["non-toxic", "borderline", "toxic"]
    per_post = results["models"][0]["per_post"][0]
    assert len(per_post["p_original"]) == 3
    assert per_post["p_toxic_original"] == pytest.approx(per_post["p_original"][2])
    assert per_post["predicted_original"] in ("non-toxic", "borderline", "toxic")
    assert results["run"]["models"][0]["label_mapping"] == {"note": "fake"}


def test_three_level_direction_is_a_severity_difference():
    results = compare_models(
        BIASED_VS_UNBIASED, DATASET, task="toxicity-3", seed=0, scorer_factory=_three_level_factory
    )
    posts = {post["id"]: post for post in results["posts"]}
    for metric in results["models"][0]["per_post"]:
        post = posts[metric["post_id"]]
        orig, swap = metric["p_original"], metric["p_gender_swap"]
        sev_orig = (orig[1] + 2 * orig[2]) / 2
        sev_swap = (swap[1] + 2 * swap[2]) / 2
        expected = sev_orig - sev_swap if post["gender"] == "female" else sev_swap - sev_orig
        assert metric["direction"] == pytest.approx(expected)


def test_three_level_biased_model_is_detected():
    results = compare_models(
        BIASED_VS_UNBIASED, DATASET, task="toxicity-3", seed=0, scorer_factory=_three_level_factory
    )
    biased = results["models"][0]["summary"]
    assert biased["mean_delta"] > 0
    assert biased["mean_direction"] > 0


def test_binary_scorer_on_a_three_level_task_is_recorded_as_an_error():
    results = compare_models(
        BIASED_VS_UNBIASED, DATASET, task="toxicity-3", seed=0, scorer_factory=_fake_factory
    )
    for model in results["models"]:
        assert model["summary"] is None
        assert "not a distribution over the 3 levels" in model["error"]


def test_non_normalised_scores_are_recorded_as_an_error():
    class _Broken(_ThreeLevelScorer):
        def score(self, texts):
            return [(0.5, 0.5, 0.5) for _ in texts]

    results = compare_models(
        BIASED_VS_UNBIASED, DATASET, task="toxicity-3", seed=0,
        scorer_factory=lambda spec, task, device: _Broken(spec.name),
    )
    assert "must sum to 1" in results["models"][0]["error"]


def _labelled_dataset(tmp_path, labels: dict[str, str]):
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["id"] in labels:
            row["label"] = labels[row["id"]]
    path = tmp_path / "labelled.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def test_gold_labels_give_an_accuracy_per_model(tmp_path):
    dataset = _labelled_dataset(tmp_path, {"p01": "non-toxic", "p02": "Non-Toxic", "p10": "toxic"})
    results = compare_models(BIASED_VS_UNBIASED, dataset, seed=0, scorer_factory=_fake_factory)
    for model in results["models"]:
        summary = model["summary"]
        assert summary["n_labeled"] == 3
        predicted = {m["post_id"]: m["predicted_original"] for m in model["per_post"]}
        expected = sum(
            predicted[pid] == label for pid, label in {"p01": "non-toxic", "p02": "non-toxic", "p10": "toxic"}.items()
        ) / 3
        assert summary["accuracy"] == pytest.approx(expected)


def test_gold_label_outside_the_task_levels_raises(tmp_path):
    dataset = _labelled_dataset(tmp_path, {"p01": "severe"})
    with pytest.raises(ValueError, match="p01='severe'"):
        compare_models(BIASED_VS_UNBIASED, dataset, seed=0, scorer_factory=_fake_factory)


def test_instruction_argument_overrides_the_task_instruction():
    results = compare_models(
        BIASED_VS_UNBIASED, DATASET, instruction="Is this post toxic?", seed=0, scorer_factory=_fake_factory
    )
    assert results["run"]["instruction"] == "Is this post toxic?"
    assert results["run"]["task"]["instruction"] == "Is this post toxic?"


def test_custom_task_spec_is_accepted():
    task = TaskSpec(name="three", levels=("ok", "rude", "hateful"), toxic_from="hateful")
    results = compare_models(BIASED_VS_UNBIASED, DATASET, task=task, seed=0, scorer_factory=_three_level_factory)
    assert results["run"]["task"]["name"] == "three"


def test_unknown_model_type_fails_before_any_work():
    models = [ModelSpec("a", type="hf-classifier"), ModelSpec("b", type="carrier-pigeon")]
    with pytest.raises(ValueError, match="carrier-pigeon"):
        compare_models(models, DATASET)


# ---------------------------------------------------------------------------
# Run warnings
# ---------------------------------------------------------------------------


def test_sample_dataset_warns_about_low_n_only():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, seed=0, scorer_factory=_fake_factory)
    assert len(results["warnings"]) == 1
    assert "Only 10 rows are eligible" in results["warnings"][0]


def test_excluded_rows_and_imbalance_are_warned_about(tmp_path):
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    lines += [json.dumps({**NO_CONTROLS_RECORD, "id": f"zero{i}"}) for i in range(3)]
    female_ids = [json.loads(line)["id"] for line in lines if '"female"' in line][:2]
    kept = [line for line in lines if json.loads(line)["id"] not in female_ids]
    path = tmp_path / "skewed.jsonl"
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")

    warnings = compare_models(BIASED_VS_UNBIASED, path, seed=0, scorer_factory=_fake_factory)["warnings"]
    assert any("3 of 11 rows are excluded from statistics (no_controls: 3)" in w for w in warnings)
    assert any("masked-lm" in w for w in warnings)
    assert any("unbalanced between groups (male: 5, female: 3)" in w for w in warnings)


def test_effect_size_is_mean_over_sd_of_deltas():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, seed=0, scorer_factory=_fake_factory)
    summary = results["models"][0]["summary"]
    deltas = [m["delta"] for m in results["models"][0]["per_post"]]
    mean = sum(deltas) / len(deltas)
    sd = math.sqrt(sum((d - mean) ** 2 for d in deltas) / (len(deltas) - 1))
    assert summary["effect_size"] == pytest.approx(mean / sd)
    assert summary["n_eligible"] == 10
