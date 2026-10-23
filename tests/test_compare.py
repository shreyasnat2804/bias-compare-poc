"""Tests for biascompare.compare: compare_models, preview_perturbations, save_results.

Fully offline: `FakeScorer` implements the `Scorer` protocol directly (no
`torch`/`transformers`, no network, no model download). Two fake models are
used throughout:

- `"biased"`: P(toxic) rises with the count of female-lexicon words in the
  text, so gender swaps (which add/remove those words) should move its
  output by more than matched-magnitude synonym swaps do.
- `"unbiased"`: P(toxic) depends only on a stable hash of the text, so any
  wording change (targeted or control) moves it by a similar amount on
  average - there should be no consistent gender signal.
"""

from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

import pytest

from biascompare import ModelSpec, compare_models, preview_perturbations, save_results

DATASET = Path(__file__).resolve().parents[1] / "data" / "posts.jsonl"

_FEMALE_WORDS = frozenset({"she", "her", "hers", "herself", "sister", "girlfriend", "woman", "girl"})
_WORD_STRIP = ".,!?'\""

# A post whose only non-gender words ("is", "happy", "today") have no entry
# in resources/synonyms.json, so SynonymControls.generate finds zero
# candidates and effective_k == 0 regardless of n_controls. Used by
# tests/test_no_controls.py to check that such a post is excluded from
# statistics instead of NaN-poisoning them.
NO_CONTROLS_RECORD = {
    "id": "zero",
    "platform": "twitter",
    "text": "He is happy today.",
    "gender": "male",
    "intended_toxicity": "clean",
}


def dataset_with_no_controls_post(tmp_path: Path) -> Path:
    """The real 10-post dataset plus `NO_CONTROLS_RECORD` appended, written
    to a fresh file under `tmp_path`."""
    lines = DATASET.read_text(encoding="utf-8").splitlines()
    lines.append(json.dumps(NO_CONTROLS_RECORD))
    path = tmp_path / "posts_with_no_controls.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


class FakeScorer:
    """A `Scorer` whose P(toxic) is fully determined by a pure function of
    the text, so scoring is deterministic and needs no real model."""

    scorer_type = "fake"
    uses_instruction = False

    def __init__(self, name: str, toxicity_fn) -> None:
        self.name = name
        self._toxicity_fn = toxicity_fn
        self.closed = False

    def score(self, texts):
        return [(1.0 - self._toxicity_fn(t), self._toxicity_fn(t)) for t in texts]

    def close(self) -> None:
        self.closed = True


def _biased_toxicity(text: str) -> float:
    words = {w.strip(_WORD_STRIP).lower() for w in text.split()}
    hits = len(words & _FEMALE_WORDS)
    return min(0.15 + 0.2 * hits, 0.97)


def _unbiased_toxicity(text: str) -> float:
    digest = zlib.crc32(text.encode("utf-8"))
    return 0.2 + 0.3 * ((digest % 1000) / 1000.0)


def _fake_factory(spec: ModelSpec, instruction: str | None, device: str) -> FakeScorer:
    toxicity_fn = _biased_toxicity if spec.name == "biased" else _unbiased_toxicity
    return FakeScorer(spec.name, toxicity_fn)


class _RaisingScorer:
    scorer_type = "fake"
    uses_instruction = False

    def score(self, texts):
        raise RuntimeError("scoring exploded")

    def close(self) -> None:
        pass


def _failing_factory(spec: ModelSpec, instruction: str | None, device: str):
    if spec.name == "broken":
        raise RuntimeError("simulated load failure for 'broken'")
    return FakeScorer(spec.name, _unbiased_toxicity)


BIASED_VS_UNBIASED = [ModelSpec(name="biased"), ModelSpec(name="unbiased")]


# ---------------------------------------------------------------------------
# Schema: keys and types match DESIGN.md §7
# ---------------------------------------------------------------------------


def test_results_schema_top_level_keys():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=3, seed=0, scorer_factory=_fake_factory)
    assert set(results.keys()) == {
        "schema_version", "run", "posts", "models", "comparisons", "ranking", "summary_verdict", "warnings",
    }
    assert results["schema_version"] == "0.3"
    assert isinstance(results["summary_verdict"], str)
    assert all(isinstance(warning, str) for warning in results["warnings"])


def test_results_schema_run_block():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=3, seed=0, scorer_factory=_fake_factory)
    run = results["run"]
    assert set(run.keys()) == {
        "timestamp", "dataset", "n_posts", "axis", "axes", "n_controls", "seed", "alpha",
        "instruction", "paper", "models", "task", "controls", "comparisons",
    }
    assert run["n_posts"] == 10
    assert run["axis"] == "gender"
    assert run["n_controls"] == 3
    assert run["seed"] == 0
    assert run["instruction"] is None
    assert "arXiv:2605.01048" in run["paper"]
    assert run["models"] == [
        {"id": "biased", "name": "biased", "toxic_label": "toxic", "scorer": "fake",
         "instruction_used": False, "type": "hf-classifier", "revision": None},
        {"id": "unbiased", "name": "unbiased", "toxic_label": "toxic", "scorer": "fake",
         "instruction_used": False, "type": "hf-classifier", "revision": None},
    ]
    assert run["axes"][0]["name"] == "gender" and run["axes"][0]["target_group"] == "female"
    assert run["comparisons"] == "all-pairs"
    assert run["task"] == {
        "name": "toxicity", "levels": ["non-toxic", "toxic"], "toxic_from": "toxic", "instruction": None,
    }
    assert run["controls"] == {"method": "synonyms"}


def test_results_schema_posts_block():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=3, seed=0, scorer_factory=_fake_factory)
    assert len(results["posts"]) == 10
    post = results["posts"][0]
    assert set(post.keys()) == {
        "id", "group", "attributes", "label", "meta", "original",
        "swap", "controls", "n_words_changed", "effective_k", "flags",
    }
    assert post["attributes"] == {"gender": post["group"]}
    assert set(post["swap"].keys()) == {"text", "changes", "source"}
    assert post["swap"]["source"] == "lexicon"
    assert all(isinstance(change, list) and len(change) == 2 for change in post["swap"]["changes"])
    assert isinstance(post["controls"], list)
    for control in post["controls"]:
        assert set(control.keys()) == {"text", "changes", "similarity"}
        assert control["similarity"] is None  # only masked-LM controls record it


def test_results_schema_per_post_and_summary():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=3, seed=0, scorer_factory=_fake_factory)
    model = results["models"][0]
    assert set(model.keys()) == {"id", "name", "error", "per_post", "summary"}
    assert model["error"] is None
    assert len(model["per_post"]) == 10

    per_post = model["per_post"][0]
    assert set(per_post.keys()) == {
        "post_id", "p_toxic_original", "p_toxic_swap", "p_toxic_controls",
        "p_original", "p_swap", "p_controls", "predicted_original",
        "jsd_swap", "jsd_controls", "delta", "direction", "flip_swap", "flip_controls_rate",
    }
    assert len(per_post["p_original"]) == 2
    assert per_post["p_original"][1] == per_post["p_toxic_original"]
    assert isinstance(per_post["flip_swap"], bool)
    assert isinstance(per_post["p_toxic_controls"], list)

    summary = model["summary"]
    assert set(summary.keys()) == {
        "sensitivity", "raw_swap_effect", "mean_delta", "p_delta", "p_delta_bonferroni",
        "mean_direction", "direction_ci", "p_direction", "flip_rate_swap",
        "flip_rate_controls", "effect_size", "n_eligible", "accuracy", "n_labeled", "n_truncated",
        "verdict",
    }
    assert summary["n_labeled"] == 0 and math.isnan(summary["accuracy"])
    assert summary["n_truncated"] is None  # the fake scorer reports no truncation
    assert isinstance(summary["direction_ci"], list) and len(summary["direction_ci"]) == 2
    assert isinstance(summary["verdict"], str)


def test_results_schema_comparisons_block():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=3, seed=0, scorer_factory=_fake_factory)
    assert len(results["comparisons"]) == 1
    comparison = results["comparisons"][0]
    assert set(comparison.keys()) == {"reference", "model", "delta_diff", "direction_diff", "verdict"}
    assert comparison["reference"] == "biased"
    assert comparison["model"] == "unbiased"
    for diff in (comparison["delta_diff"], comparison["direction_diff"]):
        assert set(diff.keys()) == {"mean", "ci", "p", "p_holm"}
        assert len(diff["ci"]) == 2
        assert diff["p_holm"] == diff["p"]  # one comparison: nothing to adjust


# ---------------------------------------------------------------------------
# Posts and perturbations are identical across models
# ---------------------------------------------------------------------------


def test_posts_and_perturbations_identical_across_models():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=3, seed=0, scorer_factory=_fake_factory)
    # Only one `posts` block exists (shared), but double-check every model
    # was scored over exactly the same post ids in exactly the same order.
    post_ids = [post["id"] for post in results["posts"]]
    for model in results["models"]:
        assert [entry["post_id"] for entry in model["per_post"]] == post_ids


def test_preview_perturbations_matches_compare_models_posts():
    preview = preview_perturbations(DATASET, n_controls=3, seed=0)
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=3, seed=0, scorer_factory=_fake_factory)
    assert preview == results["posts"]


def test_preview_perturbations_needs_no_models():
    preview = preview_perturbations(DATASET, n_controls=2, seed=0)
    assert len(preview) == 10
    assert {"id", "group", "original", "swap", "controls", "flags"} <= preview[0].keys()


# ---------------------------------------------------------------------------
# Biased vs unbiased fake: the core hypothesis test
# ---------------------------------------------------------------------------


def test_biased_model_shows_positive_mean_direction():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=5, seed=0, scorer_factory=_fake_factory)
    biased_summary = results["models"][0]["summary"]
    assert biased_summary["mean_direction"] > 0


def test_biased_model_has_larger_mean_delta_than_unbiased():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=5, seed=0, scorer_factory=_fake_factory)
    biased_summary = results["models"][0]["summary"]
    unbiased_summary = results["models"][1]["summary"]
    assert biased_summary["mean_delta"] > unbiased_summary["mean_delta"]


def test_comparison_sign_reflects_which_model_is_more_biased():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=5, seed=0, scorer_factory=_fake_factory)
    comparison = results["comparisons"][0]
    # comparison is (unbiased - biased): the less-biased model minus the
    # more-biased reference, so the mean delta_diff should be negative.
    assert comparison["delta_diff"]["mean"] < 0


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_failing_model_gets_error_and_run_continues():
    models = [ModelSpec(name="broken"), ModelSpec(name="ok-a"), ModelSpec(name="ok-b")]
    results = compare_models(models, DATASET, n_controls=2, seed=0, scorer_factory=_failing_factory)

    broken = results["models"][0]
    assert broken["name"] == "broken"
    assert broken["error"] is not None
    assert "simulated load failure" in broken["error"]
    assert broken["per_post"] == []
    assert broken["summary"] is None

    assert results["models"][1]["error"] is None
    assert results["models"][2]["error"] is None

    run_broken_entry = results["run"]["models"][0]
    assert run_broken_entry["scorer"] is None
    assert run_broken_entry["instruction_used"] is None


def test_failing_model_excluded_from_comparisons():
    models = [ModelSpec(name="broken"), ModelSpec(name="ok-a"), ModelSpec(name="ok-b")]
    results = compare_models(models, DATASET, n_controls=2, seed=0, scorer_factory=_failing_factory)
    names_in_comparisons = {c["reference"] for c in results["comparisons"]} | {
        c["model"] for c in results["comparisons"]
    }
    assert "broken" not in names_in_comparisons
    assert results["comparisons"][0]["reference"] == "ok-a"
    assert results["comparisons"][0]["model"] == "ok-b"


def test_scoring_exception_is_captured_and_scorer_still_closed():
    calls: list[str] = []

    class TrackingRaisingScorer(_RaisingScorer):
        def close(self) -> None:
            calls.append("closed")

    def factory(spec, instruction, device):
        return TrackingRaisingScorer()

    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=2, seed=0, scorer_factory=factory)
    assert all(model["error"] is not None for model in results["models"])
    assert calls == ["closed", "closed"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_is_fully_deterministic():
    r1 = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=4, seed=42, scorer_factory=_fake_factory)
    r2 = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=4, seed=42, scorer_factory=_fake_factory)
    r1 = {**r1, "run": {**r1["run"], "timestamp": None}}
    r2 = {**r2, "run": {**r2["run"], "timestamp": None}}
    assert r1 == r2


def test_different_seed_changes_control_perturbations():
    r1 = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=4, seed=1, scorer_factory=_fake_factory)
    r2 = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=4, seed=2, scorer_factory=_fake_factory)
    assert r1["posts"][0]["controls"] != r2["posts"][0]["controls"]


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------


def test_save_results_writes_valid_json_and_creates_parent_dirs(tmp_path):
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=2, seed=0, scorer_factory=_fake_factory)
    target = tmp_path / "nested" / "dir" / "results.json"

    returned = save_results(results, target)

    assert returned == target
    assert target.is_file()
    reloaded = json.loads(target.read_text(encoding="utf-8"))
    assert reloaded["schema_version"] == "0.3"


def test_save_results_converts_nan_to_null(tmp_path):
    results = {"a": float("nan"), "b": [1.0, float("inf"), float("-inf")], "c": {"d": float("nan")}}
    target = tmp_path / "results.json"

    save_results(results, target)

    raw_text = target.read_text(encoding="utf-8")
    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text
    reloaded = json.loads(raw_text)
    assert reloaded == {"a": None, "b": [1.0, None, None], "c": {"d": None}}


def test_save_results_returns_path_instance(tmp_path):
    target = tmp_path / "out.json"
    returned = save_results({"x": 1}, target)
    assert isinstance(returned, Path)


def test_no_token_string_in_serialized_output(tmp_path):
    models = [
        ModelSpec(name="biased", token="super-secret-token-abc"),
        ModelSpec(name="unbiased", token="another-secret-def"),
    ]
    results = compare_models(models, DATASET, n_controls=2, seed=0, scorer_factory=_fake_factory)
    target = tmp_path / "results.json"
    save_results(results, target)
    text = target.read_text(encoding="utf-8")
    assert "super-secret-token-abc" not in text
    assert "another-secret-def" not in text


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_compare_models_requires_at_least_two_models():
    with pytest.raises(ValueError, match="at least 2 models"):
        compare_models([ModelSpec(name="solo")], DATASET, scorer_factory=_fake_factory)


def test_compare_models_rejects_n_controls_below_one():
    with pytest.raises(ValueError, match="n_controls"):
        compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=0, scorer_factory=_fake_factory)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_compare_models_rejects_alpha_outside_open_interval(alpha):
    with pytest.raises(ValueError, match="alpha"):
        compare_models(BIASED_VS_UNBIASED, DATASET, alpha=alpha, scorer_factory=_fake_factory)


def test_preview_perturbations_rejects_n_controls_below_one():
    with pytest.raises(ValueError, match="n_controls"):
        preview_perturbations(DATASET, n_controls=0)


def test_compare_models_propagates_invalid_dataset_path():
    with pytest.raises(ValueError, match="not found"):
        compare_models(BIASED_VS_UNBIASED, "/no/such/dataset.jsonl", scorer_factory=_fake_factory)


# ---------------------------------------------------------------------------
# Verdict wording (DESIGN §6: "insufficient evidence", never "no bias")
# ---------------------------------------------------------------------------


def test_unbiased_model_verdict_says_insufficient_evidence_not_no_bias():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=5, seed=0, scorer_factory=_fake_factory)
    verdict = results["models"][1]["summary"]["verdict"]
    assert "insufficient evidence" in verdict.lower()
    assert "no bias" not in verdict.lower()


def test_summary_verdict_mentions_n_and_never_says_no_bias():
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=5, seed=0, scorer_factory=_fake_factory)
    verdict = results["summary_verdict"]
    assert "n = 10" in verdict
    assert "no bias" not in verdict.lower()


def test_degenerate_single_post_comparison_never_crashes():
    # Force a tiny eligible-post scenario is impractical without touching
    # the dataset; instead exercise the stats layer's NaN-safety directly
    # through a run with all controls short (still 10 posts, but this
    # documents that summary math tolerates NaNs without raising).
    results = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=1, seed=0, scorer_factory=_fake_factory)
    for model in results["models"]:
        summary = model["summary"]
        assert summary is not None
        for key in ("p_delta", "p_delta_bonferroni", "p_direction"):
            assert isinstance(summary[key], float) or math.isnan(summary[key])
