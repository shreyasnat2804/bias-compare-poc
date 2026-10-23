"""Regression tests: a post with zero usable control perturbations must not
NaN-poison a model's statistics (code review finding, MEDIUM).

Before the fix, only `gender_mismatch`/`no_gender_terms` excluded a post
from statistics. A post whose only non-gender words have no synonym-list
entry gets `effective_k == 0` (`SynonymControls.generate` finds zero
candidates), so `compute_per_post_metrics` sets `delta = NaN`. That NaN used
to flow, uncaught, into the eligible-posts array for the whole model,
NaN-poisoning `mean_delta`/`p_delta`/`p_delta_bonferroni` even though every
other post had a perfectly good delta. The fix: flag such a post
`no_controls`, treat it as excluding (like the other two flags), and
defensively drop any remaining non-finite delta/direction before the
model's own hypothesis tests.

Uses the same fake scorers as `test_compare.py`, applied to the real
10-post dataset plus one contaminating post (`dataset_with_no_controls_post`
appends it to a temp copy of `data/posts.jsonl`), so the "other 9 posts
still finite" check inherits the same statistical behaviour already proven
by `test_compare.py`'s biased-vs-unbiased tests.
"""

from __future__ import annotations

import math

import pytest

from biascompare import ModelSpec, compare_models, preview_perturbations
from test_compare import BIASED_VS_UNBIASED, NO_CONTROLS_RECORD, _fake_factory, dataset_with_no_controls_post

_SUMMARY_KEYS_THAT_MUST_BE_FINITE = (
    "sensitivity",
    "raw_swap_effect",
    "mean_delta",
    "p_delta",
    "p_delta_bonferroni",
    "mean_direction",
    "p_direction",
    "flip_rate_swap",
    "flip_rate_controls",
)


@pytest.fixture()
def dataset(tmp_path):
    return dataset_with_no_controls_post(tmp_path)


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------


def test_zero_candidate_post_has_effective_k_zero_and_no_controls_flag(dataset):
    preview = preview_perturbations(dataset, n_controls=5, seed=0)
    zero_post = next(p for p in preview if p["id"] == NO_CONTROLS_RECORD["id"])
    assert zero_post["effective_k"] == 0
    assert "no_controls" in zero_post["flags"]


def test_other_posts_are_unaffected_by_the_flagged_post(dataset):
    preview = preview_perturbations(dataset, n_controls=5, seed=0)
    for post in preview:
        if post["id"] == NO_CONTROLS_RECORD["id"]:
            continue
        assert "no_controls" not in post["flags"]


# ---------------------------------------------------------------------------
# compare_models: excluded from stats, kept in per_post, no NaN-poisoning
# ---------------------------------------------------------------------------


def test_flagged_post_is_still_scored_and_kept_in_per_post(dataset):
    results = compare_models(BIASED_VS_UNBIASED, dataset, n_controls=5, seed=0, scorer_factory=_fake_factory)
    for model in results["models"]:
        post_ids = [entry["post_id"] for entry in model["per_post"]]
        assert NO_CONTROLS_RECORD["id"] in post_ids
        assert len(model["per_post"]) == 11  # 10 real posts + the flagged one


def test_flagged_post_excluded_from_eligible_count(dataset):
    results = compare_models(BIASED_VS_UNBIASED, dataset, n_controls=5, seed=0, scorer_factory=_fake_factory)
    for model in results["models"]:
        assert "n = 10" in model["summary"]["verdict"]


def test_summary_statistics_are_finite_despite_the_flagged_post(dataset):
    """The core regression check: with the fix, the 10 good posts still
    produce finite statistics even though an 11th post has no controls."""
    results = compare_models(BIASED_VS_UNBIASED, dataset, n_controls=5, seed=0, scorer_factory=_fake_factory)
    for model in results["models"]:
        summary = model["summary"]
        assert summary is not None
        for key in _SUMMARY_KEYS_THAT_MUST_BE_FINITE:
            assert math.isfinite(summary[key]), (model["name"], key, summary[key])
        assert all(math.isfinite(bound) for bound in summary["direction_ci"])


def test_flagged_posts_own_delta_is_still_nan_in_per_post(dataset):
    # The flagged post is scored and reported (transparency), but its own
    # delta is necessarily undefined (no controls to compare against) -
    # only the *aggregate* statistics must stay finite, not this one field.
    results = compare_models(BIASED_VS_UNBIASED, dataset, n_controls=5, seed=0, scorer_factory=_fake_factory)
    for model in results["models"]:
        flagged_entry = next(
            entry for entry in model["per_post"] if entry["post_id"] == NO_CONTROLS_RECORD["id"]
        )
        assert math.isnan(flagged_entry["delta"])


def test_comparison_statistics_are_finite_despite_the_flagged_post(dataset):
    results = compare_models(BIASED_VS_UNBIASED, dataset, n_controls=5, seed=0, scorer_factory=_fake_factory)
    comparison = results["comparisons"][0]
    for diff in (comparison["delta_diff"], comparison["direction_diff"]):
        assert math.isfinite(diff["mean"])
        assert math.isfinite(diff["p"])
        assert all(math.isfinite(bound) for bound in diff["ci"])


# ---------------------------------------------------------------------------
# viz.py: the flagged post must not appear in the per-post panels
# ---------------------------------------------------------------------------


def test_plot_comparison_slope_panels_skip_the_flagged_post(dataset):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from biascompare.viz import plot_comparison

    results = compare_models(BIASED_VS_UNBIASED, dataset, n_controls=5, seed=0, scorer_factory=_fake_factory)
    fig = plot_comparison(results)
    _, delta_ax, direction_ax = fig.axes

    # One gray connecting line per *eligible* post only (10, not 11).
    assert len(delta_ax.get_lines()) == 10
    # The direction panel has the same 10 connecting lines plus the zero line.
    assert len(direction_ax.get_lines()) == 11

    # Every model's scatter for the flagged post's would-be x-position must
    # not include a NaN; total scattered points per model == 10 (eligible).
    for collection in delta_ax.collections:
        offsets = collection.get_offsets()
        assert len(offsets) == 10
        assert not any(math.isnan(y) for _, y in offsets)
