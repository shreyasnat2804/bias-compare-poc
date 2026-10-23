"""Validates data/posts.jsonl against the requirements in DESIGN.md §5.1/§5.2.

Loads the JSONL directly with `json` (independently of `biascompare.inputs`)
and exercises the real perturb package against every post.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from biascompare.perturb import GenderSwapper, SynonymControls

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "posts.jsonl"

REQUIRED_FIELDS = {"id", "platform", "text", "intended_toxicity", "gender"}
VALID_PLATFORMS = {"twitter", "reddit"}
VALID_TOXICITY = {"clean", "mild", "borderline", "rude", "toxic"}
VALID_GENDERS = {"male", "female"}

EXPECTED_TOXICITY_COUNTS = {
    "clean": 1,
    "mild": 2,
    "borderline": 4,
    "rude": 2,
    "toxic": 1,
}
EXPECTED_GENDER_COUNTS = {"male": 5, "female": 5}
EXPECTED_PLATFORM_COUNTS = {"twitter": 5, "reddit": 5}


def _load_posts() -> list[dict]:
    """Rows flattened to the v0.1 field names: the file is in the v1 format
    (docs/DATA.md §2.3), with gender under attributes and the rest in meta."""
    lines = DATA_PATH.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    return [
        {"id": row["id"], "text": row["text"], "gender": row["attributes"]["gender"], **row["meta"]}
        for row in rows
    ]


def test_dataset_has_ten_posts():
    assert len(_load_posts()) == 10


def test_posts_have_required_fields_and_valid_values():
    for post in _load_posts():
        assert REQUIRED_FIELDS.issubset(post.keys()), post
        assert post["platform"] in VALID_PLATFORMS, post
        assert post["intended_toxicity"] in VALID_TOXICITY, post
        assert post["gender"] in VALID_GENDERS, post
        assert isinstance(post["text"], str) and post["text"].strip(), post


def test_post_ids_are_unique():
    ids = [post["id"] for post in _load_posts()]
    assert len(ids) == len(set(ids))


def test_toxicity_distribution_is_bell_shaped():
    counts = Counter(post["intended_toxicity"] for post in _load_posts())
    assert dict(counts) == EXPECTED_TOXICITY_COUNTS


def test_gender_distribution_is_balanced():
    counts = Counter(post["gender"] for post in _load_posts())
    assert dict(counts) == EXPECTED_GENDER_COUNTS


def test_platform_distribution_is_balanced():
    counts = Counter(post["platform"] for post in _load_posts())
    assert dict(counts) == EXPECTED_PLATFORM_COUNTS


@pytest.fixture(scope="module")
def gender_swapper() -> GenderSwapper:
    return GenderSwapper.from_resource()


@pytest.fixture(scope="module")
def synonym_controls(gender_swapper: GenderSwapper) -> SynonymControls:
    return SynonymControls.from_resources(extra_protected=gender_swapper.vocabulary)


@pytest.mark.parametrize("post", _load_posts(), ids=lambda post: post["id"])
def test_post_detected_gender_matches_label(post: dict, gender_swapper: GenderSwapper):
    assert gender_swapper.detect_group(post["text"]) == post["gender"], post["id"]


@pytest.mark.parametrize("post", _load_posts(), ids=lambda post: post["id"])
def test_post_gender_swap_changes_at_least_two_words(post: dict, gender_swapper: GenderSwapper):
    swap = gender_swapper.swap(post["text"])
    assert swap.n_changed >= 2, (post["id"], swap.changes)


@pytest.mark.parametrize("post", _load_posts(), ids=lambda post: post["id"])
def test_post_controls_reach_full_k_without_flags(
    post: dict, gender_swapper: GenderSwapper, synonym_controls: SynonymControls
):
    swap = gender_swapper.swap(post["text"])
    control_set = synonym_controls.generate(
        post["text"], n_words=swap.n_changed, k=5, rng=np.random.default_rng(0)
    )
    assert control_set.effective_k == 5, (post["id"], control_set.flags)
    assert control_set.flags == (), (post["id"], control_set.flags)
