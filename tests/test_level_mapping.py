"""Tests for scoring.base.resolve_level_mapping / distribution_from_logits:
how a classifier's own labels become a distribution over task levels.
No torch needed."""

from __future__ import annotations

import math

import pytest

from biascompare.scoring.base import distribution_from_logits, resolve_level_mapping
from biascompare.tasks import BUILTIN_TASKS, DEFAULT_TASK

TASK3 = BUILTIN_TASKS["toxicity-3"]


def _softmax(xs):
    exps = [math.exp(x) for x in xs]
    return [e / sum(exps) for e in exps]


def test_binary_softmax_default_puts_every_other_label_in_the_bottom_level():
    mapping = resolve_level_mapping({0: "neutral", 1: "toxic"}, DEFAULT_TASK, multi_label=False)
    assert mapping.description == {"non-toxic": ["neutral"], "toxic": ["toxic"]}
    dist = distribution_from_logits([0.2, 1.8], mapping)
    assert dist[1] == pytest.approx(_softmax([0.2, 1.8])[1])
    assert sum(dist) == pytest.approx(1.0)


def test_multi_label_reads_one_sigmoid_head():
    id2label = {"0": "toxic", "1": "severe_toxic", "2": "insult"}
    mapping = resolve_level_mapping(id2label, DEFAULT_TASK, multi_label=True, toxic_label="Toxic")
    assert mapping.head_index == 0
    dist = distribution_from_logits([2.0, -1.0, 3.0], mapping)
    assert dist[1] == pytest.approx(1 / (1 + math.exp(-2.0)))


def test_multi_label_head_can_come_from_label_map():
    mapping = resolve_level_mapping(
        {0: "toxic", 1: "severe_toxic"}, DEFAULT_TASK, multi_label=True, label_map={"toxic": ["severe_toxic"]}
    )
    assert mapping.head_index == 1


def test_multi_label_model_cannot_take_a_three_level_task():
    with pytest.raises(ValueError, match="only be scored on a 2-level task"):
        resolve_level_mapping({0: "toxic"}, TASK3, multi_label=True)


def test_three_level_softmax_with_label_map_sums_probabilities():
    id2label = {0: "normal", 1: "offensive", 2: "hate speech", 3: "spam"}
    mapping = resolve_level_mapping(
        id2label, TASK3, multi_label=False,
        label_map={"non-toxic": ["normal", "spam"], "borderline": ["offensive"], "toxic": ["Hate Speech"]},
    )
    logits = [1.0, 0.5, -0.5, 0.0]
    probs = _softmax(logits)
    dist = distribution_from_logits(logits, mapping)
    assert dist == pytest.approx((probs[0] + probs[3], probs[1], probs[2]))


def test_lowest_level_may_be_omitted_and_collects_the_rest():
    id2label = {0: "normal", 1: "offensive", 2: "hate", 3: "spam"}
    mapping = resolve_level_mapping(
        id2label, TASK3, multi_label=False, label_map={"borderline": ["offensive"], "toxic": ["hate"]}
    )
    assert mapping.description["non-toxic"] == ["normal", "spam"]


def test_three_level_softmax_without_label_map_matches_level_names():
    mapping = resolve_level_mapping({0: "Toxic", 1: "non-toxic", 2: "borderline"}, TASK3, multi_label=False)
    assert mapping.groups == ((1,), (2,), (0,))


def test_three_level_softmax_without_label_map_and_other_names_raises():
    with pytest.raises(ValueError, match="Pass a label_map"):
        resolve_level_mapping({0: "neutral", 1: "toxic"}, TASK3, multi_label=False)


def test_label_map_errors_name_the_problem():
    id2label = {0: "normal", 1: "offensive", 2: "hate"}
    with pytest.raises(ValueError, match="not found among model labels"):
        resolve_level_mapping(id2label, TASK3, False, label_map={"toxic": ["slur"]})
    with pytest.raises(ValueError, match="has levels"):
        resolve_level_mapping(id2label, TASK3, False, label_map={"severe": ["hate"]})
    with pytest.raises(ValueError, match="under both"):
        resolve_level_mapping(id2label, TASK3, False, label_map={"borderline": ["hate"], "toxic": ["hate"]})
    with pytest.raises(ValueError, match="no model label to level"):
        resolve_level_mapping(
            id2label, TASK3, False, label_map={"non-toxic": ["normal", "offensive"], "toxic": ["hate"]}
        )
    with pytest.raises(ValueError, match="unassigned"):
        resolve_level_mapping(
            id2label, TASK3, False, label_map={"non-toxic": ["normal"], "borderline": ["offensive"]}
        )
