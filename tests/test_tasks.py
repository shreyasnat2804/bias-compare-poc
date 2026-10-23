"""Tests for biascompare.tasks: TaskSpec validation and the distribution summaries."""

from __future__ import annotations

import pytest

from biascompare.tasks import BUILTIN_TASKS, DEFAULT_TASK, TaskSpec, get_task


def test_builtin_tasks_have_two_to_four_levels():
    assert BUILTIN_TASKS["toxicity"].levels == ("non-toxic", "toxic")
    assert BUILTIN_TASKS["toxicity-3"].levels == ("non-toxic", "borderline", "toxic")
    assert BUILTIN_TASKS["toxicity-4"].levels == ("non-toxic", "mild", "toxic", "severe")
    assert BUILTIN_TASKS["toxicity-4"].toxic_from == "toxic"


@pytest.mark.parametrize("levels", [("only",), ("a", "b", "c", "d", "e")])
def test_task_rejects_too_few_or_too_many_levels(levels):
    with pytest.raises(ValueError, match="between 2 and 4 levels"):
        TaskSpec(name="t", levels=levels)


def test_task_rejects_duplicate_levels_case_insensitively():
    with pytest.raises(ValueError, match="unique"):
        TaskSpec(name="t", levels=("Toxic", "toxic"))


def test_task_toxic_from_defaults_to_second_level():
    assert TaskSpec(name="t", levels=("a", "b", "c")).toxic_from == "b"


def test_task_toxic_from_must_be_a_level_above_the_lowest():
    with pytest.raises(ValueError, match="not one of its levels"):
        TaskSpec(name="t", levels=("a", "b"), toxic_from="z")
    with pytest.raises(ValueError, match="lowest level"):
        TaskSpec(name="t", levels=("a", "b"), toxic_from="a")


def test_two_level_summaries_equal_p_toxic_exactly():
    dist = (0.3, 0.7)
    assert DEFAULT_TASK.p_toxic(dist) == 0.7
    assert DEFAULT_TASK.severity(dist) == 0.7


def test_four_level_p_toxic_and_severity():
    task = BUILTIN_TASKS["toxicity-4"]
    dist = (0.1, 0.2, 0.3, 0.4)
    assert task.p_toxic(dist) == pytest.approx(0.7)
    assert task.severity(dist) == pytest.approx((0.2 * 1 + 0.3 * 2 + 0.4 * 3) / 3)


def test_predicted_level_ties_go_to_the_more_toxic_level():
    task = BUILTIN_TASKS["toxicity-3"]
    assert task.predicted_level((0.4, 0.4, 0.2)) == 1
    assert task.predicted_level((0.5, 0.2, 0.3)) == 0
    assert DEFAULT_TASK.predicted_level((0.5, 0.5)) == 1  # the POC's P(toxic) >= 0.5 rule


def test_distribution_length_must_match_levels():
    with pytest.raises(ValueError, match="3 values but task"):
        DEFAULT_TASK.severity((0.2, 0.3, 0.5))


def test_level_index_is_case_insensitive():
    assert BUILTIN_TASKS["toxicity-3"].level_index("Borderline") == 1
    with pytest.raises(ValueError, match="not a level"):
        BUILTIN_TASKS["toxicity-3"].level_index("severe")


def test_get_task_resolves_names_and_defaults():
    assert get_task(None) is DEFAULT_TASK
    assert get_task("toxicity-4").n_levels == 4
    custom = TaskSpec(name="c", levels=("x", "y"))
    assert get_task(custom) is custom
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("sentiment")
