"""Unit tests for biascompare.perturb (gender swap + synonym controls)."""
from __future__ import annotations

import numpy as np
import pytest

from biascompare.perturb import Change, ControlSet, GenderSwapper, Perturbation, SynonymControls

# ---------------------------------------------------------------------------
# GenderSwapper
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def swapper() -> GenderSwapper:
    return GenderSwapper.from_resource()


def test_swap_pronouns_case_lowercase(swapper: GenderSwapper):
    result = swapper.swap("he is happy")
    assert result.kind == "gender"
    assert result.text == "she is happy"
    assert result.changes == (Change("he", "she"),)


def test_swap_pronouns_case_titlecase(swapper: GenderSwapper):
    result = swapper.swap("He is happy")
    assert result.text == "She is happy"


def test_swap_pronouns_case_uppercase(swapper: GenderSwapper):
    result = swapper.swap("HE IS HAPPY")
    assert result.text == "SHE IS HAPPY"


def test_swap_names_bidirectional_in_one_pass(swapper: GenderSwapper):
    result = swapper.swap("David called Sarah.")
    assert result.text == "Sarah called David."
    assert result.n_changed == 2


def test_her_disambiguation_end_of_text_favors_him(swapper: GenderSwapper):
    # "her" is the last word before the sentence ends -> object pronoun -> "him".
    result = swapper.swap("I trust her.")
    assert result.text == "I trust him."


def test_her_disambiguation_function_word_favors_him(swapper: GenderSwapper):
    # "and" is a function word -> object pronoun -> "him".
    result = swapper.swap("I gave it to her and left.")
    assert result.text == "I gave it to him and left."


def test_her_disambiguation_before_noun_favors_his(swapper: GenderSwapper):
    # "decision" is an ordinary noun, not a function word -> possessive -> "his".
    result = swapper.swap("It was her decision.")
    assert result.text == "It was his decision."


def test_her_disambiguation_before_the_favors_him(swapper: GenderSwapper):
    # "the" is in the function-word set even though it precedes a noun phrase.
    result = swapper.swap("She gave her the keys.")
    assert result.text == "He gave him the keys."


def test_swap_no_gender_words_leaves_text_unchanged(swapper: GenderSwapper):
    result = swapper.swap("The weather is nice today.")
    assert result.text == "The weather is nice today."
    assert result.n_changed == 0
    assert result.changes == ()


def test_double_swap_is_idempotent_on_unambiguous_text(swapper: GenderSwapper):
    original = "He said he would call David."
    once = swapper.swap(original)
    twice = swapper.swap(once.text)
    assert twice.text == original


def test_detect_group_male(swapper: GenderSwapper):
    assert swapper.detect_group("He said he would help him.") == "male"


def test_detect_group_female(swapper: GenderSwapper):
    assert swapper.detect_group("She said she would help her.") == "female"


def test_detect_group_none_when_no_gender_words(swapper: GenderSwapper):
    assert swapper.detect_group("The weather is nice today.") is None


def test_detect_group_none_on_tie(swapper: GenderSwapper):
    assert swapper.detect_group("He met her.") is None


def test_vocabulary_contains_expected_words(swapper: GenderSwapper):
    vocab = swapper.vocabulary
    assert isinstance(vocab, frozenset)
    for word in ("he", "she", "him", "her", "his", "brother", "sister", "david", "sarah"):
        assert word in vocab


def test_groups_attribute(swapper: GenderSwapper):
    assert swapper.groups == ("male", "female")


def test_conflicting_pairs_raise():
    with pytest.raises(ValueError):
        GenderSwapper(groups=("male", "female"), pairs=[("he", "she"), ("he", "her")])


def test_ambiguous_reverse_mapping_without_disambiguation_rule_raises():
    swapper = GenderSwapper(groups=("male", "female"), pairs=[("guy", "girl"), ("boy", "girl")])
    with pytest.raises(ValueError):
        swapper.swap("The girl smiled.")


# ---------------------------------------------------------------------------
# SynonymControls
# ---------------------------------------------------------------------------

_TOY_SYNONYMS = {
    "quick": ["fast", "speedy", "rapid"],
    "happy": ["glad", "cheerful"],
    "big": ["large", "huge"],
    "small": ["tiny", "little"],
    "loud": ["noisy", "boisterous"],
    "idiot": ["fool", "dummy"],
}
_TOY_PROTECTED = {"idiot", "stupid"}
_TOY_TEXT = (
    "The quick brown fox was happy and big, not small, not loud, "
    "and definitely not an idiot or stupid animal."
)


@pytest.fixture()
def toy_controls() -> SynonymControls:
    return SynonymControls(synonyms=_TOY_SYNONYMS, protected=_TOY_PROTECTED)


def test_candidates_excludes_protected_words_even_if_in_synonym_list(toy_controls: SynonymControls):
    matched = {m.group(0).lower() for m in toy_controls.candidates(_TOY_TEXT)}
    assert matched == {"quick", "happy", "big", "small", "loud"}
    assert "idiot" not in matched
    assert "stupid" not in matched


def test_generate_changes_exactly_n_words_matched(toy_controls: SynonymControls):
    result = toy_controls.generate(_TOY_TEXT, n_words=3, k=5, rng=np.random.default_rng(0))
    assert isinstance(result, ControlSet)
    assert result.n_words_requested == 3
    assert result.n_words_matched == 3
    for perturbation in result.perturbations:
        assert isinstance(perturbation, Perturbation)
        assert perturbation.kind == "control"
        assert perturbation.n_changed == 3


def test_generate_never_touches_protected_words(toy_controls: SynonymControls):
    result = toy_controls.generate(_TOY_TEXT, n_words=3, k=5, rng=np.random.default_rng(1))
    for perturbation in result.perturbations:
        changed_words = {c.original.lower() for c in perturbation.changes}
        assert not changed_words & _TOY_PROTECTED
        assert "idiot" in perturbation.text.lower()
        assert "stupid" in perturbation.text.lower()


def test_generate_never_touches_gender_words():
    synonyms = {"he": ["dude"], "quick": ["fast", "speedy"]}
    gender = GenderSwapper(groups=("male", "female"), pairs=[("he", "she")])
    controls = SynonymControls(synonyms=synonyms, protected=gender.vocabulary)
    matches = controls.candidates("He was quick.")
    assert all(m.group(0).lower() != "he" for m in matches)
    result = controls.generate("He was quick.", n_words=1, k=3, rng=np.random.default_rng(0))
    for perturbation in result.perturbations:
        assert all(c.original.lower() != "he" for c in perturbation.changes)


def test_generate_outputs_are_distinct(toy_controls: SynonymControls):
    result = toy_controls.generate(_TOY_TEXT, n_words=2, k=5, rng=np.random.default_rng(2))
    texts = [p.text for p in result.perturbations]
    assert len(texts) == len(set(texts))
    assert all(t != _TOY_TEXT for t in texts)


def test_generate_is_deterministic_given_seed(toy_controls: SynonymControls):
    result1 = toy_controls.generate(_TOY_TEXT, n_words=2, k=4, rng=np.random.default_rng(7))
    result2 = toy_controls.generate(_TOY_TEXT, n_words=2, k=4, rng=np.random.default_rng(7))
    texts1 = [p.text for p in result1.perturbations]
    texts2 = [p.text for p in result2.perturbations]
    assert texts1 == texts2


def test_generate_respects_k_upper_bound(toy_controls: SynonymControls):
    result = toy_controls.generate(_TOY_TEXT, n_words=2, k=2, rng=np.random.default_rng(3))
    assert result.effective_k <= 2
    assert len(result.perturbations) == result.effective_k


def test_flags_empty_when_everything_is_matched(toy_controls: SynonymControls):
    result = toy_controls.generate(_TOY_TEXT, n_words=2, k=3, rng=np.random.default_rng(4))
    assert result.n_words_matched == 2
    assert result.effective_k == 3
    assert result.flags == ()


def test_flags_control_words_short_when_too_few_candidates():
    controls = SynonymControls(synonyms={"quick": ["fast"]}, protected=set())
    result = controls.generate("The quick fox.", n_words=3, k=5, rng=np.random.default_rng(0))
    assert result.n_words_matched == 1
    assert "control_words_short" in result.flags


def test_flags_controls_short_when_variant_pool_is_limited():
    # Exactly 3 candidates, each with exactly one synonym: only one possible
    # distinct output text exists, so effective_k must fall short of k.
    controls = SynonymControls(
        synonyms={"a": ["x"], "b": ["y"], "c": ["z"]}, protected=set()
    )
    result = controls.generate("a b c end.", n_words=3, k=5, rng=np.random.default_rng(0))
    assert result.n_words_matched == 3
    assert "control_words_short" not in result.flags
    assert "controls_short" in result.flags
    assert result.effective_k == 1


def test_control_set_effective_k_property():
    perturbations = (
        Perturbation(kind="control", text="a", changes=()),
        Perturbation(kind="control", text="b", changes=()),
    )
    control_set = ControlSet(
        perturbations=perturbations, requested_k=5, n_words_requested=2, n_words_matched=2
    )
    assert control_set.effective_k == 2
    assert set(control_set.flags) == {"controls_short"}


# ---------------------------------------------------------------------------
# Resource loading integration smoke test
# ---------------------------------------------------------------------------


def test_from_resources_smoke(swapper: GenderSwapper):
    controls = SynonymControls.from_resources(extra_protected=swapper.vocabulary)
    text = "David finally beat the boss."
    swap = swapper.swap(text)
    assert swap.n_changed >= 1
    result = controls.generate(text, n_words=swap.n_changed, k=3, rng=np.random.default_rng(0))
    assert result.effective_k >= 1
    for perturbation in result.perturbations:
        changed_words = {c.original.lower() for c in perturbation.changes}
        assert not changed_words & swapper.vocabulary


def test_group_counts_counts_each_groups_lexicon_words():
    swapper = GenderSwapper.from_resource()
    assert swapper.group_counts("He told his sister.") == {"male": 2, "female": 1}
    assert swapper.group_counts("Nobody here.") == {"male": 0, "female": 0}
