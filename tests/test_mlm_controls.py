"""Tests for perturb.mlm_controls.MaskedLMControls with fake helper models,
plus ControlsSpec and compare_models with a custom control generator.

Fully offline: `FakeSubstituter` proposes fixed words per original word and
`FakeSimilarity` scores variants by a rule, so sampling, filtering and the
similarity threshold are tested without any model. The real HF helpers are
covered in test_integration_hf.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from biascompare import ModelSpec, compare_models, preview_perturbations
from biascompare.axes import load_swapper
from biascompare.perturb import ControlsSpec, MaskedLMControls
from biascompare.perturb.mlm_controls import _whole_word
from test_compare import BIASED_VS_UNBIASED, DATASET, _fake_factory

_SUGGESTIONS = {
    "match": ["game", "Fixture", "she", "the", "stupid"],
    "dramatic": ["wild", "dramatic", "x", "intense!"],
    "phone": ["mobile", "cell"],
    "texted": ["messaged", "called"],
    "comeback": ["rally"],
}


class FakeSubstituter:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.closed = False

    def substitutes(self, text, spans, top_k):
        words = [text[start:end] for start, end in spans]
        self.calls.append(words)
        return [[(w, 0.1) for w in _SUGGESTIONS.get(word.lower(), ["thing"])] for word in words]

    def close(self) -> None:
        self.closed = True


class FakeSimilarity:
    """Similarity 0.95, except 0.5 for any variant containing 'cell'."""

    def __init__(self) -> None:
        self.closed = False

    def similarities(self, original, variants):
        return [0.5 if "cell" in variant else 0.95 for variant in variants]

    def close(self) -> None:
        self.closed = True


def _controls(**kwargs) -> MaskedLMControls:
    swapper = load_swapper("gender")
    return MaskedLMControls.from_resources(
        FakeSubstituter(), FakeSimilarity(), extra_protected=swapper.vocabulary, **kwargs
    )


TEXT = "My brother texted me after the match and said the comeback was dramatic on the phone."


def test_candidates_skip_protected_stopwords_and_short_words():
    words = [m.group(0) for m in _controls().candidates(TEXT)]
    assert "brother" not in words  # gender lexicon
    assert "the" not in words and "me" not in words  # stopwords
    assert {"texted", "match", "comeback", "dramatic", "phone", "said"} <= set(words)


def test_listed_synonyms_override_the_masked_lm():
    controls = _controls()
    controls.generate(TEXT, n_words=2, k=3, rng=np.random.default_rng(0))
    asked = {word.lower() for call in controls._substituter.calls for word in call}
    assert "dramatic" not in asked and "match" not in asked  # both are in synonyms.json
    assert "phone" in asked and "texted" in asked


def test_masked_lm_suggestions_are_filtered():
    controls = _controls(use_synonym_list=False)
    matches = controls.candidates(TEXT)
    options = dict(zip([m.group(0) for m in matches], controls._options(TEXT, matches)))
    # "she" (gender), "the" (stopword), "stupid" (protected) dropped; case folded
    assert options["match"] == ("game", "fixture")
    # the original word, one-letter and non-alphabetic suggestions dropped
    assert options["dramatic"] == ("wild",)


def test_each_control_changes_exactly_n_words_and_preserves_case():
    controls = _controls()
    result = controls.generate("Phone calls ruin the Match.", n_words=2, k=4, rng=np.random.default_rng(1))
    assert result.effective_k >= 1
    for control in result.perturbations:
        assert len(control.changes) == 2
        assert control.similarity == pytest.approx(0.95)
        for change in control.changes:
            assert change.original[0].isupper() == change.replacement[0].isupper()


def test_similarity_threshold_drops_variants():
    controls = _controls(use_synonym_list=False)
    result = controls.generate("Check the phone.", n_words=1, k=5, rng=np.random.default_rng(0))
    texts = [p.text for p in result.perturbations]
    assert "Check the mobile." in texts
    assert all("cell" not in text for text in texts)


def test_generation_is_deterministic_given_the_seed():
    a = _controls().generate(TEXT, n_words=2, k=5, rng=np.random.default_rng([0, 3]))
    b = _controls().generate(TEXT, n_words=2, k=5, rng=np.random.default_rng([0, 3]))
    assert [p.text for p in a.perturbations] == [p.text for p in b.perturbations]
    assert len({p.text for p in a.perturbations}) == a.effective_k  # distinct


def test_no_usable_words_gives_an_empty_control_set():
    result = _controls().generate("He is so into it.", n_words=1, k=5, rng=np.random.default_rng(0))
    assert result.effective_k == 0
    assert result.n_words_matched == 0


def test_whole_word_rules_per_tokenizer_scheme():
    assert _whole_word("game", "wordpiece", True) == "game"
    assert _whole_word("##ing", "wordpiece", True) is None
    assert _whole_word("Ġgame", "bpe", True) == "game"
    assert _whole_word("game", "bpe", True) is None  # a word piece, not a word start
    assert _whole_word("Game", "bpe", False) == "Game"
    assert _whole_word("Ġgame", "bpe", False) is None
    assert _whole_word("▁game", "sentencepiece", True) == "game"
    assert _whole_word("Ġ123", "bpe", True) is None


def test_controls_spec_validation_and_public_dict():
    assert ControlsSpec().to_public_dict() == {"method": "synonyms"}
    public = ControlsSpec(method="masked-lm", min_similarity=0.9).to_public_dict()
    assert public["method"] == "masked-lm" and public["min_similarity"] == 0.9
    assert "device" not in public
    with pytest.raises(ValueError, match="method"):
        ControlsSpec(method="paraphrase")
    with pytest.raises(ValueError, match="min_similarity"):
        ControlsSpec(min_similarity=1.5)


def test_compare_models_with_a_custom_control_generator():
    controls = _controls()
    results = compare_models(
        BIASED_VS_UNBIASED, DATASET, seed=0, scorer_factory=_fake_factory, controls=controls
    )
    assert results["run"]["controls"] == {"method": "custom", "generator": "MaskedLMControls"}
    assert all(post["effective_k"] == 5 for post in results["posts"])
    similarities = [c["similarity"] for post in results["posts"] for c in post["controls"]]
    assert similarities and all(s == pytest.approx(0.95) for s in similarities)
    assert results["models"][0]["summary"]["mean_delta"] > 0
    assert not controls._substituter.closed  # a generator passed in is the caller's to close


def test_masked_lm_controls_cover_the_no_controls_post(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"id": "z", "text": "He is happy today.", "attributes": {"gender": "male"}}\n')
    synonym_preview = preview_perturbations(path)
    assert "no_controls" in synonym_preview[0]["flags"]
    mlm_preview = preview_perturbations(path, controls=_controls())
    assert "no_controls" not in mlm_preview[0]["flags"]
    assert mlm_preview[0]["controls"][0]["text"] != "He is happy today."


def test_controls_spec_follows_the_run_device():
    from biascompare.compare import _resolve_controls

    assert _resolve_controls("masked-lm", "cpu").device == "cpu"
    assert _resolve_controls(ControlsSpec(method="masked-lm", device="cuda"), "cpu").device == "cuda"
    assert _resolve_controls(None).device == "auto"
