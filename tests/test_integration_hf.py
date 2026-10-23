"""Integration tests: the real Hugging Face code paths, with tiny models.

No network: every model here is a randomly initialised BERT or RoBERTa
with a hand-built vocabulary, saved to a temporary directory and loaded back
through `from_pretrained`, exactly as a Hub model would be. The outputs are
meaningless (random weights); what is tested is that tokenisation, mask
handling, whole-word filtering, pooling, label mapping and the end-to-end
run all work against the installed `transformers`.

Deselect with `pytest -m "not integration"`.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from biascompare import ControlsSpec, ModelSpec, compare_models  # noqa: E402
from biascompare.perturb import build_control_generator  # noqa: E402
from biascompare.perturb.mlm_controls import (  # noqa: E402
    HFMaskedLMSubstituter,
    HFSentenceSimilarity,
    _subword_scheme,
)
from biascompare.scoring.classifier import HFClassifierScorer  # noqa: E402
from biascompare.tasks import BUILTIN_TASKS  # noqa: E402

pytestmark = pytest.mark.integration

_WORDS = (
    "the a and was is he she his her my brother sister said texted match game phone camera "
    "comeback dramatic shouting great fun happy today review video helpful better new light "
    "manager meeting team credit work boss after right never seen that through"
).split()
_SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
_PIECES = ["##s", "##ed", "##ing", ",", ".", "!", "'"]


def _bert_config(**kwargs):
    return transformers.BertConfig(
        vocab_size=len(_SPECIAL) + len(_WORDS) + len(_PIECES),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=128,
        **kwargs,
    )


def _bert_tokenizer():
    vocab = {token: index for index, token in enumerate(_SPECIAL + _WORDS + _PIECES)}
    tokenizer = transformers.BertTokenizer(vocab=vocab)
    tokenizer.model_max_length = 128
    return tokenizer


@pytest.fixture(scope="module")
def tiny_bert_mlm(tmp_path_factory):
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("bert-mlm")
    transformers.BertForMaskedLM(_bert_config()).save_pretrained(path)
    _bert_tokenizer().save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_bert_encoder(tmp_path_factory):
    torch.manual_seed(1)
    path = tmp_path_factory.mktemp("bert-encoder")
    transformers.BertModel(_bert_config()).save_pretrained(path)
    _bert_tokenizer().save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_roberta_mlm(tmp_path_factory):
    from tokenizers import ByteLevelBPETokenizer

    path = tmp_path_factory.mktemp("roberta-mlm")
    bpe = ByteLevelBPETokenizer()
    corpus = [" ".join(_WORDS)] * 5 + ["my brother texted me after the match", "she said the camera was better"] * 5
    bpe.train_from_iterator(
        corpus, vocab_size=400, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )
    bpe.save_model(str(path))
    tokenizer = transformers.RobertaTokenizer(
        vocab=str(path / "vocab.json"), merges=str(path / "merges.txt")
    )
    tokenizer.model_max_length = 128
    tokenizer.save_pretrained(path)
    torch.manual_seed(2)
    config = transformers.RobertaConfig(
        vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=130,
    )
    transformers.RobertaForMaskedLM(config).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_three_label_classifier(tmp_path_factory):
    torch.manual_seed(3)
    path = tmp_path_factory.mktemp("bert-cls")
    config = _bert_config(
        num_labels=3,
        id2label={0: "normal", 1: "offensive", 2: "hate"},
        label2id={"normal": 0, "offensive": 1, "hate": 2},
    )
    transformers.BertForSequenceClassification(config).save_pretrained(path)
    _bert_tokenizer().save_pretrained(path)
    return str(path)


def test_wordpiece_substituter_proposes_whole_words_only(tiny_bert_mlm):
    substituter = HFMaskedLMSubstituter(tiny_bert_mlm, device="cpu")
    assert substituter._scheme == "wordpiece"
    text = "my brother texted after the match"
    spans = [(text.index("texted"), text.index("texted") + 6), (text.index("match"), len(text))]
    suggestions = substituter.substitutes(text, spans, top_k=5)
    assert len(suggestions) == 2
    for words in suggestions:
        assert words, "a random model still ranks every vocabulary token"
        for word, prob in words:
            assert word.isalpha() and not word.startswith("##") and word not in _SPECIAL
            assert 0.0 <= prob <= 1.0
    substituter.close()


def test_bpe_substituter_respects_word_boundaries(tiny_roberta_mlm):
    substituter = HFMaskedLMSubstituter(tiny_roberta_mlm, device="cpu")
    assert substituter._scheme == "bpe"
    text = "brother texted after the match"
    start = text.index("match")
    (words,) = substituter.substitutes(text, [(start, start + 5)], top_k=5)
    vocab = substituter._tokenizer.get_vocab()
    assert words
    for word, _ in words:
        assert word.isalpha()
        assert "Ġ" + word in vocab  # proposed as a word start after a space
    (first,) = substituter.substitutes(text, [(0, 7)], top_k=5)  # start of text: no leading space
    for word, _ in first:
        assert word in vocab
    substituter.close()


def test_sentence_similarity_is_cosine_of_mean_pooled_embeddings(tiny_bert_encoder):
    similarity = HFSentenceSimilarity(tiny_bert_encoder, device="cpu")
    text = "he said the game was great fun"
    scores = similarity.similarities(text, [text, "she said the match was helpful"])
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert -1.0 <= scores[1] <= 1.0
    assert similarity.similarities(text, []) == []
    similarity.close()


def test_subword_scheme_detection(tiny_bert_mlm, tiny_roberta_mlm):
    assert _subword_scheme(transformers.AutoTokenizer.from_pretrained(tiny_bert_mlm)) == "wordpiece"
    assert _subword_scheme(transformers.AutoTokenizer.from_pretrained(tiny_roberta_mlm)) == "bpe"


def test_classifier_with_three_labels_on_a_three_level_task(tiny_three_label_classifier):
    spec = ModelSpec(
        name=tiny_three_label_classifier,
        label_map={"non-toxic": ["normal"], "borderline": ["offensive"], "toxic": ["hate"]},
    )
    scorer = HFClassifierScorer(spec, task=BUILTIN_TASKS["toxicity-3"], device="cpu", batch_size=2)
    dists = scorer.score(["he said hello", "she said the game was fun", "the match"])
    assert len(dists) == 3
    for dist in dists:
        assert len(dist) == 3 and sum(dist) == pytest.approx(1.0)
    assert scorer.metadata["head"] == "softmax"
    scorer.close()


def test_end_to_end_run_with_masked_lm_controls(tmp_path, tiny_bert_mlm, tiny_bert_encoder, tiny_three_label_classifier):
    rows = [
        {"id": "a", "text": "my brother texted after the match", "attributes": {"gender": "male"}},
        {"id": "b", "text": "she said the camera review was helpful", "attributes": {"gender": "female"},
         "label": "non-toxic"},
        {"id": "c", "text": "he never seen that comeback", "attributes": {"gender": "male"}},
        {"id": "d", "text": "her manager took credit for the work", "attributes": {"gender": "female"}},
    ]
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    controls = ControlsSpec(
        method="masked-lm", mlm_model=tiny_bert_mlm, similarity_model=tiny_bert_encoder,
        min_similarity=0.0, device="cpu",
    )
    label_map = {"non-toxic": ["normal"], "borderline": ["offensive"], "toxic": ["hate"]}
    models = [
        ModelSpec(name=tiny_three_label_classifier, label_map=label_map),
        ModelSpec(name=tiny_three_label_classifier, label_map=label_map, settings={"batch_size": 1}),
    ]

    results = compare_models(models, dataset, task="toxicity-3", n_controls=3, seed=0, device="cpu", controls=controls)

    assert results["run"]["controls"]["method"] == "masked-lm"
    for post in results["posts"]:
        assert "no_controls" not in post["flags"] and post["controls"]
        for control in post["controls"]:
            assert control["similarity"] is not None
            assert len(control["changes"]) == post["n_words_changed"] or "control_words_short" in post["flags"]
    for model in results["models"]:
        assert model["error"] is None, model["error"]
        assert model["summary"]["n_labeled"] == 1
    # identical model and inputs: no difference between the two
    diff = results["comparisons"][0]["delta_diff"]["mean"]
    assert diff == pytest.approx(0.0, abs=1e-6)


def test_control_generator_is_deterministic_with_real_models(tiny_bert_mlm, tiny_bert_encoder):
    spec = ControlsSpec(
        method="masked-lm", mlm_model=tiny_bert_mlm, similarity_model=tiny_bert_encoder,
        min_similarity=0.0, device="cpu",
    )
    text = "my brother texted after the match"
    runs = []
    for _ in range(2):
        generator = build_control_generator(spec, extra_protected={"brother", "my"})
        runs.append([p.text for p in generator.generate(text, 2, 4, np.random.default_rng(7)).perturbations])
        generator.close()
    assert runs[0] == runs[1] and runs[0]
