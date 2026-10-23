"""Tests for biascompare.scoring: base helpers and HFClassifierScorer.

HFClassifierScorer is tested with monkeypatched `transformers.AutoTokenizer`
and `AutoModelForSequenceClassification` `from_pretrained` classmethods, so
no real model is ever downloaded or loaded. Real `torch` tensors are used
for the fake model's inputs/outputs (torch is a project dependency), only
`from_pretrained` (the network-touching part) is faked.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from biascompare.inputs import ModelSpec
from biascompare.scoring.base import binary_from_logits, resolve_label_index

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from biascompare.scoring.classifier import HFClassifierScorer  # noqa: E402


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(xs: list[float]) -> list[float]:
    exp_vals = [math.exp(x) for x in xs]
    total = sum(exp_vals)
    return [v / total for v in exp_vals]


# ---------------------------------------------------------------------------
# resolve_label_index
# ---------------------------------------------------------------------------


def test_resolve_label_index_with_int_keys():
    id2label = {0: "toxic", 1: "severe_toxic"}
    assert resolve_label_index(id2label, "toxic") == 0
    assert resolve_label_index(id2label, "severe_toxic") == 1


def test_resolve_label_index_with_str_keys():
    id2label = {"0": "neutral", "1": "toxic"}
    assert resolve_label_index(id2label, "toxic") == 1


def test_resolve_label_index_is_case_insensitive():
    id2label = {0: "Toxic"}
    assert resolve_label_index(id2label, "TOXIC") == 0
    assert resolve_label_index(id2label, "toxic") == 0


def test_resolve_label_index_missing_label_lists_available_labels():
    id2label = {0: "neutral", 1: "toxic"}
    with pytest.raises(ValueError) as exc_info:
        resolve_label_index(id2label, "offensive")
    message = str(exc_info.value)
    assert "offensive" in message
    assert "neutral" in message
    assert "toxic" in message


# ---------------------------------------------------------------------------
# binary_from_logits
# ---------------------------------------------------------------------------


def test_binary_from_logits_multi_label_uses_sigmoid():
    p_non_toxic, p_toxic = binary_from_logits([2.0, -1.0], toxic_index=0, multi_label=True)
    expected = _sigmoid(2.0)
    assert p_toxic == pytest.approx(expected, abs=1e-9)
    assert p_non_toxic == pytest.approx(1.0 - expected, abs=1e-9)


def test_binary_from_logits_multi_label_ignores_other_heads():
    # Other logits (e.g. severe_toxic, insult) must not affect the result.
    p_a = binary_from_logits([2.0, -1.0, 5.0], toxic_index=0, multi_label=True)
    p_b = binary_from_logits([2.0, 99.0, -99.0], toxic_index=0, multi_label=True)
    assert p_a[1] == pytest.approx(p_b[1], abs=1e-9)


def test_binary_from_logits_single_label_uses_softmax():
    logits = [1.0, 2.0, 0.5]
    toxic_index = 1
    p_non_toxic, p_toxic = binary_from_logits(logits, toxic_index=toxic_index, multi_label=False)
    expected = _softmax(logits)[toxic_index]
    assert p_toxic == pytest.approx(expected, abs=1e-9)
    assert p_non_toxic == pytest.approx(1.0 - expected, abs=1e-9)


def test_binary_from_logits_out_of_range_index_raises():
    with pytest.raises(ValueError):
        binary_from_logits([1.0, 2.0], toxic_index=5, multi_label=False)


# ---------------------------------------------------------------------------
# HFClassifierScorer (monkeypatched load)
# ---------------------------------------------------------------------------


class _FakeConfig:
    def __init__(self, id2label, problem_type):
        self.id2label = id2label
        self.problem_type = problem_type


class _FakeBatchEncoding(dict):
    def to(self, device):  # noqa: ARG002 - fake is device-agnostic
        return self


class _FakeTokenizer:
    model_max_length = 64

    def __call__(self, texts, padding=True, truncation=True, max_length=None, return_tensors="pt"):
        n = len(texts)
        return _FakeBatchEncoding(
            {
                "input_ids": torch.zeros((n, 4), dtype=torch.long),
                "attention_mask": torch.ones((n, 4), dtype=torch.long),
            }
        )


class _FakeModel:
    """Returns pre-baked logits rows in order, tracking position across batches."""

    def __init__(self, config, logits_rows):
        self.config = config
        self._logits_rows = logits_rows
        self._offset = 0

    def to(self, device):  # noqa: ARG002
        return self

    def eval(self):
        return self

    def __call__(self, **encoded):
        batch_size = encoded["input_ids"].shape[0]
        rows = self._logits_rows[self._offset : self._offset + batch_size]
        self._offset += batch_size
        return SimpleNamespace(logits=torch.tensor(rows, dtype=torch.float32))


def test_hf_classifier_scorer_multi_label_batches_and_maps_labels(monkeypatch):
    id2label = {0: "toxic", 1: "severe_toxic"}
    config = _FakeConfig(id2label=id2label, problem_type="multi_label_classification")
    logits_rows = [
        [2.0, -1.0],
        [-3.0, 5.0],
        [0.0, 0.0],
        [1.0, 1.0],
        [-5.0, -5.0],
    ]
    fake_model = _FakeModel(config, logits_rows)
    fake_tokenizer = _FakeTokenizer()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: fake_tokenizer)
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(lambda *a, **k: fake_model),
    )

    spec = ModelSpec(name="fake/model", toxic_label="toxic", token="super-secret-token")
    scorer = HFClassifierScorer(spec, device="cpu", batch_size=2)

    assert scorer.name == "fake/model"
    assert scorer.scorer_type == "hf-classifier"
    assert scorer.uses_instruction is False

    texts = ["a", "b", "c", "d", "e"]
    results = scorer.score(texts)

    assert len(results) == 5
    for (p_non_toxic, p_toxic), row in zip(results, logits_rows):
        expected = _sigmoid(row[0])
        assert p_toxic == pytest.approx(expected, abs=1e-5)
        assert p_non_toxic == pytest.approx(1.0 - expected, abs=1e-5)

    # Token must never leak through the spec's repr or public serialisation.
    assert "super-secret-token" not in repr(spec)
    assert "super-secret-token" not in str(spec.to_public_dict())

    scorer.close()


def test_hf_classifier_scorer_single_label_uses_softmax(monkeypatch):
    id2label = {0: "neutral", 1: "toxic"}
    config = _FakeConfig(id2label=id2label, problem_type="single_label_classification")
    logits_rows = [[0.2, 1.8], [3.0, -1.0]]
    fake_model = _FakeModel(config, logits_rows)
    fake_tokenizer = _FakeTokenizer()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: fake_tokenizer)
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(lambda *a, **k: fake_model),
    )

    spec = ModelSpec(name="fake/other-model", toxic_label="toxic")
    scorer = HFClassifierScorer(spec, device="cpu", batch_size=16)

    results = scorer.score(["x", "y"])
    for (p_non_toxic, p_toxic), row in zip(results, logits_rows):
        expected = _softmax(row)[1]
        assert p_toxic == pytest.approx(expected, abs=1e-5)
        assert p_non_toxic == pytest.approx(1.0 - expected, abs=1e-5)

    scorer.close()


def test_hf_classifier_scorer_wraps_load_failure_in_runtime_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise OSError("model not found")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(_raise))

    spec = ModelSpec(name="does/not-exist")
    with pytest.raises(RuntimeError, match="does/not-exist"):
        HFClassifierScorer(spec, device="cpu")


def test_hf_classifier_scorer_unknown_toxic_label_raises_value_error(monkeypatch):
    id2label = {0: "neutral", 1: "toxic"}
    config = _FakeConfig(id2label=id2label, problem_type="single_label_classification")
    fake_model = _FakeModel(config, [[0.0, 0.0]])
    fake_tokenizer = _FakeTokenizer()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", staticmethod(lambda *a, **k: fake_tokenizer)
    )
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification,
        "from_pretrained",
        staticmethod(lambda *a, **k: fake_model),
    )

    spec = ModelSpec(name="fake/model", toxic_label="offensive")
    with pytest.raises(ValueError, match="offensive"):
        HFClassifierScorer(spec, device="cpu")


# ---------------------------------------------------------------------------
# HFClassifierScorer with task levels and label_map
# ---------------------------------------------------------------------------


def _patch_loaders(monkeypatch, fake_model, calls=None):
    def _tokenizer(*args, **kwargs):
        if calls is not None:
            calls.append(kwargs)
        return _FakeTokenizer()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(_tokenizer))
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", staticmethod(lambda *a, **k: fake_model)
    )


def test_hf_classifier_scorer_maps_three_labels_onto_three_levels(monkeypatch):
    from biascompare.tasks import BUILTIN_TASKS

    config = _FakeConfig(id2label={0: "normal", 1: "offensive", 2: "hate"}, problem_type=None)
    logits_rows = [[1.0, 0.0, -1.0], [-2.0, 0.5, 2.0]]
    _patch_loaders(monkeypatch, _FakeModel(config, logits_rows))

    spec = ModelSpec(
        name="fake/hatexplain",
        label_map={"non-toxic": ["normal"], "borderline": ["offensive"], "toxic": ["hate"]},
    )
    scorer = HFClassifierScorer(spec, task=BUILTIN_TASKS["toxicity-3"], device="cpu")
    for dist, row in zip(scorer.score(["x", "y"]), logits_rows):
        assert dist == pytest.approx(tuple(_softmax(row)), abs=1e-6)
    assert scorer.metadata["label_mapping"] == {"non-toxic": ["normal"], "borderline": ["offensive"], "toxic": ["hate"]}
    assert scorer.metadata["head"] == "softmax"
    scorer.close()


def test_hf_classifier_scorer_passes_revision_and_token_env(monkeypatch):
    monkeypatch.setenv("MY_HF_TOKEN", "from-env")
    calls: list[dict] = []
    config = _FakeConfig(id2label={0: "neutral", 1: "toxic"}, problem_type=None)
    _patch_loaders(monkeypatch, _FakeModel(config, [[0.0, 0.0]]), calls)

    spec = ModelSpec(name="fake/m", revision="abc123", token_env="MY_HF_TOKEN")
    scorer = HFClassifierScorer(spec, device="cpu")
    assert calls == [{"token": "from-env", "revision": "abc123"}]
    assert scorer.metadata["revision"] == "abc123"
    scorer.close()


def test_hf_classifier_scorer_rejects_unknown_settings(monkeypatch):
    spec = ModelSpec(name="fake/m", settings={"temperature": 0.5})
    with pytest.raises(ValueError, match="temperature"):
        HFClassifierScorer(spec, device="cpu")


def test_hf_classifier_scorer_batch_size_setting(monkeypatch):
    config = _FakeConfig(id2label={0: "neutral", 1: "toxic"}, problem_type=None)
    _patch_loaders(monkeypatch, _FakeModel(config, [[0.0, 1.0]] * 5))
    scorer = HFClassifierScorer(ModelSpec(name="fake/m", settings={"batch_size": 2}), device="cpu")
    assert scorer._batch_size == 2
    assert len(scorer.score(["a", "b", "c", "d", "e"])) == 5
    scorer.close()
