"""Tests for scoring.causal_lm.HFCausalLMScorer on tiny causal LMs built locally.

GPT-2 (absolute positions) and Llama (rotary positions) models with a few
thousand parameters and a byte-level BPE vocabulary trained on a toy corpus
are saved to temporary folders and loaded through `from_pretrained`. Their
outputs are random; the tests check rendering, answer tokenisation, that
left padding with position ids makes a text's scores independent of its
batch, both scoring modes, option mass, and an end-to-end run.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from biascompare import Example, ModelSpec, PromptSpec, compare_models  # noqa: E402
from biascompare.scoring.causal_lm import HFCausalLMScorer  # noqa: E402
from biascompare.tasks import BUILTIN_TASKS, DEFAULT_TASK  # noqa: E402

pytestmark = pytest.mark.integration

_CORPUS = [
    "is the following social media post toxic answer with exactly one of non-toxic toxic yes no",
    "post he said the game was great fun she said the match was helpful answer",
    "my brother texted after the match her review video was helpful his hands were shaking",
    "user assistant system borderline mild severe",
] * 10
_CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)


def _tokenizer(path, chat: bool):
    from tokenizers import ByteLevelBPETokenizer

    bpe = ByteLevelBPETokenizer()
    bpe.train_from_iterator(_CORPUS, vocab_size=320, min_frequency=1, special_tokens=["<|endoftext|>"])
    bpe.save_model(str(path))
    tokenizer = transformers.GPT2Tokenizer(vocab=str(path / "vocab.json"), merges=str(path / "merges.txt"))
    tokenizer.model_max_length = 512
    if chat:
        tokenizer.chat_template = _CHAT_TEMPLATE
    tokenizer.save_pretrained(path)
    return tokenizer


@pytest.fixture(scope="module")
def tiny_gpt2(tmp_path_factory):
    path = tmp_path_factory.mktemp("gpt2")
    tokenizer = _tokenizer(path, chat=False)
    torch.manual_seed(0)
    config = transformers.GPT2Config(
        vocab_size=len(tokenizer), n_embd=32, n_layer=2, n_head=2, n_positions=512,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    transformers.GPT2LMHeadModel(config).save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_llama_chat(tmp_path_factory):
    path = tmp_path_factory.mktemp("llama")
    tokenizer = _tokenizer(path, chat=True)
    torch.manual_seed(1)
    config = transformers.LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


def _scorer(path, **settings):
    prompt = settings.pop("prompt", None)
    task = settings.pop("task", DEFAULT_TASK)
    return HFCausalLMScorer(ModelSpec(path, type="hf-causal-lm", settings=settings, prompt=prompt), task, "cpu")


def test_plain_prompt_ends_where_the_answer_starts(tiny_gpt2):
    scorer = _scorer(tiny_gpt2)
    rendered = scorer.render("he said hi")
    assert rendered.startswith("Is the following social media post toxic?")
    assert "Post: he said hi" in rendered
    assert rendered.endswith("Answer with exactly one of: non-toxic, toxic.\nAnswer:")
    assert scorer.metadata["chat_template"] is False
    assert scorer.metadata["answers"] == ["non-toxic", "toxic"]
    scorer.close()


def test_chat_template_wraps_the_prompt(tiny_llama_chat):
    prompt = PromptSpec(system="You are a content moderator.", options={"non-toxic": "No", "toxic": "Yes"},
                        instruction="Is this post toxic?")
    scorer = _scorer(tiny_llama_chat, prompt=prompt)
    rendered = scorer.render("she said hi")
    assert rendered.startswith("<|system|>\nYou are a content moderator.\n<|user|>\nIs this post toxic?")
    assert rendered.endswith("Answer with exactly one of: No, Yes.\n<|assistant|>\n")
    assert scorer.metadata["chat_template"] is True
    scorer.close()


@pytest.mark.parametrize("model_fixture", ["tiny_gpt2", "tiny_llama_chat"])
def test_scores_do_not_depend_on_batch_composition(model_fixture, request):
    path = request.getfixturevalue(model_fixture)
    scorer = _scorer(path, batch_size=4)
    texts = ["he said hi", "my brother texted after the match and his hands were shaking the whole time", "ok"]
    together = scorer.score(texts)
    masses = scorer.diagnostics["option_mass"]
    alone = [scorer.score([text])[0] for text in texts]
    for a, b in zip(together, alone):
        assert a == pytest.approx(b, abs=1e-5)
    for dist in together:
        assert len(dist) == 2 and sum(dist) == pytest.approx(1.0)
    assert len(masses) == 3 and all(0.0 < m <= 1.0 for m in masses)
    scorer.close()


def test_options_sharing_a_first_token_need_sequence_scoring(tiny_gpt2):
    shared = PromptSpec(options={"non-toxic": "toxic no", "toxic": "toxic yes"})
    with pytest.raises(ValueError, match="start with the same token"):
        _scorer(tiny_gpt2, prompt=shared)
    scorer = _scorer(tiny_gpt2, prompt=shared, scoring="sequence")
    dists = scorer.score(["he said hi", "she said hi"])
    assert all(len(d) == 2 and sum(d) == pytest.approx(1.0) for d in dists)
    assert all(0.0 < m <= 1.0 for m in scorer.diagnostics["option_mass"])
    scorer.close()


def test_four_levels_and_few_shot_examples(tiny_gpt2):
    prompt = PromptSpec(
        instruction="How toxic is this post?",
        examples=(Example("you are great", "non-toxic"), Example("you idiot", "severe")),
        options={"non-toxic": "no", "mild": "mild", "toxic": "toxic", "severe": "severe"},
    )
    scorer = _scorer(tiny_gpt2, prompt=prompt, task=BUILTIN_TASKS["toxicity-4"])
    rendered = scorer.render("he said hi")
    assert "Post: you idiot\nAnswer: severe" in rendered
    assert rendered.endswith("one of: no, mild, toxic, severe.\nAnswer:")
    (dist,) = scorer.score(["he said hi"])
    assert len(dist) == 4 and sum(dist) == pytest.approx(1.0)
    scorer.close()


def test_unknown_settings_and_choices_are_rejected(tiny_gpt2):
    with pytest.raises(ValueError, match="Unknown setting"):
        _scorer(tiny_gpt2, temperature=0.0)
    with pytest.raises(ValueError, match="scoring must be one of"):
        _scorer(tiny_gpt2, scoring="greedy")


def test_end_to_end_classifier_style_run_with_llms(tmp_path, tiny_gpt2, tiny_llama_chat):
    rows = [
        {"id": "a", "text": "my brother texted after the match", "attributes": {"gender": "male"},
         "synonyms": {"texted": ["messaged"], "match": ["game"]}},
        {"id": "b", "text": "she said the match was helpful", "attributes": {"gender": "female"},
         "synonyms": {"said": ["told"], "match": ["game"], "helpful": ["useful"]}},
    ]
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    models = [
        ModelSpec(tiny_gpt2, type="hf-causal-lm", id="gpt2"),
        ModelSpec(tiny_llama_chat, type="hf-causal-lm", id="llama",
                  prompt=PromptSpec(options={"non-toxic": "No", "toxic": "Yes"})),
    ]
    results = compare_models(models, dataset, n_controls=2, device="cpu", controls="tagged")
    for entry in results["models"]:
        assert entry["error"] is None, entry["error"]
        assert 0.0 < entry["summary"]["mean_option_mass"] <= 1.0
    run_models = {entry["id"]: entry for entry in results["run"]["models"]}
    assert run_models["llama"]["answers"] == ["No", "Yes"] and run_models["llama"]["scorer"] == "hf-causal-lm"
    assert any("not answering in the requested format" in w for w in results["warnings"])  # random weights
    assert len(results["comparisons"]) == 1


_QWEN3_STYLE_TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)


@pytest.fixture(scope="module")
def tiny_hybrid_reasoner(tmp_path_factory):
    """A tiny Llama with a Qwen3-style template that opens a thinking block
    unless `enable_thinking=False` is passed."""
    path = tmp_path_factory.mktemp("qwen3-style")
    tokenizer = _tokenizer(path, chat=True)
    tokenizer.chat_template = _QWEN3_STYLE_TEMPLATE
    tokenizer.save_pretrained(path)
    torch.manual_seed(2)
    config = transformers.LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id,
    )
    transformers.LlamaForCausalLM(config).save_pretrained(path)
    return str(path)


def test_thinking_is_switched_off_by_default_for_hybrid_reasoners(tiny_hybrid_reasoner):
    scorer = _scorer(tiny_hybrid_reasoner, prompt=PromptSpec(options={"non-toxic": "No", "toxic": "Yes"}))
    rendered = scorer.render("he said hi")
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert scorer.metadata["enable_thinking"] is False
    assert scorer.metadata["device"] == "cpu"
    (dist,) = scorer.score(["he said hi"])
    assert sum(dist) == pytest.approx(1.0)
    scorer.close()

    thinking = _scorer(tiny_hybrid_reasoner, enable_thinking=True)
    assert thinking.render("he said hi").endswith("<|im_start|>assistant\n")
    thinking.close()


def test_preview_llm_prompt_reports_an_unreachable_model(monkeypatch):
    from biascompare.scoring import preview_llm_prompt

    def _raise(*args, **kwargs):
        raise OSError("org/missing is not a local folder and is not a valid model identifier")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", staticmethod(_raise))
    spec = ModelSpec("org/missing", type="hf-causal-lm")
    with pytest.raises(RuntimeError, match="Failed to load tokenizer of 'org/missing': 'org/missing' is neither"):
        preview_llm_prompt(spec, "he said hi")


def test_preview_llm_prompt_matches_the_scorer_without_loading_the_model(tiny_hybrid_reasoner):
    from biascompare.scoring import preview_llm_prompt

    prompt = PromptSpec(options={"non-toxic": "No", "toxic": "Yes"})
    spec = ModelSpec(tiny_hybrid_reasoner, type="hf-causal-lm", id="qwen-like", prompt=prompt)
    preview = preview_llm_prompt(spec, "he said hi")
    scorer = HFCausalLMScorer(spec, DEFAULT_TASK, "cpu")
    assert preview["prompt"] == scorer.render("he said hi")
    assert preview["chat_template"] is True and preview["model"] == "qwen-like"
    assert list(preview["answer_tokens"]) == ["No", "Yes"]
    assert [tokens[0] for tokens in preview["answer_tokens"].values()] == [
        scorer._tokenizer.convert_ids_to_tokens(ids)[0] for ids in scorer._option_ids
    ]
    scorer.close()
