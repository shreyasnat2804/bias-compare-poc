"""Hugging Face causal LM scorer (`type: hf-causal-lm`): open-weight LLMs run
locally (on Colab), scored by their probabilities over the answer options.
See docs/PLAN.md §4.3.

Each post is rendered with the model's prompt (instruction, few-shot
examples, `{text}`), followed by the list of answer options, and wrapped in
the model's chat template when it has one. The distribution over the task's
levels is the model's next-token probability on each option's first token,
renormalised over the options (`scoring: first-token`, the default), or the
probability of each full option string (`scoring: sequence`, for options
that share a first token). The probability mass the model put on the
options before renormalising is recorded per text as `option_mass`: a low
value means the model was not answering in the requested format.

Batches are left-padded with explicit position ids, so a text's scores do
not depend on what else is in its batch. `torch`/`transformers` are
imported lazily.
"""

from __future__ import annotations

from collections.abc import Sequence

from biascompare.devices import free_accelerator_cache, pick_device
from biascompare.inputs import ModelSpec
from biascompare.metrics import Distribution
from biascompare.prompts import PromptSpec
from biascompare.tasks import DEFAULT_TASK, TaskSpec

_DEFAULTS = {
    "batch_size": 8,
    "max_length": 2048,
    "dtype": "auto",
    "quantization": "none",
    "use_chat_template": True,
    "scoring": "first-token",
    "answer_prefix": "",
}
_DTYPES = ("auto", "float32", "float16", "bfloat16")
_QUANTIZATION = ("none", "8bit", "4bit")
_SCORING = ("first-token", "sequence")


class HFCausalLMScorer:
    """Scorer backed by `AutoModelForCausalLM`.

    Settings (all optional, on `ModelSpec.settings`): `batch_size` (8),
    `max_length` (2048 tokens; longer prompts are cut from the left),
    `dtype` (`auto`, `float32`, `float16`, `bfloat16`), `quantization`
    (`none`, `8bit`, `4bit`; needs `bitsandbytes` and a GPU),
    `use_chat_template` (true; ignored if the tokenizer has none),
    `scoring` (`first-token` or `sequence`), `answer_prefix` (text placed
    after the assistant turn starts, before the answer, default empty).
    """

    scorer_type = "hf-causal-lm"
    uses_instruction = True

    def __init__(self, spec: ModelSpec, task: TaskSpec | None = None, device: str = "auto") -> None:
        settings = {**_DEFAULTS, **dict(spec.settings)}
        unknown = sorted(set(spec.settings) - set(_DEFAULTS))
        if unknown:
            raise ValueError(
                f"Unknown setting(s) {unknown} for hf-causal-lm {spec.name!r}; allowed: {sorted(_DEFAULTS)}"
            )
        _check_choice(spec.name, "dtype", settings["dtype"], _DTYPES)
        _check_choice(spec.name, "quantization", settings["quantization"], _QUANTIZATION)
        _check_choice(spec.name, "scoring", settings["scoring"], _SCORING)

        self.name = spec.name
        self._task = task or DEFAULT_TASK
        self._prompt = spec.prompt or PromptSpec(name="default")
        self._prompt.check_task(self._task)
        self._answers = self._prompt.answers(self._task)
        self._batch_size = int(settings["batch_size"])
        self._max_length = int(settings["max_length"])
        self._scoring = settings["scoring"]
        self._answer_prefix = str(settings["answer_prefix"])
        self.diagnostics: dict[str, list] = {}

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("HFCausalLMScorer requires the `torch` and `transformers` packages") from exc
        self._torch = torch

        token = spec.resolved_token()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(spec.name, token=token, revision=spec.revision)
            model_kwargs = _model_kwargs(torch, settings, device)
            self._model = AutoModelForCausalLM.from_pretrained(
                spec.name, token=token, revision=spec.revision, **model_kwargs
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model {spec.name!r}. If this is a gated model, accept its licence on the "
                "Hugging Face Hub and pass a token (`token_env` or HF_TOKEN). Original error: "
                f"{exc}"
            ) from exc

        quantized = settings["quantization"] != "none"
        self._device = pick_device(device) if not quantized else "cuda"
        if not quantized:
            self._model.to(self._device)
        self._model.eval()
        self._input_device = next(self._model.parameters()).device

        tokenizer = self._tokenizer
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        self._chat = bool(settings["use_chat_template"]) and bool(getattr(tokenizer, "chat_template", None))
        self._add_special_tokens = not self._chat

        sample = self.render("example post")
        self._option_ids = [self._continuation_ids(sample, answer) for answer in self._answers]
        first_tokens = [ids[0] for ids in self._option_ids]
        if self._scoring == "first-token" and len(set(first_tokens)) != len(first_tokens):
            raise ValueError(
                f"The answer options {self._answers} of {spec.name!r} start with the same token, so "
                "first-token scoring cannot tell them apart. Use options that differ in their first "
                "word (e.g. Yes/No, A/B/C) or set `scoring: sequence`."
            )
        self.metadata = {
            "revision": getattr(self._model.config, "_commit_hash", None) or spec.revision,
            "answers": list(self._answers),
            "answer_tokens": [tokenizer.convert_ids_to_tokens(ids) for ids in self._option_ids],
            "chat_template": self._chat,
            "scoring": self._scoring,
            "dtype": str(getattr(self._model, "dtype", settings["dtype"])),
            "quantization": settings["quantization"],
            "max_length": self._max_length,
        }

    def render(self, text: str) -> str:
        """The exact string the model reads for `text`, ending where the answer starts."""
        user = (
            self._prompt.render(text, self._task)
            + "\n\nAnswer with exactly one of: "
            + ", ".join(self._answers)
            + "."
        )
        if self._chat:
            messages = [{"role": "user", "content": user}]
            if self._prompt.system:
                messages.insert(0, {"role": "system", "content": self._prompt.system})
            rendered = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            return rendered + self._answer_prefix
        system = f"{self._prompt.system}\n\n" if self._prompt.system else ""
        return f"{system}{user}\nAnswer:{self._answer_prefix}"

    def score(self, texts: Sequence[str]) -> list[Distribution]:
        results: list[Distribution] = []
        masses: list[float] = []
        truncated: list[bool] = []
        for start in range(0, len(texts), self._batch_size):
            prompts = [self.render(text) for text in texts[start : start + self._batch_size]]
            lengths = self._tokenizer(prompts, add_special_tokens=self._add_special_tokens)["input_ids"]
            truncated.extend(len(ids) > self._max_length for ids in lengths)
            if self._scoring == "first-token":
                batch, batch_mass = self._score_first_token(prompts)
            else:
                batch, batch_mass = self._score_sequences(prompts)
            results.extend(batch)
            masses.extend(batch_mass)
        self.diagnostics = {"option_mass": masses, "truncated": truncated}
        return results

    def close(self) -> None:
        if hasattr(self, "_model"):
            del self._model
        free_accelerator_cache(getattr(self, "_device", "cpu"))

    def _continuation_ids(self, prompt: str, answer: str) -> list[int]:
        """Token ids of `answer` as it continues `prompt`: the tokens the
        joint text adds after the prompt's own tokens. The answer follows
        a space unless the prompt already ends in whitespace (as chat
        templates do after the assistant marker)."""
        candidate = answer if prompt[-1:].isspace() else " " + answer
        prompt_ids = self._encode(prompt)
        joint = self._encode(prompt + candidate)
        if joint[: len(prompt_ids)] == prompt_ids and len(joint) > len(prompt_ids):
            return joint[len(prompt_ids):]
        return self._tokenizer(candidate, add_special_tokens=False)["input_ids"]

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer(text, add_special_tokens=self._add_special_tokens)["input_ids"]

    def _forward_last_logits(self, prompts: Sequence[str], suffix_ids: Sequence[int] = ()):
        torch = self._torch
        encoded = self._tokenizer(
            list(prompts), return_tensors="pt", padding=True, truncation=True,
            max_length=self._max_length, add_special_tokens=self._add_special_tokens,
        )
        input_ids, attention = encoded["input_ids"], encoded["attention_mask"]
        if suffix_ids:
            suffix = torch.tensor([list(suffix_ids)] * input_ids.shape[0], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, suffix], dim=1)
            attention = torch.cat([attention, torch.ones_like(suffix)], dim=1)
        position_ids = (attention.cumsum(dim=-1) - 1).clamp(min=0)
        with torch.inference_mode():
            logits = self._model(
                input_ids=input_ids.to(self._input_device),
                attention_mask=attention.to(self._input_device),
                position_ids=position_ids.to(self._input_device),
            ).logits
        return logits.float()

    def _score_first_token(self, prompts: Sequence[str]) -> tuple[list[Distribution], list[float]]:
        torch = self._torch
        probs = torch.softmax(self._forward_last_logits(prompts)[:, -1, :], dim=-1)
        option_probs = probs[:, [ids[0] for ids in self._option_ids]].cpu()
        mass = option_probs.sum(dim=-1)
        dists = (option_probs / mass.unsqueeze(-1).clamp(min=1e-30)).tolist()
        return [tuple(float(p) for p in row) for row in dists], [float(m) for m in mass]

    def _score_sequences(self, prompts: Sequence[str]) -> tuple[list[Distribution], list[float]]:
        torch = self._torch
        log_probs = []
        for option_ids in self._option_ids:
            logits = self._forward_last_logits(prompts, option_ids)
            n = len(option_ids)
            step_logprobs = torch.log_softmax(logits[:, -n - 1 : -1, :], dim=-1)
            target = torch.tensor(option_ids, device=step_logprobs.device).expand(len(prompts), n)
            log_probs.append(step_logprobs.gather(-1, target.unsqueeze(-1)).squeeze(-1).sum(dim=-1).cpu())
        stacked = torch.stack(log_probs, dim=-1)
        mass = stacked.exp().sum(dim=-1)
        dists = torch.softmax(stacked, dim=-1).tolist()
        return [tuple(float(p) for p in row) for row in dists], [float(m) for m in mass]


def hf_causal_lm_factory(spec: ModelSpec, task: TaskSpec | None, device: str) -> HFCausalLMScorer:
    """`ScorerFactory` for `type: hf-causal-lm`."""
    return HFCausalLMScorer(spec=spec, task=task, device=device)


def _check_choice(model: str, key: str, value: object, allowed: Sequence[str]) -> None:
    if value not in allowed:
        raise ValueError(f"hf-causal-lm {model!r}: {key} must be one of {list(allowed)}, got {value!r}")


def _model_kwargs(torch, settings: dict, device: str) -> dict:
    """`from_pretrained` keyword arguments for the dtype and quantisation settings."""
    import transformers

    kwargs: dict = {}
    dtype = settings["dtype"]
    if dtype != "auto":
        dtype = getattr(torch, dtype)
    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = dtype
    if settings["quantization"] != "none":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("quantization needs the `bitsandbytes` package: pip install bitsandbytes") from exc
        if settings["quantization"] == "4bit":
            compute = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=compute)
        else:
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = "auto"
    return kwargs
