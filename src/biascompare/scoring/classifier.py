"""Hugging Face text-classification scorer (`type: hf-classifier`). See
DESIGN.md §4 and docs/PLAN.md §4.1.

`torch` and `transformers` are imported lazily, inside `HFClassifierScorer`
methods only, so importing this module (and the rest of the package) never
requires either to be installed.
"""

from __future__ import annotations

from collections.abc import Sequence

from biascompare.devices import free_accelerator_cache, pick_device
from biascompare.inputs import ModelSpec
from biascompare.metrics import Distribution
from biascompare.scoring.base import distribution_from_logits, resolve_level_mapping
from biascompare.tasks import DEFAULT_TASK, TaskSpec

__all__ = ["HFClassifierScorer", "hf_classifier_factory", "pick_device"]

_DEFAULT_MAX_LENGTH = 512
_DEFAULT_BATCH_SIZE = 16
_KNOWN_SETTINGS = frozenset({"batch_size", "max_length"})


class HFClassifierScorer:
    """Scorer backed by `AutoModelForSequenceClassification` (DESIGN §4).

    The model's own labels are mapped onto the task's levels by
    `resolve_level_mapping` (`toxic_label` or `label_map` on the spec).
    Loads eagerly in `__init__`; call `close()` once done so its memory (and
    CUDA/MPS cache) is freed before the next model loads: one model is
    loaded at a time.
    """

    scorer_type = "hf-classifier"
    uses_instruction = False  # classifiers score the raw text; the task instruction is ignored

    def __init__(
        self,
        spec: ModelSpec,
        task: TaskSpec | None = None,
        device: str = "auto",
        batch_size: int | None = None,
    ) -> None:
        unknown = sorted(set(spec.settings) - _KNOWN_SETTINGS)
        if unknown:
            raise ValueError(
                f"Unknown setting(s) {unknown} for hf-classifier {spec.name!r}; "
                f"allowed: {sorted(_KNOWN_SETTINGS)}"
            )
        self.name = spec.name
        self._task = task or DEFAULT_TASK
        self._batch_size = int(batch_size or spec.settings.get("batch_size") or _DEFAULT_BATCH_SIZE)
        self._device = pick_device(device)

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HFClassifierScorer requires the `torch` and `transformers` packages "
                "to be installed. Install the project's dependencies (see "
                "pyproject.toml) before scoring."
            ) from exc

        token = spec.resolved_token()
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(spec.name, token=token, revision=spec.revision)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                spec.name, token=token, revision=spec.revision
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load model {spec.name!r}. If this is a gated or private "
                "model, pass a valid `token` on its ModelSpec, name an environment variable "
                f"with `token_env`, or set HF_TOKEN. Original error: {exc}"
            ) from exc

        self._model.to(self._device)
        self._model.eval()

        config = self._model.config
        self._mapping = resolve_level_mapping(
            config.id2label,
            self._task,
            multi_label=config.problem_type == "multi_label_classification",
            toxic_label=spec.toxic_label,
            label_map=spec.label_map,
            model_name=repr(spec.name),
        )
        max_length = spec.settings.get("max_length") or _DEFAULT_MAX_LENGTH
        self._max_length = min(self._tokenizer.model_max_length, int(max_length))
        self._torch = torch
        self.metadata = {
            "revision": getattr(config, "_commit_hash", None) or spec.revision,
            "head": "sigmoid" if self._mapping.multi_label else "softmax",
            "label_mapping": dict(self._mapping.description),
            "max_length": self._max_length,
        }

    def score(self, texts: Sequence[str]) -> list[Distribution]:
        """Batch-score `texts`, truncating/padding per DESIGN §4."""
        torch = self._torch
        results: list[Distribution] = []

        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.inference_mode():
                outputs = self._model(**encoded)

            logits = outputs.logits.detach().cpu().numpy()
            for row in logits:
                results.append(distribution_from_logits(row, self._mapping))

        return results

    def close(self) -> None:
        """Free the model and any accelerator cache it held."""
        if hasattr(self, "_model"):
            del self._model
        free_accelerator_cache(self._device)


def hf_classifier_factory(spec: ModelSpec, task: TaskSpec | None, device: str) -> HFClassifierScorer:
    """`ScorerFactory` for `type: hf-classifier`."""
    return HFClassifierScorer(spec=spec, task=task, device=device)
