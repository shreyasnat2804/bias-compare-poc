"""Choosing and building the control generator for a run. See docs/PLAN.md §3.3.

`ControlsSpec` is the serialisable description of how controls are made;
`build_control_generator` turns it into a generator:

- `synonyms`: the POC's hand-written list, extended by each row's tagged
  synonyms. The default, so existing runs reproduce exactly.
- `tagged`: each row's tagged synonyms only (the study's verified tags).
- `masked-lm`: a helper masked LM proposes replacements for any content
  word; tagged and listed synonyms take precedence for their words.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from .controls import ControlSet, SynonymControls

CONTROL_METHODS = ("synonyms", "tagged", "masked-lm")


class ControlGenerator(Protocol):
    def generate(
        self,
        text: str,
        n_words: int,
        k: int,
        rng: np.random.Generator,
        row_synonyms: Mapping[str, Sequence[str]] | None = None,
        row_protected: Iterable[str] = (),
    ) -> ControlSet: ...


@dataclass(frozen=True)
class ControlsSpec:
    """How control perturbations are generated (docs/DATA.md §3.5).

    Only `method` matters for `synonyms`. For `masked-lm`: the helper
    masked LM and sentence-similarity model, how many suggestions to keep
    per word (`top_k`), the minimum similarity a control must keep to the
    original, and whether words listed in `synonyms.json` keep using their
    listed synonyms.
    """

    method: str = "synonyms"
    mlm_model: str = "distilroberta-base"
    similarity_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 10
    min_similarity: float = 0.85
    use_synonym_list: bool = True
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.method not in CONTROL_METHODS:
            raise ValueError(f"controls.method must be one of {list(CONTROL_METHODS)}, got {self.method!r}")
        if not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError(f"controls.top_k must be an integer >= 1, got {self.top_k!r}")
        if not (0.0 <= float(self.min_similarity) <= 1.0):
            raise ValueError(f"controls.min_similarity must be in [0, 1], got {self.min_similarity!r}")

    def to_public_dict(self) -> dict:
        if self.method in ("synonyms", "tagged"):
            return {"method": self.method}
        public = asdict(self)
        public.pop("device")
        return public


def build_control_generator(spec: ControlsSpec, extra_protected: Iterable[str] = ()) -> ControlGenerator:
    """Build the generator `spec` describes. Masked-LM helper models are
    loaded here; call `close_control_generator` when done with it."""
    extra_protected = tuple(extra_protected)
    if spec.method in ("synonyms", "tagged"):
        return SynonymControls.from_resources(
            extra_protected=extra_protected, use_synonym_list=spec.method == "synonyms"
        )

    from .mlm_controls import HFMaskedLMSubstituter, HFSentenceSimilarity, MaskedLMControls

    substituter = HFMaskedLMSubstituter(spec.mlm_model, device=spec.device)
    try:
        similarity = HFSentenceSimilarity(spec.similarity_model, device=spec.device)
    except Exception:
        substituter.close()
        raise
    return MaskedLMControls.from_resources(
        substituter,
        similarity,
        extra_protected=extra_protected,
        use_synonym_list=spec.use_synonym_list,
        top_k=spec.top_k,
        min_similarity=float(spec.min_similarity),
    )


def close_control_generator(generator: ControlGenerator) -> None:
    close = getattr(generator, "close", None)
    if close is not None:
        close()


__all__ = [
    "CONTROL_METHODS",
    "ControlGenerator",
    "ControlsSpec",
    "build_control_generator",
    "close_control_generator",
]
