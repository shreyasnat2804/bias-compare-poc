"""Adapter registry: `ModelSpec.type` -> scorer factory. See docs/PLAN.md §3.2.

Orchestration code never imports an adapter directly; it asks the registry
for the factory matching each model's `type`: `hf-classifier` (BERT-family
classifiers) and `hf-causal-lm` (LLMs). New adapters register here.
"""

from __future__ import annotations

from biascompare.inputs import ModelSpec
from biascompare.scoring.base import Scorer, ScorerFactory
from biascompare.scoring.causal_lm import hf_causal_lm_factory
from biascompare.scoring.classifier import hf_classifier_factory
from biascompare.tasks import TaskSpec

_REGISTRY: dict[str, ScorerFactory] = {
    "hf-classifier": hf_classifier_factory,
    "hf-causal-lm": hf_causal_lm_factory,
}


def register_scorer(scorer_type: str, factory: ScorerFactory) -> None:
    """Make `factory` the builder for models whose `type` is `scorer_type`."""
    if not scorer_type or not isinstance(scorer_type, str):
        raise ValueError(f"scorer type must be a non-empty string, got {scorer_type!r}")
    _REGISTRY[scorer_type] = factory


def available_scorer_types() -> list[str]:
    return sorted(_REGISTRY)


def check_scorer_type(scorer_type: str) -> None:
    """Raise ValueError if no adapter is registered for `scorer_type`."""
    if scorer_type not in _REGISTRY:
        raise ValueError(
            f"Unknown model type {scorer_type!r}; available types are {available_scorer_types()}"
        )


def default_scorer_factory(spec: ModelSpec, task: TaskSpec, device: str) -> Scorer:
    """The default `ScorerFactory`: dispatch on `spec.type`."""
    check_scorer_type(spec.type)
    return _REGISTRY[spec.type](spec, task, device)
