"""Scoring package: the `Scorer` protocol, label-to-level mapping helpers,
the adapter registry, and the HF classifier scorer. See DESIGN.md §4 and
docs/PLAN.md §3.2.

Importing this package does not require torch/transformers: adapters only
import them lazily, inside their methods.
"""

from __future__ import annotations

from biascompare.scoring.base import (
    LevelMapping,
    ProbPair,
    Scorer,
    ScorerFactory,
    binary_from_logits,
    distribution_from_logits,
    resolve_label_index,
    resolve_level_mapping,
)
from biascompare.scoring.causal_lm import HFCausalLMScorer, hf_causal_lm_factory, preview_llm_prompt
from biascompare.scoring.classifier import HFClassifierScorer, hf_classifier_factory
from biascompare.scoring.registry import (
    available_scorer_types,
    check_scorer_type,
    default_scorer_factory,
    register_scorer,
)

__all__ = [
    "LevelMapping",
    "ProbPair",
    "Scorer",
    "ScorerFactory",
    "resolve_label_index",
    "binary_from_logits",
    "resolve_level_mapping",
    "distribution_from_logits",
    "HFClassifierScorer",
    "hf_classifier_factory",
    "HFCausalLMScorer",
    "hf_causal_lm_factory",
    "preview_llm_prompt",
    "available_scorer_types",
    "check_scorer_type",
    "default_scorer_factory",
    "register_scorer",
]
