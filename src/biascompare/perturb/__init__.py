"""Perturbation package: targeted lexicon swaps and matched-magnitude controls."""
from .controls import ControlSet, SynonymControls
from .factory import ControlGenerator, ControlsSpec, build_control_generator, close_control_generator
from .gender import GenderSwapper
from .mlm_controls import MaskedLMControls
from .types import Change, Perturbation

__all__ = [
    "Change",
    "Perturbation",
    "GenderSwapper",
    "ControlSet",
    "SynonymControls",
    "MaskedLMControls",
    "ControlGenerator",
    "ControlsSpec",
    "build_control_generator",
    "close_control_generator",
]
