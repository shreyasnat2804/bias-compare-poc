"""Comparative counterfactual bias testing for text models. See docs/PLAN.md.

Public API. `plot_comparison`/`summary_rows` are imported from
`biascompare.viz`, which only imports matplotlib lazily inside
`plot_comparison`, so importing this package never requires matplotlib.
"""

from __future__ import annotations

from biascompare.compare import compare_models, preview_perturbations
from biascompare.config import RunConfig, config_from_dict, load_config, run
from biascompare.inputs import ModelSpec, Post, load_posts
from biascompare.perturb import ControlsSpec
from biascompare.results import save_results
from biascompare.tasks import BUILTIN_TASKS, TaskSpec, get_task
from biascompare.viz import plot_comparison, summary_rows

__all__ = [
    "ModelSpec",
    "Post",
    "load_posts",
    "TaskSpec",
    "BUILTIN_TASKS",
    "get_task",
    "ControlsSpec",
    "compare_models",
    "preview_perturbations",
    "save_results",
    "RunConfig",
    "load_config",
    "config_from_dict",
    "run",
    "plot_comparison",
    "summary_rows",
]
