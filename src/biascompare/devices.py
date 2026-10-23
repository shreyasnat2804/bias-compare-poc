"""Accelerator selection shared by every component that loads a torch model.

`torch` is imported lazily so this module imports without it installed.
"""

from __future__ import annotations


def pick_device(requested: str = "auto") -> str:
    """Resolve "auto" to cuda > mps > cpu; any other value is returned unchanged."""
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def free_accelerator_cache(device: str) -> None:
    """Release cached accelerator memory after a model has been deleted."""
    import gc

    gc.collect()
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()
    elif device == "mps":
        import torch

        torch.mps.empty_cache()
