"""Tests for scoring.base.load_error: Hub load failures name the fix.

The exception chains mirror what `transformers` raises (an `OSError` wrapping
`huggingface_hub`'s error); the Hub's own messages are copied verbatim.
No torch needed."""

from __future__ import annotations

import pytest

from biascompare.inputs import ModelSpec
from biascompare.scoring.base import load_error

LLAMA = ModelSpec("meta-llama/Llama-3.1-8B-Instruct", type="hf-causal-lm")


class GatedRepoError(Exception):
    pass


class RepositoryNotFoundError(Exception):
    pass


def _wrapped(inner: Exception, outer: str) -> OSError:
    try:
        try:
            raise inner
        except Exception as exc:
            raise OSError(f"{outer}\n{exc}") from exc
    except OSError as wrapped:
        return wrapped


NOT_GRANTED = _wrapped(
    GatedRepoError(
        "403 Client Error.\n\nCannot access gated repo for url https://huggingface.co/meta-llama/"
        "Llama-3.1-8B-Instruct/resolve/main/config.json.\nAccess to model meta-llama/Llama-3.1-8B-Instruct "
        "is restricted and you are not in the authorized list. Visit "
        "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct to ask for access."
    ),
    "You are trying to access a gated repo.",
)
NO_TOKEN = _wrapped(
    GatedRepoError(
        "401 Client Error.\n\nCannot access gated repo for url https://huggingface.co/meta-llama/"
        "Llama-3.1-8B-Instruct/resolve/main/config.json.\nAccess to model meta-llama/Llama-3.1-8B-Instruct "
        "is restricted. You must have access to it and be authenticated to access it. Please log in."
    ),
    "You are trying to access a gated repo.",
)
FINE_GRAINED = _wrapped(
    GatedRepoError(
        "403 Forbidden: Please enable access to public gated repositories in your fine-grained token "
        "settings to view this repository."
    ),
    "You are trying to access a gated repo.",
)
MISSING = _wrapped(
    RepositoryNotFoundError("404 Client Error."),
    "meta-llama/Llama-3.1-8B-Instruct is not a local folder and is not a valid model identifier",
)


@pytest.mark.parametrize(
    "exc, hint",
    [
        (NOT_GRANTED, "has not been granted access yet"),
        (NO_TOKEN, "no valid token was sent"),
        (FINE_GRAINED, "fine-grained token cannot read gated repos"),
        (MISSING, "neither a Hub model id nor a local folder"),
        (ValueError("CUDA out of memory"), "If this is a gated or private model"),
    ],
)
def test_load_error_names_the_fix_on_its_first_line(exc, hint):
    error = load_error(LLAMA, exc)
    first, *rest = str(error).splitlines()
    assert first.startswith("Failed to load model 'meta-llama/Llama-3.1-8B-Instruct': ")
    assert hint in first
    assert rest and rest[0].startswith("Original error: ")


def test_not_granted_points_to_the_model_page():
    first = str(load_error(LLAMA, NOT_GRANTED, what="tokenizer of")).splitlines()[0]
    assert first.startswith("Failed to load tokenizer of ")
    assert "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct" in first
