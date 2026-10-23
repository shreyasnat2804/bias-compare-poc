"""Frozen value types shared by the perturbation modules."""
from __future__ import annotations

from dataclasses import dataclass

from ._text import word_edit_distance


@dataclass(frozen=True)
class Change:
    """One substitution made by a perturbation: a word, or a phrase for
    tagged multi-word swaps."""

    original: str
    replacement: str

    @property
    def n_words(self) -> int:
        """Words this substitution changes (word-level edit distance)."""
        return word_edit_distance(self.original, self.replacement)


@dataclass(frozen=True)
class Perturbation:
    """A perturbed version of a post's text.

    `kind` is the axis name for the targeted swap ("gender", "disability")
    or "control" for a baseline word swap. `similarity` is the sentence-embedding cosine similarity to
    the original text, recorded by generators that filter on it (masked-LM
    controls); it is `None` otherwise.
    """

    kind: str
    text: str
    changes: tuple[Change, ...]
    similarity: float | None = None

    @property
    def n_changed(self) -> int:
        """Words changed in total: the sum of each change's word edit
        distance (one per change for single-word swaps)."""
        return sum(change.n_words for change in self.changes)
