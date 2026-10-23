"""Matched-magnitude synonym-swap controls: the baseline perturbation (DESIGN.md §5.2)."""
from __future__ import annotations

import importlib.resources
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from ._text import WORD_RE, match_case
from .types import Change, Perturbation

_DEFAULT_MAX_ATTEMPTS = 50


@dataclass(frozen=True)
class ControlSet:
    """The distinct control perturbations generated for one post."""

    perturbations: tuple[Perturbation, ...]
    requested_k: int
    n_words_requested: int
    n_words_matched: int  # words actually changed per control = min(requested, #candidates)

    @property
    def effective_k(self) -> int:
        return len(self.perturbations)

    @property
    def flags(self) -> tuple[str, ...]:
        flags: list[str] = []
        if self.n_words_matched < self.n_words_requested:
            flags.append("control_words_short")
        if self.effective_k < self.requested_k:
            flags.append("controls_short")
        return tuple(flags)


def _load_resource_json(name: str) -> dict:
    resource = importlib.resources.files("biascompare") / "resources" / name
    with importlib.resources.as_file(resource) as path:
        return json.loads(Path(path).read_text(encoding="utf-8"))


class SynonymControls:
    """Generates baseline perturbations by swapping random non-protected
    words for a listed synonym.

    `protected` words are never touched, even if they also appear in
    `synonyms` — this is how toxic/insult words and (via the caller passing
    the gender lexicon vocabulary as `extra_protected`) gendered words are
    kept out of the baseline.
    """

    def __init__(self, synonyms: Mapping[str, Sequence[str]], protected: Iterable[str]) -> None:
        self._synonyms = {word.lower(): tuple(options) for word, options in synonyms.items()}
        self._protected = frozenset(word.lower() for word in protected)

    @classmethod
    def from_resources(
        cls, extra_protected: Iterable[str] = (), use_synonym_list: bool = True
    ) -> "SynonymControls":
        """Controls from the shipped synonym list, or (`use_synonym_list=False`,
        the `tagged` method) from each row's tagged synonyms only."""
        synonyms = _load_resource_json("synonyms.json") if use_synonym_list else {}
        protected_data = _load_resource_json("protected_words.json")
        protected = set(protected_data["words"]) | set(extra_protected)
        return cls(synonyms=synonyms, protected=protected)

    def candidates(
        self,
        text: str,
        synonyms: Mapping[str, Sequence[str]] | None = None,
        protected: frozenset[str] | None = None,
    ) -> list[re.Match]:
        """Word matches in `text` that are in the synonym list and not
        protected (case-insensitive)."""
        synonyms = self._synonyms if synonyms is None else synonyms
        protected = self._protected if protected is None else protected
        return [
            match
            for match in WORD_RE.finditer(text)
            if match.group(0).lower() in synonyms
            and match.group(0).lower() not in protected
        ]

    def generate(
        self,
        text: str,
        n_words: int,
        k: int,
        rng: np.random.Generator,
        max_attempts: int = _DEFAULT_MAX_ATTEMPTS,
        row_synonyms: Mapping[str, Sequence[str]] | None = None,
        row_protected: Iterable[str] = (),
    ) -> ControlSet:
        """Generate up to `k` distinct control perturbations, each replacing
        exactly `min(n_words, #candidates)` distinct candidate positions with
        a random synonym. Deterministic given `rng`'s seed.

        `row_synonyms` (a row's tagged synonyms) extend and override the
        list for this text; `row_protected` (the row's swap words) are never
        changed."""
        synonyms, protected = merge_row_resources(self._synonyms, self._protected, row_synonyms, row_protected)
        candidate_matches = self.candidates(text, synonyms, protected)
        n_matched = min(n_words, len(candidate_matches))
        seen_texts = {text}
        perturbations: list[Perturbation] = []
        attempts = 0
        while len(perturbations) < k and attempts < max_attempts and n_matched > 0:
            attempts += 1
            chosen = sorted(
                rng.choice(len(candidate_matches), size=n_matched, replace=False).tolist()
            )
            new_text, changes = self._apply(text, candidate_matches, chosen, rng, synonyms)
            if new_text == text or new_text in seen_texts:
                continue
            seen_texts.add(new_text)
            perturbations.append(
                Perturbation(kind="control", text=new_text, changes=tuple(changes))
            )
        return ControlSet(
            perturbations=tuple(perturbations),
            requested_k=k,
            n_words_requested=n_words,
            n_words_matched=n_matched,
        )

    def _apply(
        self,
        text: str,
        candidate_matches: Sequence[re.Match],
        chosen_indices: Sequence[int],
        rng: np.random.Generator,
        synonyms: Mapping[str, Sequence[str]],
    ) -> tuple[str, list[Change]]:
        pieces: list[str] = []
        changes: list[Change] = []
        last_end = 0
        for idx in chosen_indices:
            match = candidate_matches[idx]
            word = match.group(0)
            options = synonyms[word.lower()]
            synonym = options[int(rng.integers(0, len(options)))]
            replacement = match_case(word, synonym)
            pieces.append(text[last_end : match.start()])
            pieces.append(replacement)
            changes.append(Change(original=word, replacement=replacement))
            last_end = match.end()
        pieces.append(text[last_end:])
        return "".join(pieces), changes


def merge_row_resources(
    synonyms: Mapping[str, Sequence[str]],
    protected: frozenset[str],
    row_synonyms: Mapping[str, Sequence[str]] | None,
    row_protected: Iterable[str],
) -> tuple[Mapping[str, Sequence[str]], frozenset[str]]:
    """The synonym list and protected words for one row: the row's tagged
    synonyms override the shared list (options that are protected words are
    dropped), and the row's swap words join the protected set."""
    row_protected = frozenset(word.lower() for word in row_protected)
    if not row_synonyms and not row_protected:
        return synonyms, protected
    protected = protected | row_protected
    merged = dict(synonyms)
    for word, options in (row_synonyms or {}).items():
        kept = tuple(option for option in options if option.lower() not in protected)
        if kept:
            merged[word.lower()] = kept
    return merged, protected
