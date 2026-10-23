"""Bidirectional gender-lexicon swap: the targeted perturbation (DESIGN.md §5.1)."""
from __future__ import annotations

import importlib.resources
import json
from pathlib import Path
from typing import Sequence

from ._text import WORD_RE, match_case
from .types import Change, Perturbation

# Function words that, immediately following "her", indicate the object-
# pronoun reading ("I gave it to her and left" -> "her" is an object, not a
# possessive determiner). Punctuation or end-of-text is handled separately
# and also favors the object reading ("him").
_FUNCTION_WORDS = frozenset(
    {
        "to", "and", "or", "but", "the", "a", "an", "that", "with", "for",
        "about", "in", "on", "at", "from", "if", "as", "again", "is", "was",
        "off", "out", "up",
    }
)


def _favors_object_form(text: str, pos: int) -> bool:
    """True when the token following index `pos` in `text` favors "him" over
    "his": punctuation, end-of-text, or a function word."""
    rest = text[pos:].lstrip()
    if not rest:
        return True
    if not rest[0].isalpha():
        return True
    match = WORD_RE.match(rest)
    word = match.group(0).lower() if match else ""
    return word in _FUNCTION_WORDS


class GenderSwapper:
    """Bidirectional word-level gender swap driven by a lexicon of pairs.

    `pairs` is a sequence of (male_word, female_word) tuples. The mapping is
    bidirectional and case-insensitive (keys are lowercased). A female word
    that comes from more than one male word (only "her", from "him"/"his")
    is disambiguated at swap time by `_favors_object_form`; any other
    many-to-one mapping is a lexicon authoring error and raises.
    """

    def __init__(self, groups: tuple[str, str], pairs: Sequence[tuple[str, str]]) -> None:
        self.groups = groups
        male_to_female: dict[str, str] = {}
        female_to_male: dict[str, list[str]] = {}
        for male_word, female_word in pairs:
            male_lower, female_lower = male_word.lower(), female_word.lower()
            existing = male_to_female.get(male_lower)
            if existing is not None and existing != female_lower:
                raise ValueError(
                    f"conflicting gender lexicon entry for {male_lower!r}: "
                    f"{existing!r} vs {female_lower!r}"
                )
            male_to_female[male_lower] = female_lower
            candidates = female_to_male.setdefault(female_lower, [])
            if male_lower not in candidates:
                candidates.append(male_lower)
        self._male_to_female = male_to_female
        self._female_to_male = {
            word: tuple(candidates) for word, candidates in female_to_male.items()
        }

    @classmethod
    def from_file(cls, path: str | Path) -> "GenderSwapper":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        groups = tuple(data["groups"])
        if len(groups) != 2:
            raise ValueError(f"expected exactly two groups, got {groups!r}")
        pairs = [tuple(pair) for pair in data["pairs"]]
        return cls(groups=groups, pairs=pairs)

    @classmethod
    def from_resource(cls, name: str = "gender_lexicon.json") -> "GenderSwapper":
        resource = importlib.resources.files("biascompare") / "resources" / name
        with importlib.resources.as_file(resource) as path:
            return cls.from_file(path)

    @property
    def vocabulary(self) -> frozenset[str]:
        return frozenset(self._male_to_female) | frozenset(self._female_to_male)

    def group_counts(self, text: str) -> dict[str, int]:
        """How many lexicon words of each group `text` contains."""
        male_hits = 0
        female_hits = 0
        for match in WORD_RE.finditer(text):
            lower = match.group(0).lower()
            if lower in self._male_to_female:
                male_hits += 1
            elif lower in self._female_to_male:
                female_hits += 1
        return {self.groups[0]: male_hits, self.groups[1]: female_hits}

    def detect_group(self, text: str) -> str | None:
        """Majority lexicon group referenced in `text`, or None if there are
        no gendered words or the two groups tie."""
        counts = self.group_counts(text)
        male_hits, female_hits = counts[self.groups[0]], counts[self.groups[1]]
        if male_hits == female_hits:
            return None
        return self.groups[0] if male_hits > female_hits else self.groups[1]

    def _resolve(self, lower_word: str, text: str, pos_after: int) -> str | None:
        if lower_word in self._male_to_female:
            return self._male_to_female[lower_word]
        candidates = self._female_to_male.get(lower_word)
        if candidates is None:
            return None
        if len(candidates) == 1:
            return candidates[0]
        if set(candidates) == {"him", "his"}:
            return "him" if _favors_object_form(text, pos_after) else "his"
        raise ValueError(
            f"ambiguous female-to-male mapping for {lower_word!r} "
            f"({candidates}) has no disambiguation rule"
        )

    def swap(self, text: str) -> Perturbation:
        """Replace every lexicon word in `text` with its counterpart,
        preserving case. Builds the output immutably from spans."""
        pieces: list[str] = []
        changes: list[Change] = []
        last_end = 0
        for match in WORD_RE.finditer(text):
            word = match.group(0)
            replacement_lower = self._resolve(word.lower(), text, match.end())
            if replacement_lower is None:
                continue
            replacement = match_case(word, replacement_lower)
            pieces.append(text[last_end : match.start()])
            pieces.append(replacement)
            changes.append(Change(original=word, replacement=replacement))
            last_end = match.end()
        pieces.append(text[last_end:])
        return Perturbation(kind="gender", text="".join(pieces), changes=tuple(changes))
