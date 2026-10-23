"""Targeted swaps from per-row tags. See docs/DATA.md §2.3.

The study tags each post (with an LLM, then by hand) with the words to swap
along each axis, including co-referring words and descriptive modifiers
(she/woman/her -> he/man/his, "wheelchair user" -> "person"). This module
applies those tags: every occurrence of each tagged phrase is replaced at
once, longest phrases first, as whole words, case-insensitively, keeping
the original's capitalisation.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from ._text import match_case, words_of
from .types import Change, Perturbation


def apply_tagged_swap(
    text: str, pairs: Sequence[tuple[str, str]], kind: str
) -> tuple[Perturbation, list[str]]:
    """Apply `pairs` of (phrase, replacement) to `text`.

    Returns the perturbation and the tagged phrases that never occurred in
    the text (a tagging error worth flagging). Phrases are matched on word
    boundaries with any whitespace between their words.
    """
    replacements: dict[str, str] = {}
    order: list[str] = []
    for original, replacement in pairs:
        key = " ".join(words_of(original))
        if key and key not in replacements:
            replacements[key] = replacement
            order.append(original)
    if not replacements:
        return Perturbation(kind=kind, text=text, changes=()), []

    phrases = sorted(replacements, key=lambda key: (-len(key.split()), -len(key)))
    pattern = re.compile("|".join(_phrase_pattern(key) for key in phrases), flags=re.IGNORECASE)

    changes: list[Change] = []

    def _substitute(match: re.Match) -> str:
        matched = match.group(0)
        replacement = match_case(matched.split()[0], replacements[" ".join(words_of(matched))])
        changes.append(Change(original=matched, replacement=replacement))
        return replacement

    new_text = pattern.sub(_substitute, text)
    # A phrase counts as found if it occurs anywhere, even inside a longer
    # tagged phrase that took precedence ("her" inside "her sister").
    missing = [
        original for original in order
        if not re.search(_phrase_pattern(" ".join(words_of(original))), text, flags=re.IGNORECASE)
    ]
    return Perturbation(kind=kind, text=new_text, changes=tuple(changes)), missing


def _phrase_pattern(key: str) -> str:
    """Regex for a lowercase, single-spaced phrase: whole words, any whitespace between them."""
    return r"(?<![A-Za-z])" + r"\s+".join(map(re.escape, key.split())) + r"(?![A-Za-z])"
