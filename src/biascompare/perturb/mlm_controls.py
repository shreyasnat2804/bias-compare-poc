"""Masked-LM controls: matched-magnitude word substitutions for arbitrary
text, the baseline perturbation for real datasets. See docs/PLAN.md §3.3.

The synonym-list controls (`controls.py`) only know the words in
`resources/synonyms.json`. This generator instead asks a small helper
masked language model what could replace each candidate word in context,
filters those suggestions, builds K distinct variants that each change
exactly as many words as the gender swap did, and keeps only variants whose
sentence embedding stays close to the original.

The two helper models sit behind small protocols (`Substituter`,
`SimilarityModel`) so the sampling and filtering logic is tested offline
with fakes; `HFMaskedLMSubstituter` and `HFSentenceSimilarity` are the real
implementations, importing torch/transformers lazily. They are separate
from the models under test and are freed before any of those load.
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

import numpy as np

from ._text import WORD_RE, match_case
from .controls import ControlSet, _load_resource_json, merge_row_resources
from .types import Change, Perturbation

_ASCII_WORD = re.compile(r"^[A-Za-z]+$")
_DEFAULT_MAX_ATTEMPTS = 50
_OVERSAMPLE = 3  # draft this many times K variants before the similarity filter


class Substituter(Protocol):
    """Suggests replacement words for spans of a text, in context."""

    def substitutes(
        self, text: str, spans: Sequence[tuple[int, int]], top_k: int
    ) -> list[list[tuple[str, float]]]:
        """For each `(start, end)` span, up to a few `top_k`-scale
        `(word, probability)` suggestions, most likely first."""
        ...

    def close(self) -> None: ...


class SimilarityModel(Protocol):
    """Scores how close each variant stays to the original text."""

    def similarities(self, original: str, variants: Sequence[str]) -> list[float]: ...

    def close(self) -> None: ...


class MaskedLMControls:
    """Generates baseline perturbations by replacing randomly chosen
    candidate words with in-context suggestions from a masked LM.

    Candidate words are ASCII words that are not protected (gender lexicon,
    toxic vocabulary) and are either listed in `synonyms` or are content
    words (not a stopword, at least `min_word_length` letters). A listed
    word uses its listed synonyms; any other word uses the masked LM's
    suggestions, which are dropped if they are not a whole ASCII word, equal
    the original, are protected, or are stopwords.
    """

    def __init__(
        self,
        substituter: Substituter,
        similarity: SimilarityModel,
        protected: Iterable[str],
        stopwords: Iterable[str] = (),
        synonyms: Mapping[str, Sequence[str]] | None = None,
        top_k: int = 10,
        min_similarity: float = 0.85,
        min_word_length: int = 3,
    ) -> None:
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")
        if not (0.0 <= min_similarity <= 1.0):
            raise ValueError(f"min_similarity must be in [0, 1], got {min_similarity}")
        self._substituter = substituter
        self._similarity = similarity
        self._protected = frozenset(word.lower() for word in protected)
        self._stopwords = frozenset(word.lower() for word in stopwords)
        self._synonyms = {
            word.lower(): tuple(options) for word, options in (synonyms or {}).items()
        }
        self._top_k = top_k
        self._min_similarity = min_similarity
        self._min_word_length = min_word_length

    @classmethod
    def from_resources(
        cls,
        substituter: Substituter,
        similarity: SimilarityModel,
        extra_protected: Iterable[str] = (),
        use_synonym_list: bool = True,
        **kwargs,
    ) -> "MaskedLMControls":
        protected = set(_load_resource_json("protected_words.json")["words"]) | set(extra_protected)
        stopwords = _load_resource_json("stopwords.json")["words"]
        synonyms = _load_resource_json("synonyms.json") if use_synonym_list else None
        return cls(substituter, similarity, protected, stopwords, synonyms, **kwargs)

    def candidates(
        self,
        text: str,
        synonyms: Mapping[str, Sequence[str]] | None = None,
        protected: frozenset[str] | None = None,
    ) -> list[re.Match]:
        """Word matches in `text` that a control may change."""
        synonyms = self._synonyms if synonyms is None else synonyms
        protected = self._protected if protected is None else protected
        found = []
        for match in WORD_RE.finditer(text):
            lower = match.group(0).lower()
            if lower in protected:
                continue
            if lower in synonyms or (
                lower not in self._stopwords and len(lower) >= self._min_word_length
            ):
                found.append(match)
        return found

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
        """Generate up to `k` distinct controls, each replacing exactly
        `min(n_words, #usable positions)` words, keeping only variants with
        similarity >= `min_similarity`. Deterministic given `rng`'s seed.

        A row's tagged synonyms act like listed synonyms for that text; its
        swap words are never changed or introduced."""
        synonyms, protected = merge_row_resources(self._synonyms, self._protected, row_synonyms, row_protected)
        matches = self.candidates(text, synonyms, protected)
        options = self._options(text, matches, synonyms, protected) if matches and n_words > 0 else []
        usable = [(match, opts) for match, opts in zip(matches, options) if opts]
        n_matched = min(n_words, len(usable))

        drafts: list[tuple[str, list[Change]]] = []
        seen = {text}
        attempts = 0
        while n_matched > 0 and len(drafts) < k * _OVERSAMPLE and attempts < max_attempts:
            attempts += 1
            chosen = sorted(rng.choice(len(usable), size=n_matched, replace=False).tolist())
            new_text, changes = _apply(text, [usable[i] for i in chosen], rng)
            if new_text in seen:
                continue
            seen.add(new_text)
            drafts.append((new_text, changes))

        accepted: list[Perturbation] = []
        if drafts:
            scores = self._similarity.similarities(text, [draft for draft, _ in drafts])
            for (draft, changes), score in zip(drafts, scores):
                if score >= self._min_similarity and len(accepted) < k:
                    accepted.append(
                        Perturbation(kind="control", text=draft, changes=tuple(changes), similarity=float(score))
                    )

        return ControlSet(
            perturbations=tuple(accepted),
            requested_k=k,
            n_words_requested=n_words,
            n_words_matched=n_matched,
        )

    def close(self) -> None:
        self._substituter.close()
        self._similarity.close()

    def _options(
        self,
        text: str,
        matches: Sequence[re.Match],
        synonyms: Mapping[str, Sequence[str]] | None = None,
        protected: frozenset[str] | None = None,
    ) -> list[tuple[str, ...]]:
        """Allowed replacement words (lowercase) for every candidate match."""
        synonyms = self._synonyms if synonyms is None else synonyms
        protected = self._protected if protected is None else protected
        options: list[tuple[str, ...] | None] = [None] * len(matches)
        to_ask: list[int] = []
        for index, match in enumerate(matches):
            listed = synonyms.get(match.group(0).lower())
            if listed is not None:
                options[index] = self._filter(match.group(0), listed, protected)
            else:
                to_ask.append(index)

        if to_ask:
            spans = [(matches[i].start(), matches[i].end()) for i in to_ask]
            suggestions = self._substituter.substitutes(text, spans, self._top_k)
            for index, words in zip(to_ask, suggestions):
                options[index] = self._filter(matches[index].group(0), [word for word, _ in words], protected)
        return [opts or () for opts in options]

    def _filter(self, original: str, words: Iterable[str], protected: frozenset[str]) -> tuple[str, ...]:
        kept: list[str] = []
        original_lower = original.lower()
        for word in words:
            lower = word.strip().lower()
            if (
                len(lower) < 2
                or not _ASCII_WORD.match(lower)
                or lower == original_lower
                or lower in protected
                or lower in self._stopwords
                or lower in kept
            ):
                continue
            kept.append(lower)
            if len(kept) == self._top_k:
                break
        return tuple(kept)


def _apply(
    text: str, chosen: Sequence[tuple[re.Match, tuple[str, ...]]], rng: np.random.Generator
) -> tuple[str, list[Change]]:
    pieces: list[str] = []
    changes: list[Change] = []
    last_end = 0
    for match, opts in chosen:
        word = match.group(0)
        replacement = match_case(word, opts[int(rng.integers(0, len(opts)))])
        pieces.append(text[last_end : match.start()])
        pieces.append(replacement)
        changes.append(Change(original=word, replacement=replacement))
        last_end = match.end()
    pieces.append(text[last_end:])
    return "".join(pieces), changes


def _subword_scheme(tokenizer) -> str:
    """"bpe" (RoBERTa/GPT-2 style, Ġ marks a leading space),
    "sentencepiece" (▁ marks a leading space) or "wordpiece" (## marks a
    continuation)."""
    tokens = tokenizer.tokenize("hello world")
    if any(token.startswith("Ġ") for token in tokens):
        return "bpe"
    if any(token.startswith("▁") for token in tokens):
        return "sentencepiece"
    return "wordpiece"


def _whole_word(token: str, scheme: str, preceded_by_space: bool) -> str | None:
    """The word a vocabulary token stands for, if it is a complete word that
    fits where the masked word was; otherwise None."""
    if scheme == "wordpiece":
        word = None if token.startswith("##") else token
    else:
        marker = "Ġ" if scheme == "bpe" else "▁"
        if token.startswith(marker) != preceded_by_space:
            return None
        word = token[len(marker):] if preceded_by_space else token
    if word and _ASCII_WORD.match(word):
        return word
    return None


class HFMaskedLMSubstituter:
    """`Substituter` backed by `AutoModelForMaskedLM` (default `distilroberta-base`).

    Each span is replaced by one mask token and the model's top predictions
    at that position are kept if they are whole words. A word the tokenizer
    splits into several pieces is still masked with a single token, so only
    single-token replacements are proposed.
    """

    def __init__(
        self,
        model_name: str = "distilroberta-base",
        device: str = "auto",
        batch_size: int = 32,
        revision: str | None = None,
        token: str | None = None,
    ) -> None:
        from biascompare.devices import pick_device

        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self._torch = torch
        self._device = pick_device(device)
        self._batch_size = batch_size
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, token=token)
            self._model = AutoModelForMaskedLM.from_pretrained(model_name, revision=revision, token=token)
        except Exception as exc:
            raise RuntimeError(f"Failed to load masked LM {model_name!r} for controls: {exc}") from exc
        if self._tokenizer.mask_token is None:
            raise ValueError(f"{model_name!r} has no mask token, so it cannot propose substitutions")
        self._model.to(self._device)
        self._model.eval()
        self._scheme = _subword_scheme(self._tokenizer)
        self._max_length = min(self._tokenizer.model_max_length, 512)

    def substitutes(
        self, text: str, spans: Sequence[tuple[int, int]], top_k: int
    ) -> list[list[tuple[str, float]]]:
        torch = self._torch
        tokenizer = self._tokenizer
        masked = [text[:start] + tokenizer.mask_token + text[end:] for start, end in spans]
        results: list[list[tuple[str, float]]] = []
        for offset in range(0, len(masked), self._batch_size):
            batch = masked[offset : offset + self._batch_size]
            encoded = tokenizer(
                batch, padding=True, truncation=True, max_length=self._max_length, return_tensors="pt"
            ).to(self._device)
            with torch.inference_mode():
                logits = self._model(**encoded).logits
            for row, (start, _) in enumerate(spans[offset : offset + self._batch_size]):
                positions = (encoded["input_ids"][row] == tokenizer.mask_token_id).nonzero()
                if len(positions) == 0:  # the mask fell past the truncation point
                    results.append([])
                    continue
                probs = torch.softmax(logits[row, int(positions[0])], dim=-1)
                top = torch.topk(probs, k=min(top_k * 5, probs.shape[-1]))
                preceded_by_space = start > 0 and text[start - 1].isspace()
                words: list[tuple[str, float]] = []
                for token_id, prob in zip(top.indices.tolist(), top.values.tolist()):
                    token = tokenizer.convert_ids_to_tokens(token_id)
                    word = _whole_word(token, self._scheme, preceded_by_space)
                    if word is not None:
                        words.append((word, float(prob)))
                results.append(words)
        return results

    def close(self) -> None:
        from biascompare.devices import free_accelerator_cache

        if hasattr(self, "_model"):
            del self._model
        free_accelerator_cache(self._device)


class HFSentenceSimilarity:
    """`SimilarityModel` from mean-pooled transformer embeddings (default
    `sentence-transformers/all-MiniLM-L6-v2`, which is trained for exactly
    this pooling), compared by cosine similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        batch_size: int = 32,
        revision: str | None = None,
        token: str | None = None,
    ) -> None:
        from biascompare.devices import pick_device

        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self._device = pick_device(device)
        self._batch_size = batch_size
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, token=token)
            self._model = AutoModel.from_pretrained(model_name, revision=revision, token=token)
        except Exception as exc:
            raise RuntimeError(f"Failed to load similarity model {model_name!r} for controls: {exc}") from exc
        self._model.to(self._device)
        self._model.eval()
        self._max_length = min(self._tokenizer.model_max_length, 512)

    def similarities(self, original: str, variants: Sequence[str]) -> list[float]:
        if not variants:
            return []
        embeddings = self._embed([original, *variants])
        return (embeddings[1:] @ embeddings[0]).tolist()

    def _embed(self, texts: Sequence[str]):
        torch = self._torch
        chunks = []
        for offset in range(0, len(texts), self._batch_size):
            encoded = self._tokenizer(
                list(texts[offset : offset + self._batch_size]),
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            chunks.append(torch.nn.functional.normalize(pooled, dim=-1).cpu())
        return torch.cat(chunks).numpy()

    def close(self) -> None:
        from biascompare.devices import free_accelerator_cache

        if hasattr(self, "_model"):
            del self._model
        free_accelerator_cache(self._device)
