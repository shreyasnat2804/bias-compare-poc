"""Per-sample divergence and label metrics. See DESIGN.md §6.

A `Distribution` is one probability per task level, ordered from least to
most toxic (see `biascompare.tasks`), the shape every `Scorer` returns per
text. `ProbPair`, `(P(non-toxic), P(toxic))`, is the 2-level case and is
kept as an alias. `predicted_label`/`flipped` with a threshold are the
binary rules; `TaskSpec.predicted_level` is the general one.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Distribution = tuple[float, ...]
ProbPair = Distribution

_EPS = 1e-12


def js_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    """Jensen-Shannon divergence between two discrete distributions, base 2.

    `p` and `q` are eps-clipped and renormalised before computing, so a zero
    probability never causes a log(0). Base-2 JSD is mathematically bounded
    in [0, 1]; the result is clamped to that range to absorb floating-point
    error (e.g. a tiny negative value from cancellation when p == q).
    """
    p_norm = _clip_and_normalise(p)
    q_norm = _clip_and_normalise(q)
    if len(p_norm) != len(q_norm):
        raise ValueError(
            f"js_divergence: p and q must have the same length, got {len(p_norm)} and {len(q_norm)}"
        )

    m = [(pi + qi) / 2.0 for pi, qi in zip(p_norm, q_norm)]
    divergence = 0.5 * _kl_divergence(p_norm, m) + 0.5 * _kl_divergence(q_norm, m)
    return max(0.0, min(1.0, divergence))


def _clip_and_normalise(dist: Sequence[float]) -> list[float]:
    if len(dist) == 0:
        raise ValueError("js_divergence: distributions must be non-empty")
    clipped = [max(_EPS, float(x)) for x in dist]
    total = sum(clipped)
    return [x / total for x in clipped]


def _kl_divergence(p: Sequence[float], q: Sequence[float]) -> float:
    return sum(pi * math.log2(pi / qi) for pi, qi in zip(p, q))


def predicted_label(p: ProbPair, threshold: float = 0.5) -> int:
    """1 if P(toxic) = p[1] >= threshold, else 0."""
    return 1 if p[1] >= threshold else 0


def flipped(p: ProbPair, q: ProbPair, threshold: float = 0.5) -> bool:
    """True if the predicted label of `p` differs from that of `q` (binary rule)."""
    return predicted_label(p, threshold) != predicted_label(q, threshold)


def validate_distribution(dist: Sequence[float], n_levels: int) -> None:
    """Raise ValueError unless `dist` has `n_levels` finite, non-negative
    values summing to 1 (within 1e-6)."""
    if len(dist) != n_levels:
        raise ValueError(f"expected {n_levels} probabilities, got {len(dist)}: {list(dist)}")
    values = [float(x) for x in dist]
    if not all(math.isfinite(x) and x >= 0.0 for x in values):
        raise ValueError(f"probabilities must be finite and non-negative, got {values}")
    total = math.fsum(values)
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"probabilities must sum to 1, got {total} from {values}")
