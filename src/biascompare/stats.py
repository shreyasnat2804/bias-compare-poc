"""Hypothesis tests and resampling utilities. See DESIGN.md §6.

Every function here returns `float('nan')` (or a tuple of nans) on
degenerate input instead of raising, so that one low-power or edge-case
comparison never crashes a full comparison run — DESIGN §6: "Degenerate
inputs (n < 2, all-zero differences) return NaN p-values, serialised as
null, never a crash."
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import stats as scipy_stats

_NAN = float("nan")


def mean_or_nan(values: Sequence[float]) -> float:
    """Arithmetic mean of `values`, or nan if it is empty."""
    if len(values) == 0:
        return _NAN
    return float(np.mean(values))


def paired_ttest_greater(diffs: Sequence[float]) -> float:
    """One-sided one-sample t-test, H1: mean(diffs) > 0 (the paper's test on Δi).

    Returns nan if n < 2, or if every diff is (numerically) identical: zero
    variance leaves the t-statistic undefined. This means a constant,
    strictly positive `diffs` also yields nan rather than a spurious
    p ≈ 0 — documented here since it may look surprising at first glance.
    """
    arr = np.asarray(diffs, dtype=float)
    if arr.size < 2 or np.allclose(arr, arr[0]):
        return _NAN

    statistic, two_sided_p = scipy_stats.ttest_1samp(arr, popmean=0.0)
    if not np.isfinite(statistic):
        return _NAN
    one_sided_p = two_sided_p / 2.0 if statistic > 0 else 1.0 - two_sided_p / 2.0
    return float(one_sided_p)


def ttest_two_sided(values: Sequence[float]) -> float:
    """Two-sided one-sample t-test, H1: mean(values) != 0.

    Returns nan if n < 2 or all values are (numerically) identical.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size < 2 or np.allclose(arr, arr[0]):
        return _NAN

    statistic, p_value = scipy_stats.ttest_1samp(arr, popmean=0.0)
    if not np.isfinite(statistic):
        return _NAN
    return float(p_value)


def wilcoxon_two_sided(diffs: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank test on `diffs`.

    Returns nan if n < 2 or every diff is zero (scipy raises ValueError for
    an all-zero input; that is treated the same as any other degenerate
    case here).
    """
    arr = np.asarray(diffs, dtype=float)
    if arr.size < 2 or np.all(arr == 0):
        return _NAN

    try:
        _, p_value = scipy_stats.wilcoxon(arr, alternative="two-sided")
    except ValueError:
        return _NAN
    return float(p_value)


def bootstrap_ci(
    values: Sequence[float],
    rng: np.random.Generator,
    n_resamples: int = 2000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean of `values`.

    Returns (nan, nan) for an empty input, and (v, v) for a single value
    (resampling a single point can never produce a different mean).
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (_NAN, _NAN)
    if arr.size == 1:
        value = float(arr[0])
        return (value, value)

    resample_indices = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    resample_means = arr[resample_indices].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(resample_means, 100 * (alpha / 2.0)))
    upper = float(np.percentile(resample_means, 100 * (1.0 - alpha / 2.0)))
    return (lower, upper)


def bonferroni(p: float, n_tests: int) -> float:
    """Bonferroni-corrected p-value: min(1, p * n_tests). nan passes through."""
    if math.isnan(p):
        return _NAN
    return float(min(1.0, p * n_tests))


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control across
    the model-vs-model comparisons of one run). nan entries are left as nan
    and do not count towards the family size."""
    indexed = [(p, i) for i, p in enumerate(p_values) if not math.isnan(p)]
    adjusted = [_NAN] * len(p_values)
    m = len(indexed)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[i] = float(running)
    return adjusted
