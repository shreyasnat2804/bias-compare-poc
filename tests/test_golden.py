"""Regression lock: the framework must reproduce the POC's numbers exactly.

`golden/poc_v01_fake_scorers.json` was written by the POC code (commit
7d4020d) for `data/posts.jsonl` with the two deterministic fake scorers from
`test_compare.py`, `n_controls=5`, `seed=0`. Every key and value in that file
must still appear, unchanged, in a run of the current code with the same
inputs. New keys may be added (the results schema grows), but nothing that
was there may change.

Only regenerate the golden file for a deliberate, documented change to the
numbers, never to make this test pass after a refactor.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from biascompare import compare_models
from test_compare import BIASED_VS_UNBIASED, DATASET, _fake_factory

GOLDEN = Path(__file__).resolve().parent / "golden" / "poc_v01_fake_scorers.json"

# Keys whose values are expected to differ between runs or schema versions.
_IGNORED_PATHS = {("schema_version",), ("run", "timestamp"), ("run", "dataset")}


def _assert_superset(expected: object, actual: object, path: tuple) -> None:
    if path in _IGNORED_PATHS:
        return
    where = "/".join(str(part) for part in path) or "<root>"
    if isinstance(expected, dict):
        assert isinstance(actual, dict), f"{where}: expected an object, got {type(actual).__name__}"
        for key, value in expected.items():
            assert key in actual, f"{where}: missing key {key!r}"
            _assert_superset(value, actual[key], path + (key,))
    elif isinstance(expected, list):
        assert isinstance(actual, (list, tuple)), f"{where}: expected a list, got {type(actual).__name__}"
        assert len(actual) == len(expected), f"{where}: length {len(actual)} != {len(expected)}"
        for index, (exp_item, act_item) in enumerate(zip(expected, actual)):
            _assert_superset(exp_item, act_item, path + (index,))
    elif expected is None:
        assert actual is None or (isinstance(actual, float) and math.isnan(actual)), (
            f"{where}: expected null, got {actual!r}"
        )
    elif isinstance(expected, float) or isinstance(actual, float):
        assert math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12), (
            f"{where}: {actual!r} != {expected!r}"
        )
    else:
        assert actual == expected, f"{where}: {actual!r} != {expected!r}"


def test_current_code_reproduces_poc_results():
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    actual = compare_models(BIASED_VS_UNBIASED, DATASET, n_controls=5, seed=0, scorer_factory=_fake_factory)
    _assert_superset(expected, actual, ())


def test_legacy_format_copy_of_the_dataset_reproduces_poc_results():
    """The sample data is stored in the v1 format; the original v0.1 file is
    kept next to the golden results and must give the same numbers."""
    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    legacy = GOLDEN.parent / "posts_v01.jsonl"
    actual = compare_models(BIASED_VS_UNBIASED, legacy, n_controls=5, seed=0, scorer_factory=_fake_factory)
    _assert_superset(expected, actual, ())
