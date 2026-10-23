"""Tests for biascompare.config (load_config, run) and biascompare.cli.

A fake scorer type is registered in the adapter registry, so configs go
through the real registry dispatch with no model downloads.
"""

from __future__ import annotations

import json

import pytest

from biascompare import load_config, run
from biascompare.cli import main
from biascompare.config import config_from_dict
from biascompare.scoring.registry import _REGISTRY, register_scorer
from test_compare import DATASET, FakeScorer, _biased_toxicity, _unbiased_toxicity


@pytest.fixture(autouse=True)
def fake_scorer_type():
    def factory(spec, task, device):
        fn = _biased_toxicity if spec.name == "biased" else _unbiased_toxicity
        return FakeScorer(spec.name, fn)

    register_scorer("fake", factory)
    yield
    _REGISTRY.pop("fake", None)


def _config(**overrides) -> dict:
    raw = {
        "dataset": str(DATASET),
        "models": [{"type": "fake", "name": "biased"}, {"type": "fake", "name": "unbiased"}],
        "n_controls": 3,
    }
    raw.update(overrides)
    return raw


def test_json_config_round_trip(tmp_path):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_config(task="toxicity", seed=4)), encoding="utf-8")
    config = load_config(path)
    assert config.seed == 4 and config.n_controls == 3
    assert [spec.name for spec in config.models] == ["biased", "unbiased"]
    assert config.task.name == "toxicity"
    assert config.controls.method == "synonyms"


def test_yaml_config_with_relative_paths_task_block_and_settings(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "rows.jsonl").write_text(DATASET.read_text(encoding="utf-8"), encoding="utf-8")
    path = tmp_path / "run.yaml"
    path.write_text(
        """
dataset: data/rows.jsonl
output: out/results.json
task:
  name: three
  levels: [non-toxic, borderline, toxic]
  toxic_from: toxic
controls:
  method: masked-lm
  min_similarity: 0.9
models:
  - type: hf-classifier
    name: org/hatexplain
    label_map: {non-toxic: [normal], borderline: [offensive], toxic: [hate]}
    revision: abc123
    token_env: HATE_TOKEN
    batch_size: 8
  - name: org/other
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.dataset == tmp_path / "data" / "rows.jsonl"
    assert config.output == tmp_path / "out" / "results.json"
    assert config.task.levels == ("non-toxic", "borderline", "toxic")
    assert config.controls.method == "masked-lm" and config.controls.min_similarity == 0.9
    first, second = config.models
    assert first.label_map["toxic"] == ("hate",)
    assert first.revision == "abc123" and first.token_env == "HATE_TOKEN"
    assert dict(first.settings) == {"batch_size": 8}
    assert second.type == "hf-classifier"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"modles": []}, "unknown key"),
        ({"models": [{"name": "a"}]}, "at least 2"),
        ({"models": [{"name": "a", "token": "s3cret"}, {"name": "b"}]}, "never put a token"),
        ({"models": [{"type": "fake"}, {"name": "b"}]}, "missing required key 'name'"),
        ({"seed": "zero"}, "'seed' must be int"),
        ({"task": {"levels": ["a"]}}, "between 2 and 4"),
        ({"task": {"name": "x"}}, "missing required key 'levels'"),
        ({"controls": {"method": "masked-lm", "temperature": 1}}, "unknown key"),
    ],
)
def test_invalid_configs_raise_actionable_errors(overrides, message):
    with pytest.raises(ValueError, match=message):
        config_from_dict(_config(**overrides))


def test_run_executes_the_config():
    results = run(config_from_dict(_config(seed=0)))
    assert results["models"][0]["error"] is None
    assert results["run"]["models"][0]["type"] == "fake"
    assert results["comparisons"][0]["reference"] == "biased"


def test_cli_run_writes_results_next_to_the_config(tmp_path, capsys):
    path = tmp_path / "toxicity_gender.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    assert main(["run", str(path)]) == 0
    output = tmp_path / "results" / "toxicity_gender.json"
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "0.2"
    printed = capsys.readouterr().out
    assert "results written to" in printed and "warning: Only 10 rows" in printed


def test_cli_preview_reports_flags_without_scoring(tmp_path, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    assert main(["preview", str(path)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("10 rows, controls: synonyms")
    assert "swap:" in printed


def test_cli_reports_config_errors_without_a_traceback(tmp_path, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps({"dataset": "x"}), encoding="utf-8")
    assert main(["run", str(path)]) == 2
    assert "missing required key 'models'" in capsys.readouterr().err


def test_shipped_example_configs_load():
    from pathlib import Path

    configs = Path(__file__).resolve().parents[1] / "configs"
    poc = load_config(configs / "poc.yaml")
    assert poc.dataset.resolve() == DATASET.resolve()
    assert poc.models[1].label_map == {"toxic": ("toxic",), "non-toxic": ("neutral",)}
    validation = load_config(configs / "validation.yaml")
    assert validation.controls.method == "masked-lm"
