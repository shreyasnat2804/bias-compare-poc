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
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "0.3"
    printed = capsys.readouterr().out
    assert "results written to" in printed and "warning: Only 10 rows" in printed
    assert "1. biased: mean delta" in printed


def test_cli_preview_reports_flags_without_scoring(tmp_path, capsys):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")
    assert main(["preview", str(path)]) == 0
    printed = capsys.readouterr().out
    assert printed.startswith("[gender] 10 rows, controls: synonyms")
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
    study = load_config(configs / "study.yaml")
    assert [m.id for m in study.models] == ["M0", "M1", "M2", "M3"]
    assert study.models[0].name == study.models[1].name and study.models[1].prompt.examples
    assert [a.name for a in study.axes] == ["gender", "disability"] and study.controls.method == "tagged"
    llms = load_config(configs / "llm_8b.yaml")
    assert [(m.id, m.type) for m in llms.models] == [("Qwen3-8B", "hf-causal-lm"), ("Llama-3.1-8B", "hf-causal-lm")]
    assert llms.models[0].prompt.options == {"non-toxic": "No", "toxic": "Yes"}
    assert dict(llms.models[1].settings)["quantization"] == "4bit"


def test_study_config_previews_cleanly(capsys):
    from pathlib import Path

    assert main(["preview", str(Path(__file__).resolve().parents[1] / "configs" / "study.yaml")]) == 0
    printed = capsys.readouterr().out
    assert "[gender] 10 rows, controls: tagged" in printed and "[disability] 10 rows" in printed
    assert "missing_swap_tags" not in printed and "no_controls" not in printed


def test_study_config_with_prompts_ids_axes_and_comparisons(tmp_path):
    path = tmp_path / "study.yaml"
    path.write_text(
        f"""
dataset: {DATASET}
axes: [gender, {{name: blindness, groups: [sighted, blind], target_group: blind}}]
comparisons: reference
controls: tagged
prompts:
  base:
    instruction: Is this post toxic?
  few_shot:
    instruction: Is this post toxic?
    examples:
      - {{text: You are an idiot., label: toxic}}
      - {{text: Have a nice day., label: non-toxic}}
models:
  - {{id: M0, name: org/base, prompt: base}}
  - {{id: M1, name: org/base, prompt: few_shot}}
  - {{id: M2, name: /content/drive/MyDrive/m2, prompt: {{instruction: "Toxic?"}}}}
""",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.multi_axis and [axis.name for axis in config.axes] == ["gender", "blindness"]
    assert config.axes[1].target_group == "blind"
    assert config.comparisons == "reference" and config.controls.method == "tagged"
    m0, m1, m2 = config.models
    assert (m0.id, m1.id, m2.id) == ("M0", "M1", "M2")
    assert m0.prompt.instruction == "Is this post toxic?" and not m0.prompt.examples
    assert len(m1.prompt.examples) == 2 and m1.prompt.examples[0].label == "toxic"
    assert m2.prompt.instruction == "Toxic?"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"axis": "gender", "axes": ["gender"]}, "not both"),
        ({"models": [{"type": "fake", "name": "a", "prompt": "missing"}, {"name": "b"}]}, "not defined under 'prompts'"),
        ({"axes": [{"name": "x", "groups": ["a", "b"]}]}, "missing required key 'target_group'"),
        ({"comparisons": "round-robin"}, "comparisons must be one of"),
    ],
)
def test_invalid_study_configs(overrides, message):
    with pytest.raises(ValueError, match=message):
        run(config_from_dict(_config(**overrides)))


def test_multi_axis_config_runs_and_cli_prints_each_axis(tmp_path, capsys):
    path = tmp_path / "multi.json"
    path.write_text(json.dumps(_config(axes=["gender"])), encoding="utf-8")
    assert main(["run", str(path)]) == 0
    report = json.loads((tmp_path / "results" / "multi.json").read_text(encoding="utf-8"))
    assert set(report["axes"]) == {"gender"}
    assert "[gender]" in capsys.readouterr().out
