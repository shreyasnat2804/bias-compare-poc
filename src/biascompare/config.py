"""Run configuration files. See docs/PLAN.md §7 and docs/DATA.md §3.2.

A run is described by one JSON or YAML file so it can be versioned and
rerun. `load_config` validates it strictly (unknown keys are errors, with
the allowed keys listed) and returns a frozen `RunConfig`; `run` executes
it. Relative paths in the file are resolved against the file's directory.

Secrets never go in the file: a model entry names an environment variable
with `token_env` instead.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from biascompare.compare import compare_models
from biascompare.inputs import ModelSpec
from biascompare.perturb import ControlsSpec
from biascompare.tasks import TaskSpec, get_task

_RUN_KEYS = frozenset(
    {"dataset", "models", "task", "axis", "n_controls", "seed", "alpha", "device", "controls", "output"}
)
_MODEL_KEYS = frozenset({"type", "name", "toxic_label", "label_map", "revision", "token_env"})
_TASK_KEYS = frozenset({"name", "levels", "toxic_from", "instruction"})
_CONTROLS_KEYS = frozenset(field.name for field in fields(ControlsSpec))


@dataclass(frozen=True)
class RunConfig:
    """Everything `compare_models` needs, as read from a config file."""

    dataset: Path
    models: tuple[ModelSpec, ...]
    task: TaskSpec
    axis: str = "gender"
    n_controls: int = 5
    seed: int = 0
    alpha: float = 0.05
    device: str = "auto"
    controls: ControlsSpec = ControlsSpec()
    output: Path | None = None


def load_config(path: str | Path) -> RunConfig:
    """Read and validate a run config (`.json`, `.yaml` or `.yml`)."""
    config_path = Path(path)
    if not config_path.is_file():
        raise ValueError(f"Config file not found: {config_path}")
    raw = _read_mapping(config_path)
    return config_from_dict(raw, base_dir=config_path.parent)


def config_from_dict(raw: Mapping, base_dir: str | Path = ".") -> RunConfig:
    """Validate an already parsed config mapping. Relative paths resolve
    against `base_dir`."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"A run config must be an object at the top level, got {type(raw).__name__}")
    _reject_unknown("run config", raw, _RUN_KEYS)
    for required in ("dataset", "models"):
        if required not in raw:
            raise ValueError(f"run config is missing required key {required!r}")

    base = Path(base_dir)
    models = raw["models"]
    if not isinstance(models, list) or len(models) < 2:
        raise ValueError("run config 'models' must be a list of at least 2 model entries")

    output = raw.get("output")
    return RunConfig(
        dataset=_resolve(base, raw["dataset"], "dataset"),
        models=tuple(_model_from_dict(entry, index) for index, entry in enumerate(models)),
        task=_task_from_value(raw.get("task")),
        axis=_typed(raw, "axis", str, "gender"),
        n_controls=_typed(raw, "n_controls", int, 5),
        seed=_typed(raw, "seed", int, 0),
        alpha=float(_typed(raw, "alpha", (int, float), 0.05)),
        device=_typed(raw, "device", str, "auto"),
        controls=_controls_from_value(raw.get("controls")),
        output=None if output is None else _resolve(base, output, "output"),
    )


def run(config: RunConfig) -> dict:
    """Execute `config` and return the results dict (see `compare_models`)."""
    return compare_models(
        list(config.models),
        config.dataset,
        n_controls=config.n_controls,
        seed=config.seed,
        alpha=config.alpha,
        device=config.device,
        task=config.task,
        axis=config.axis,
        controls=config.controls,
    )


def _read_mapping(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("Reading YAML configs needs the `pyyaml` package") from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    raise ValueError(f"Config file must end in .json, .yaml or .yml, got {path.name!r}")


def _reject_unknown(where: str, raw: Mapping, allowed: frozenset[str]) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed keys are {sorted(allowed)}")


def _typed(raw: Mapping, key: str, kind, default):
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, kind):
        names = kind.__name__ if isinstance(kind, type) else " or ".join(k.__name__ for k in kind)
        raise ValueError(f"run config {key!r} must be {names}, got {value!r}")
    return value


def _resolve(base: Path, value: object, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"run config {key!r} must be a non-empty path string, got {value!r}")
    path = Path(value)
    return path if path.is_absolute() else base / path


def _model_from_dict(entry: object, index: int) -> ModelSpec:
    where = f"models[{index}]"
    if not isinstance(entry, Mapping):
        raise ValueError(f"{where} must be an object, got {entry!r}")
    if "token" in entry:
        raise ValueError(
            f"{where}: never put a token in a config file. Store it in an environment variable "
            "and name that variable with 'token_env'."
        )
    if "name" not in entry:
        raise ValueError(f"{where} is missing required key 'name'")
    # Any other key is an adapter setting (batch_size, max_length, ...); the
    # adapter rejects the ones it does not know.
    settings = {key: value for key, value in entry.items() if key not in _MODEL_KEYS}
    try:
        return ModelSpec(
            name=entry["name"],
            toxic_label=entry.get("toxic_label", "toxic"),
            type=entry.get("type", "hf-classifier"),
            label_map=entry.get("label_map"),
            revision=entry.get("revision"),
            token_env=entry.get("token_env"),
            settings=settings,
        )
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None


def _task_from_value(value: object) -> TaskSpec:
    if value is None or isinstance(value, str):
        return get_task(value)
    if not isinstance(value, Mapping):
        raise ValueError(f"run config 'task' must be a built-in task name or an object, got {value!r}")
    _reject_unknown("task", value, _TASK_KEYS)
    if "levels" not in value:
        raise ValueError("task block is missing required key 'levels'")
    return TaskSpec(
        name=value.get("name", "custom"),
        levels=tuple(value["levels"]),
        toxic_from=value.get("toxic_from"),
        instruction=value.get("instruction"),
    )


def _controls_from_value(value: object) -> ControlsSpec:
    if value is None:
        return ControlsSpec()
    if isinstance(value, str):
        return ControlsSpec(method=value)
    if not isinstance(value, Mapping):
        raise ValueError(f"run config 'controls' must be a method name or an object, got {value!r}")
    _reject_unknown("controls", value, _CONTROLS_KEYS)
    return ControlsSpec(**value)
