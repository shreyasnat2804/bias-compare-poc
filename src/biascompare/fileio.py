"""Reading JSON and YAML files (run configs, prompt files)."""

from __future__ import annotations

import json
from pathlib import Path


def read_mapping(path: Path) -> object:
    """Parse a `.json`, `.yaml` or `.yml` file. Raises ValueError with the file
    name on a parse error or an unsupported extension."""
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
            raise RuntimeError("Reading YAML files needs the `pyyaml` package") from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
    raise ValueError(f"File must end in .json, .yaml or .yml, got {path.name!r}")
