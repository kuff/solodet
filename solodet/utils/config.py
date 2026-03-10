"""YAML configuration loading with optional overrides."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path | None, overrides: dict[str, Any] | None = None) -> dict:
    """Load a YAML config file with optional key overrides.

    Args:
        path: Path to YAML file. Returns empty dict if None.
        overrides: Dict of key=value overrides to apply on top.

    Returns:
        Merged configuration dict.
    """
    if path is None:
        return overrides or {}

    path = Path(path)
    if not path.is_file():
        return overrides or {}

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    if overrides:
        cfg.update(overrides)

    return cfg


def save_config(cfg: dict, path: str | Path) -> None:
    """Save a configuration dict to YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
