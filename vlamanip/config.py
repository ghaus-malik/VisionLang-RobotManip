"""Load the YAML configuration into nested attribute-style objects.

    cfg = load_config("configs/default.yaml", overrides=["train.batch_size=16"])
    cfg.workspace.resolution  # 0.0035
"""

from __future__ import annotations

import copy
import os
from typing import Any

import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


class Node(dict):
    """A dict whose keys are also attributes (recursively)."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(f"config has no key '{name}'") from e

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> dict:
        return {k: v.to_dict() if isinstance(v, Node) else copy.deepcopy(v) for k, v in self.items()}


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return Node({k: _wrap(v) for k, v in value.items()})
    return value


def _apply_override(cfg: Node, override: str) -> None:
    """``"a.b.c=value"``; the value is parsed as YAML (numbers, lists, booleans)."""
    if "=" not in override:
        raise ValueError(f"Override '{override}' must look like key.path=value")
    path, raw = override.split("=", 1)
    keys = path.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node:
            raise KeyError(f"Unknown config section '{key}' in override '{override}'")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"Unknown config key '{path}'")
    node[keys[-1]] = _wrap(yaml.safe_load(raw))


def load_config(path: str = DEFAULT_CONFIG, overrides: list[str] | None = None) -> Node:
    with open(path) as f:
        cfg = _wrap(yaml.safe_load(f))
    for override in overrides or []:
        _apply_override(cfg, override)
    return cfg
