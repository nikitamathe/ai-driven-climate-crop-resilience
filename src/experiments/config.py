"""Experiment configuration loading and hashing (E08).

Each experiment is defined by a YAML file (see ``experiments/*.yaml``) carrying
the seed, temporal split cutoff, model name, and model parameters. This module
loads such a file, validates its required fields, and computes a stable
``config_hash`` used to name the artifact bundle under
``data/processed/experiments/<config_hash>/``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from src.data.loader import PROJECT_ROOT

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Required top-level keys in every experiment config.
REQUIRED_KEYS = ("name", "seed", "split_cutoff", "model", "params")

# Supported models (see runner). Only random_forest is wired in for now.
SUPPORTED_MODELS = ("random_forest",)


class ConfigError(Exception):
    """Raised when an experiment config is missing or malformed."""


def load_config(path: Path | str) -> dict:
    """Load and validate an experiment YAML configuration.

    Returns the config as a plain dict. Raises :class:`ConfigError` if the
    file is missing, not valid YAML, or missing a required key.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            config = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML in {path}: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigError(f"config must be a mapping, got {type(config).__name__}")

    missing = [k for k in REQUIRED_KEYS if k not in config]
    if missing:
        raise ConfigError(f"config {path} missing required keys: {missing}")

    if config["model"] not in SUPPORTED_MODELS:
        raise ConfigError(
            f"unsupported model '{config['model']}' (supported: {SUPPORTED_MODELS})"
        )

    if "seed" not in config or not isinstance(config["seed"], int):
        raise ConfigError("config 'seed' must be an integer")

    return config


def resolve_config_path(name: str) -> Path:
    """Resolve an experiment name/path to a YAML file.

    If ``name`` has a ``.yaml``/``.yml`` suffix it is treated as a path (relative
    path resolved against the experiments dir if not absolute). Otherwise ``name``
    is matched against ``experiments/<name>.yaml``.
    """
    p = Path(name)
    if p.suffix in (".yaml", ".yml"):
        return p if p.is_absolute() else (EXPERIMENTS_DIR / p)
    return EXPERIMENTS_DIR / f"{name}.yaml"


def config_hash(config: dict) -> str:
    """Return a stable short hash (first 8 hex chars of SHA-256) of a config.

    Serialization is deterministic: the dict is sorted recursively and JSON
    encoded with sorted keys, so identical configs always hash the same.
    """
    canonical = json.dumps(_sort_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]


def _sort_dict(value):
    if isinstance(value, dict):
        return {str(k): _sort_dict(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_sort_dict(v) for v in value]
    return value
