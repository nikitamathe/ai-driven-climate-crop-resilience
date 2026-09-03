"""Experiment runner and artifact logging (E08).

Reads a YAML experiment config, runs the pipeline with those settings, and
writes a self-describing artifact bundle to
``data/processed/experiments/<config_hash>/``:

* ``config.yaml``   — the input experiment configuration (reproducibility)
* ``params.json``   — resolved model hyper-parameters
* ``metrics.json``  — config hash, git commit hash, run timestamps, model metrics
* ``model.joblib``  — serialized trained estimator (joblib)
* ``summary.csv``   — Year x State x District x Crop resilience summary

The config hash ties a metric record to the exact configuration that produced
it, making "config X -> metrics Y" a versioned, verifiable statement.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib

from src.data.loader import PROCESSED_DIR
from src.experiments.config import ConfigError, config_hash, load_config, resolve_config_path
from src.pipeline.run_pipeline import run as run_pipeline

# Artifact files written per experiment run.
ARTIFACT_FILES = ("config.yaml", "params.json", "metrics.json", "model.joblib", "summary.csv")


def git_commit_hash() -> str:
    """Return the current short git commit hash, or "<no-git>" if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=str(PROCESSED_DIR.parents[1]),
        )
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "<no-git>"


def run_experiment(
    config_path: Path | str,
    out_root: Path | str = PROCESSED_DIR / "experiments",
) -> dict:
    """Run a single experiment from a YAML config and log its artifacts.

    Parameters
    ----------
    config_path : Path to an experiment YAML config.
    out_root : Directory under which artifact bundles are written
        (default: ``data/processed/experiments``).

    Returns
    -------
    dict describing the run: config, config_hash, run_dir, and metrics.
    """
    config = load_config(config_path)
    params = dict(config["params"]) if config["params"] else {}
    if "random_state" not in params:
        params["random_state"] = config["seed"]

    h = config_hash(config)

    # Run the full pipeline with this experiment's settings. The pipeline
    # itself re-validates the data (E07) and reproduces all standard outputs.
    result = run_pipeline(
        cutoff_year=int(config["split_cutoff"]),
        model_params=params,
        model=config["model"],
    )

    run_dir = Path(out_root) / h
    run_dir.mkdir(parents=True, exist_ok=True)

    # --- Resolved model artifact: the estimators exposed params ---
    model = result["model"]
    fitted_params = getattr(model, "get_params", lambda: params)()

    # Persist artifacts.
    _write_json(
        run_dir / "params.json",
        {"model": config["model"], "fitted_params": fitted_params},
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "experiment": config["name"],
            "config_hash": h,
            "config_file": str(config_path),
            "git_commit": git_commit_hash(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "seed": config["seed"],
            "split_cutoff": config["split_cutoff"],
            "model": config["model"],
            "params": params,
            "metrics": result["metrics"],
            "validation_warnings": result["validation"]["warnings"],
        },
    )
    _write_yaml(run_dir / "config.yaml", config)
    joblib.dump(model, run_dir / "model.joblib")
    result["summary"].to_csv(run_dir / "summary.csv", index=False)

    print(f"\n[experiment] wrote artifacts to {run_dir}")
    for fname in ARTIFACT_FILES:
        print(f"  - {run_dir / fname}")

    return {
        "config": config,
        "config_hash": h,
        "run_dir": str(run_dir),
        "metrics": result["metrics"],
    }


def _write_json(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def _write_yaml(path: Path, data: dict) -> None:
    import yaml

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)


def discover_configs(experiments_dir: Path) -> list[Path]:
    """Return the sorted list of experiment YAML configs in a directory."""
    return sorted(experiments_dir.glob("*.yaml")) + sorted(experiments_dir.glob("*.yml"))
