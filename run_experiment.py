#!/usr/bin/env python3
"""Run a single configured experiment and log its artifacts (E08).

Usage:

    python run_experiment.py experiments/base.yaml
    python run_experiment.py base            # resolves to experiments/base.yaml
    python run_experiment.py --all           # run every config in experiments/

Each run writes a self-describing bundle to
``data/processed/experiments/<config_hash>/`` containing the config, resolved
params, model metrics (with the git commit hash), the serialized model, and
the resilience summary CSV.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.experiments.config import (
    EXPERIMENTS_DIR,
    ConfigError,
    resolve_config_path,
)
from src.experiments.runner import discover_configs, run_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", help="Experiment config path or name")
    parser.add_argument(
        "--all", action="store_true",
        help="Run every experiment config found in experiments/",
    )
    args = parser.parse_args(argv)

    if args.all:
        configs = discover_configs(EXPERIMENTS_DIR)
        if not configs:
            print(f"No experiment configs found in {EXPERIMENTS_DIR}")
            return 1
    elif args.config:
        configs = [resolve_config_path(args.config)]
    else:
        parser.print_help()
        return 1

    exit_code = 0
    for cfg in configs:
        print(f"\n=== Running experiment: {cfg} ===")
        try:
            run_experiment(cfg)
        except ConfigError as exc:
            print(f"[error] {exc}")
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
