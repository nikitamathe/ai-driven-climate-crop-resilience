"""Tests for E08 experiment configuration, artifact logging, and CI helpers."""

from __future__ import annotations

import json
import yaml

import pytest

from src.experiments.config import (
    ConfigError,
    config_hash,
    load_config,
    resolve_config_path,
)
from src.experiments.runner import ARTIFACT_FILES, git_commit_hash, run_experiment
from src.data.loader import PROJECT_ROOT

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
BASE_CONFIG = EXPERIMENTS_DIR / "base.yaml"


# ------------------------------------------------------------------ config load

class TestLoadConfig:
    def test_base_config_loads(self):
        cfg = load_config(BASE_CONFIG)
        assert cfg["name"] == "base-random-forest"
        assert cfg["seed"] == 42
        assert cfg["split_cutoff"] == 2014
        assert cfg["model"] == "random_forest"
        assert isinstance(cfg["params"], dict)

    def test_missing_file(self):
        with pytest.raises(ConfigError, match="config not found"):
            load_config(EXPERIMENTS_DIR / "does_not_exist.yaml")

    def test_invalid_yaml(self, tmp_path):
        p = tmp_path / "bad.yaml"
        p.write_text("name: [unclosed\n  - boom")
        with pytest.raises(ConfigError, match="invalid YAML"):
            load_config(p)

    def test_missing_required_key(self, tmp_path):
        p = tmp_path / "no_seed.yaml"
        p.write_text(yaml.safe_dump({"name": "x", "model": "random_forest", "split_cutoff": 2014, "params": {}}))
        with pytest.raises(ConfigError, match="missing required keys"):
            load_config(p)

    def test_unsupported_model(self, tmp_path):
        p = tmp_path / "bad_model.yaml"
        p.write_text(yaml.safe_dump({"name": "x", "seed": 1, "split_cutoff": 2014, "model": "xgboost", "params": {}}))
        with pytest.raises(ConfigError, match="unsupported model"):
            load_config(p)

    def test_non_mapping_config(self, tmp_path):
        p = tmp_path / "scalar.yaml"
        p.write_text("just a string")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(p)


# ------------------------------------------------------------------ config hash

class TestConfigHash:
    def test_deterministic(self):
        cfg = load_config(BASE_CONFIG)
        assert config_hash(cfg) == config_hash(cfg)

    def test_changes_with_params(self):
        base = load_config(BASE_CONFIG)
        alt = dict(base, params={"n_estimators": 500, "max_depth": 20})
        assert config_hash(base) != config_hash(alt)

    def test_changes_with_seed(self):
        base = load_config(BASE_CONFIG)
        alt = dict(base, seed=99)
        assert config_hash(base) != config_hash(alt)

    def test_insensitive_to_key_order(self):
        base = load_config(BASE_CONFIG)
        reordered = {k: base[k] for k in reversed(list(base.keys()))}
        assert config_hash(base) == config_hash(reordered)

    def test_is_short_hex(self):
        h = config_hash(load_config(BASE_CONFIG))
        assert len(h) == 8
        int(h, 16)  # must be hex


# ------------------------------------------------------------------ path resolve

class TestResolveConfigPath:
    def test_name_resolves_to_yaml(self):
        assert resolve_config_path("base") == BASE_CONFIG

    def test_relative_yaml(self):
        assert resolve_config_path("base.yaml") == BASE_CONFIG

    def test_absolute_path(self, tmp_path):
        p = tmp_path / "custom.yaml"
        p.write_text("name: x\nseed: 1\nsplit_cutoff: 2014\nmodel: random_forest\nparams: {}\n")
        assert resolve_config_path(str(p)) == p


# ------------------------------------------------------------------ git hash

class TestGitCommitHash:
    def test_returns_string(self):
        h = git_commit_hash()
        assert isinstance(h, str)
        assert h != ""


# ------------------------------------------------------------------ checksum helpers

class TestManifestChecksum:
    def test_compute_and_assert_roundtrip(self, tmp_path, monkeypatch):
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        (raw_dir / "a.csv").write_bytes(b"a" * 100)
        (raw_dir / "b.csv").write_bytes(b"b" * 200)

        import hashlib
        from scripts import check_data_checksums as chk
        monkeypatch.setattr(chk, "RAW_DIR", raw_dir)
        monkeypatch.setattr(chk, "WATCHED_FILES", ["a.csv", "b.csv"])

        chk.write_manifest()
        assert (raw_dir / "manifest.json").exists()
        assert chk.main(["--assert"]) == 0

        # Corrupt a file -> assert fails
        (raw_dir / "a.csv").write_bytes(b"x" * 100)
        assert chk.main(["--assert"]) == 1


# ------------------------------------------------------------------ runner integration

class TestRunExperiment:
    def test_writes_artifact_bundle(self, tmp_path):
        out_root = tmp_path / "experiments"
        result = run_experiment(BASE_CONFIG, out_root=out_root)

        run_dir = out_root / result["config_hash"]
        assert run_dir.is_dir()
        for fname in ARTIFACT_FILES:
            assert (run_dir / fname).exists(), f"missing artifact {fname}"

        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["config_hash"] == result["config_hash"]
        assert metrics["experiment"] == "base-random-forest"
        assert "metrics" in metrics and "r2" in metrics["metrics"]
        assert metrics["git_commit"]

        params = json.loads((run_dir / "params.json").read_text())
        assert params["model"] == "random_forest"
        assert params["fitted_params"]["max_depth"] == 15

        # Structural read-back: model.joblib must reload as an estimator
        import joblib
        model = joblib.load(run_dir / "model.joblib")
        assert hasattr(model, "predict")
