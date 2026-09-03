"""Tests for E04 SHAP-based model explainability.

All tests use a small synthetic merged DataFrame and small tree ensembles so
they run in seconds and never touch the real dataset during unit testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.explain import (
    compute_shap,
    consistency_report,
    explain_models,
    mean_abs_shap,
    permutation_importance,
    write_shap_outputs,
)
from src.features.engineering import default_features


# ---------------------------------------------------------------------------
# Small synthetic frame with a clear feature->target signal
# ---------------------------------------------------------------------------

def _make_merged_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2000, 2008):
        for _ in range(3):
            rows.append({
                "Year": year,
                "Area_ha": rng.uniform(50, 200),
                "pH": rng.uniform(5.5, 7.5),
                "Temperature_C": 25.0,
                "Humidity_%": 80.0,
                "Rainfall_mm": 1200.0,
                "Wind_Speed_m_s": 2.0,
                "Solar_Radiation_MJ_m2_day": 18.0,
                "AvgTemp": 25.0 + 0.2 * (year - 2000),
                "MaxTemp": 30.0 + 0.2 * (year - 2000),
                "MinTemp": 15.0,
                "Yield_kg_per_ha": (
                    2000 + 60 * (year - 2000) + 3 * rng.uniform(0, 1)
                ),
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def Xy():
    merged = _make_merged_df()
    features = default_features()
    from src.features.engineering import build_xy
    X, y = build_xy(merged, features, "Yield_kg_per_ha")
    # holdout: rows from the last year
    last_year = merged["Year"].max()
    train_mask = merged["Year"] < last_year
    return (
        X[train_mask], y[train_mask],
        X[~train_mask], y[~train_mask],
        features,
    )


def _fit_small_models(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=20, max_depth=4, random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return {"tuned_rf": rf}


# ---------------------------------------------------------------------------
# SHAP values
# ---------------------------------------------------------------------------

class TestComputeShap:
    def test_values_shape(self, Xy):
        X_train, y_train, X_test, _, features = Xy
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
        m.fit(X_train, y_train)
        result = compute_shap(m, X_test)
        assert result["values"].shape == (len(X_test), len(features))
        assert np.ndim(result["expected_value"]) == 0
        assert result["feature_names"] == features

    def test_sum_approx_explains_prediction(self, Xy):
        X_train, y_train, X_test, _, _ = Xy
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        result = compute_shap(m, X_test)
        # base + sum(shap) == prediction (TreeExplainer is additive)
        reconstructed = result["expected_value"] + result["values"].sum(axis=1)
        assert np.allclose(reconstructed, preds, atol=1e-4)


class TestMeanAbsShap:
    def test_keys_match_features(self, Xy):
        X_train, _, X_test, _, features = Xy
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
        m.fit(X_train, Xy[1])
        result = compute_shap(m, X_test)
        mas = mean_abs_shap(result)
        assert set(mas.keys()) == set(features)
        assert all(v >= 0 for v in mas.values())

    def test_explicit_feature_names(self, Xy):
        result = {"values": np.ones((2, 2)), "expected_value": 0.0, "feature_names": ["a", "b"]}
        mas = mean_abs_shap(result, ["a", "b"])
        assert mas == {"a": 1.0, "b": 1.0}


class TestPermutationImportance:
    def test_keys_and_nonnegative(self, Xy):
        X_train, y_train, X_test, y_test, features = Xy
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
        m.fit(X_train, y_train)
        pi = permutation_importance(m, X_test, y_test.to_numpy(), n_repeats=2)
        assert set(pi.keys()) == set(features)
        assert all(np.isfinite(v) for v in pi.values())


class TestExplainModels:
    def test_returns_per_model(self, Xy):
        X_train, y_train, X_test, y_test, _ = Xy
        models = _fit_small_models(X_train, y_train)
        results = explain_models(
            models, X_test, y_test.to_numpy(), n_repeats=2
        )
        assert set(results.keys()) == {"tuned_rf"}
        res = results["tuned_rf"]
        assert "shap" in res and "mean_abs_shap" in res and "permutation_importance" in res
        assert set(res["mean_abs_shap"].keys()) == set(res["permutation_importance"].keys())

    def test_no_background_needed(self, Xy):
        """tree_path_dependent SHAP needs no reference distribution, so there
        is no background set that could leak test-year information."""
        from src.models.explain import compute_shap
        X_train, y_train, X_test, _, _ = Xy
        from sklearn.ensemble import RandomForestRegressor
        m = RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42)
        m.fit(X_train, y_train)
        result = compute_shap(m, X_test)
        assert "expected_value" in result
        assert result["values"].size > 0


class TestConsistencyReport:
    def test_returns_string_with_model_names(self, Xy):
        X_train, y_train, X_test, y_test, features = Xy
        models = _fit_small_models(X_train, y_train)
        results = explain_models(
            models, X_test, y_test.to_numpy(), n_repeats=2
        )
        report = consistency_report(results, top_n=3)
        assert isinstance(report, str)
        assert "tuned_rf" in report
        assert "mean |SHAP|" in report.lower() or "mean|shap|" in report.lower() or "|shap|" in report.lower()


class TestWriteOutputs:
    def test_writes_csv_and_report(self, Xy, tmp_path):
        X_train, y_train, X_test, y_test, _ = Xy
        models = _fit_small_models(X_train, y_train)
        results = explain_models(
            models, X_test, y_test.to_numpy(), n_repeats=2
        )
        paths = write_shap_outputs(results, tmp_path)
        files = {p.name for p in paths}
        assert "feature_shap_summary.csv" in files
        assert "consistency_report.txt" in files
        # per-model dirs
        assert any(p.parent.name == "tuned_rf" for p in paths)
        assert (tmp_path / "feature_shap_summary.csv").exists()

    def test_writes_plots(self, Xy, tmp_path):
        X_train, y_train, X_test, y_test, _ = Xy
        models = _fit_small_models(X_train, y_train)
        results = explain_models(
            models, X_test, y_test.to_numpy(), n_repeats=2
        )
        write_shap_outputs(results, tmp_path)
        model_dir = tmp_path / "tuned_rf"
        for name in ("beeswarm.png", "dependence_avgtemp.png", "dependence_rainfall_mm.png",
                     "shap_vs_permutation_importance.png"):
            p = model_dir / name
            assert p.exists() and p.stat().st_size > 0


# ---------------------------------------------------------------------------
# Lazy-import guarantee (protects E01-E10 + core pipeline)
# ---------------------------------------------------------------------------

class TestLazyImport:
    def test_core_modules_import_without_shap_entrypoints(self):
        """Importing explain + shap_plots must not fail even if shap usage
        is only triggered lazily at call time."""
        import importlib
        import src.models.explain as ex
        import src.visualization.shap_plots as sp
        assert callable(ex.compute_shap)
        assert callable(sp.plot_beeswarm)
