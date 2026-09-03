"""Tests for E03 hyperparameter optimization (RandomizedSearchCV, year-grouped CV).

All tuning tests use a small synthetic "merged" DataFrame so they run fast
and never touch the real dataset during unit testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.cross_validation import YearGroupCV
from src.models.tuning import (
    RF_SEARCH_SPACE,
    XGB_SEARCH_SPACE,
    compare_models,
    tune_random_forest,
    tune_xgboost,
)
from src.features.engineering import default_features


# ---------------------------------------------------------------------------
# Small synthetic merged frame (years 2000-2006, ~3 rows/year)
# ---------------------------------------------------------------------------

def _make_merged_df(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2000, 2007):
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
                "Yield_kg_per_ha": 2000 + 50 * (year - 2000) + rng.normal(0, 10),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def merged():
    return _make_merged_df()


@pytest.fixture
def Xy(merged):
    features = default_features()
    from src.features.engineering import build_xy
    X, y = build_xy(merged, features, "Yield_kg_per_ha")
    return X, y, merged["Year"].to_numpy()


# ---------------------------------------------------------------------------
# Year-grouped CV
# ---------------------------------------------------------------------------

class TestYearGroupCV:
    def test_no_years_overlap_between_folds(self, merged):
        """Training and test year sets must be disjoint in every fold."""
        cv = YearGroupCV(n_test_years=1, minimum_train_years=3)
        for train_idx, test_idx in cv.split(merged):
            train_years = set(merged.iloc[train_idx]["Year"])
            test_years = set(merged.iloc[test_idx]["Year"])
            assert train_years.isdisjoint(test_years)

    def test_test_years_are_later_than_train_years(self, merged):
        """Expanding window: test years must be strictly greater than train years."""
        cv = YearGroupCV(n_test_years=1, minimum_train_years=3)
        for train_idx, test_idx in cv.split(merged):
            train_max = merged.iloc[train_idx]["Year"].max()
            test_min = merged.iloc[test_idx]["Year"].min()
            assert test_min > train_max

    def test_n_folds_correct(self, merged):
        """With 7 years, min_train=3, n_test=1 -> 7-3-1+1 = 4 folds."""
        cv = YearGroupCV(n_test_years=1, minimum_train_years=3)
        n_folds = cv.get_n_splits(merged)
        assert n_folds == 4
        assert len(list(cv.split(merged))) == n_folds

    def test_multi_year_test_window(self, merged):
        """n_test_years=2 tests on a 2-year window in each fold."""
        cv = YearGroupCV(n_test_years=2, minimum_train_years=3)
        for train_idx, test_idx in cv.split(merged):
            test_years = set(merged.iloc[test_idx]["Year"])
            assert len(test_years) == 2

    def test_year_column_or_groups(self, merged):
        """Either the Year column or the groups parameter must work."""
        cv = YearGroupCV(n_test_years=1, minimum_train_years=3)
        X = merged.drop(columns=["Year"])
        with pytest.raises((TypeError, ValueError)):
            list(cv.split(X))
        # Passing groups explicitly works even without a Year column
        n_years_before = cv.get_n_splits(X, groups=merged["Year"].to_numpy())
        assert n_years_before == 4

    def test_invalid_constructor_args(self):
        with pytest.raises(ValueError):
            YearGroupCV(n_test_years=0)
        with pytest.raises(ValueError):
            YearGroupCV(minimum_train_years=0)


# ---------------------------------------------------------------------------
# Search-space validation
# ---------------------------------------------------------------------------

class TestSearchSpaces:
    def test_rf_space_keys(self):
        assert set(RF_SEARCH_SPACE) == {"n_estimators", "max_depth", "min_samples_split"}
        assert all(isinstance(k, list) for k in RF_SEARCH_SPACE.values())
        assert set(RF_SEARCH_SPACE["n_estimators"]) <= {100, 200, 300, 500}
        assert None in RF_SEARCH_SPACE["max_depth"] or 10 in RF_SEARCH_SPACE["max_depth"]
        assert 2 in RF_SEARCH_SPACE["min_samples_split"]

    def test_xgb_space_keys(self):
        assert set(XGB_SEARCH_SPACE) == {
            "learning_rate", "max_depth", "subsample", "colsample_bytree",
        }
        # learning_rate (the sklearn/xgb modern name for eta)
        assert 0 < min(XGB_SEARCH_SPACE["learning_rate"])
        assert max(XGB_SEARCH_SPACE["learning_rate"]) <= 1.0
        assert all(0 < v <= 1 for v in XGB_SEARCH_SPACE["subsample"])
        assert all(0 < v <= 1 for v in XGB_SEARCH_SPACE["colsample_bytree"])


# ---------------------------------------------------------------------------
# RF tuning
# ---------------------------------------------------------------------------

class TestTuneRandomForest:
    def test_returns_best_params_and_model(self, Xy, merged):
        X, y, year_groups = Xy
        result = tune_random_forest(
            X, y, year_groups=year_groups, n_iter=5, cv_n_test_years=1, random_state=42,
        )
        assert "best_params" in result
        assert set(result["best_params"]) <= set(RF_SEARCH_SPACE)
        assert hasattr(result["model"], "predict")
        # Retrained on all data
        assert result["model"].n_features_in_ == X.shape[1]

    def test_search_results_rows_match_n_iter(self, Xy, merged):
        X, y, year_groups = Xy
        result = tune_random_forest(
            X, y, year_groups=year_groups, n_iter=5, cv_n_test_years=1, random_state=42,
        )
        assert len(result["search_results"]) == 5

    def test_deterministic_given_seed(self, Xy):
        X, y, year_groups = Xy
        a = tune_random_forest(X, y, year_groups=year_groups, n_iter=5, cv_n_test_years=1, random_state=7)
        b = tune_random_forest(X, y, year_groups=year_groups, n_iter=5, cv_n_test_years=1, random_state=7)
        assert a["best_params"] == b["best_params"]

    def test_requires_year_groups(self, Xy):
        X, y, _ = Xy
        with pytest.raises(ValueError, match="year_groups"):
            tune_random_forest(X, y, n_iter=5, cv_n_test_years=1, random_state=42)


# ---------------------------------------------------------------------------
# XGBoost tuning
# ---------------------------------------------------------------------------

class TestTuneXGBoost:
    def test_returns_best_params_and_model(self, Xy, merged):
        X, y, year_groups = Xy
        result = tune_xgboost(
            X, y, year_groups=year_groups, n_iter=5, cv_n_test_years=1, random_state=42,
        )
        assert "best_params" in result
        assert set(result["best_params"]) <= set(XGB_SEARCH_SPACE)
        assert hasattr(result["model"], "predict")
        assert result["model"].n_features_in_ == X.shape[1]

    def test_search_results_rows_match_n_iter(self, Xy):
        X, y, year_groups = Xy
        result = tune_xgboost(
            X, y, year_groups=year_groups, n_iter=5, cv_n_test_years=1, random_state=42,
        )
        assert len(result["search_results"]) == 5

    def test_learning_rate_is_sampled(self, Xy):
        X, y, year_groups = Xy
        result = tune_xgboost(
            X, y, year_groups=year_groups, n_iter=5, cv_n_test_years=1, random_state=42,
        )
        assert "learning_rate" in result["best_params"]

    def test_requires_year_groups(self, Xy):
        X, y, _ = Xy
        with pytest.raises(ValueError, match="year_groups"):
            tune_xgboost(X, y, n_iter=5, cv_n_test_years=1, random_state=42)


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_all_four_models(self, merged):
        results = compare_models(
            merged, n_iter=3, cv_n_test_years=1, random_state=42,
        )
        assert set(results.keys()) == {
            "default_rf", "tuned_rf", "default_xgb", "tuned_xgb",
        }

    def test_metrics_present_and_finite(self, merged):
        results = compare_models(
            merged, n_iter=3, cv_n_test_years=1, random_state=42,
        )
        for name, m in results.items():
            assert "r2" in m and "rmse" in m and "mae" in m
            assert np.isfinite(m["r2"])
            assert np.isfinite(m["rmse"])
            assert np.isfinite(m["mae"])

    def test_best_params_populated_for_tuned(self, merged):
        results = compare_models(
            merged, n_iter=3, cv_n_test_years=1, random_state=42,
        )
        assert results["tuned_rf"]["best_params"]
        assert results["tuned_xgb"]["best_params"]

    def test_models_have_predict(self, merged):
        results = compare_models(
            merged, n_iter=3, cv_n_test_years=1, random_state=42,
        )
        for name, m in results.items():
            assert hasattr(m["model"], "predict"), f"{name} missing predict"


# ---------------------------------------------------------------------------
# Tuned experiment configurations
# ---------------------------------------------------------------------------

class TestTunedExperimentConfig:
    @pytest.mark.parametrize("name", ["tuned_rf", "tuned_xgb"])
    def test_tuned_config_loads(self, name):
        from src.data.loader import PROJECT_ROOT
        from src.experiments.config import load_config

        path = PROJECT_ROOT / "experiments" / f"{name}.yaml"
        cfg = load_config(path)
        assert cfg["model"] in ("random_forest", "xgboost")
        assert isinstance(cfg["params"], dict)
        assert cfg["seed"] == 42

    def test_tuned_xgb_supported(self):
        """xgboost must be accepted by the config loader (E03)."""
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import yaml
        from src.experiments.config import load_config

        with TemporaryDirectory() as td:
            p = Path(td) / "xgb.yaml"
            p.write_text(yaml.safe_dump({
                "name": "x", "seed": 1, "split_cutoff": 2014,
                "model": "xgboost", "params": {"n_estimators": 100},
            }))
            cfg = load_config(p)
            assert cfg["model"] == "xgboost"
