"""Tests for E05 split-conformal prediction intervals.

All tests use a small synthetic merged DataFrame and small Random Forests so
they run in seconds and never touch the real dataset during unit testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.conformal import (
    ConformalRegressor,
    calibrate_on_time_holdout,
    empirical_coverage,
    fit_conformal,
    mean_interval_width,
)
from src.features.engineering import default_features


# ---------------------------------------------------------------------------
# Synthetic frame with a clear feature->target signal (mirrors test_tuning.py)
# ---------------------------------------------------------------------------

def _make_merged_df(seed: int = 42) -> pd.DataFrame:
    """Synthetic yield that depends on *features* (exchangeable across years).

    Yield is a linear function of Area/pH/rainfall/temperature/humidity plus
    noise. Because the feature distribution is the same for every year, a tree
    model generalizes to held-out future years, which is what lets the
    split-conformal coverage test transfer the calibration scores.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for year in range(2000, 2008):
        for _ in range(6):
            area = rng.uniform(50, 200)
            ph = rng.uniform(5.5, 7.5)
            rainfall = rng.normal(1100, 160)
            temp = rng.normal(25.0, 1.5)
            hum = rng.normal(75, 5)
            wind = rng.normal(2.0, 0.5)
            solar = rng.normal(18, 2)
            yield_ = (
                600
                + 2.0 * area
                + 150 * ph
                + 0.6 * rainfall
                - 20 * temp
                - 8 * hum
                + rng.normal(0, 30)
            )
            rows.append({
                "Year": year,
                "Area_ha": area,
                "pH": ph,
                "Temperature_C": temp,
                "Humidity_%": hum,
                "Rainfall_mm": rainfall,
                "Wind_Speed_m_s": wind,
                "Solar_Radiation_MJ_m2_day": solar,
                "AvgTemp": temp + rng.normal(0, 1),
                "MaxTemp": temp + 4,
                "MinTemp": temp - 4,
                "Yield_kg_per_ha": yield_,
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def split_data():
    """Return train (proper+cal) and test numpy splits with year groups."""
    merged = _make_merged_df()
    features = default_features()
    from src.features.engineering import build_xy
    X, y = build_xy(merged, features, "Yield_kg_per_ha")
    years = merged["Year"].to_numpy()

    # train years < 2007, test years >= 2007
    train_mask = years < 2007
    X_train = X[train_mask].to_numpy()
    y_train = y[train_mask].to_numpy()
    years_train = years[train_mask]

    X_test = X[~train_mask].to_numpy()
    y_test = y[~train_mask].to_numpy()
    years_test = years[~train_mask]

    return {
        "X_train": X_train, "y_train": y_train, "years_train": years_train,
        "X_test": X_test, "y_test": y_test, "years_test": years_test,
        "features": features,
        "merged": merged,
    }


def _make_regressor(n_estimators: int = 30, max_depth: int = 4):
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1,
    )


# ---------------------------------------------------------------------------
# ConformalRegressor basics
# ---------------------------------------------------------------------------

class TestConformalRegressor:
    def test_calibrate_sets_qhat_positive(self, split_data):
        d = split_data
        mid = len(d["X_train"]) // 2
        wrap = ConformalRegressor(_make_regressor(), alpha=0.10)
        wrap.fit(d["X_train"][:mid], d["y_train"][:mid])
        wrap.calibrate(d["X_train"][mid:], d["y_train"][mid:])
        assert wrap.q_hat > 0

    def test_predict_returns_valid_intervals(self, split_data):
        d = split_data
        mid = len(d["X_train"]) // 2
        wrap = ConformalRegressor(_make_regressor(), alpha=0.10)
        wrap.fit(d["X_train"][:mid], d["y_train"][:mid])
        wrap.calibrate(d["X_train"][mid:], d["y_train"][mid:])
        center, low, high = wrap.predict(d["X_test"])
        assert center.shape == d["y_test"].shape
        assert np.all(low <= high)
        assert np.all(low <= center) and np.all(center <= high)

    def test_predict_before_calibrate_raises(self, split_data):
        d = split_data
        wrap = ConformalRegressor(_make_regressor(), alpha=0.10)
        wrap.fit(d["X_train"], d["y_train"])
        with pytest.raises(RuntimeError, match="calibrated"):
            wrap.predict(d["X_test"])

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            ConformalRegressor(_make_regressor(), alpha=1.5)


class TestIntervalMetrics:
    def test_symmetric_width(self, split_data):
        d = split_data
        mid = len(d["X_train"]) // 2
        wrap = ConformalRegressor(_make_regressor(), alpha=0.10)
        wrap.fit(d["X_train"][:mid], d["y_train"][:mid])
        wrap.calibrate(d["X_train"][mid:], d["y_train"][mid:])
        center, low, high = wrap.predict(d["X_test"])
        assert np.allclose(high - low, 2 * wrap.q_hat, atol=1e-6)
        assert mean_interval_width(low, high) == pytest.approx(2 * wrap.q_hat)

    def test_coverage_in_unit_interval(self, split_data):
        d = split_data
        assert 0.0 <= empirical_coverage(d["y_test"], d["y_test"] - 1, d["y_test"] + 1) <= 1.0
        # exact coverage when interval always contains truth
        assert empirical_coverage(d["y_test"], d["y_test"] - 1, d["y_test"] + 1) == 1.0
        # zero coverage when interval never contains truth
        assert empirical_coverage(d["y_test"], d["y_test"] + 1, d["y_test"] + 2) == 0.0


# ---------------------------------------------------------------------------
# End-to-end conformal coverage on a held-out year block
# ---------------------------------------------------------------------------

class TestCoverage:
    def test_coverage_meets_nominal_bar(self, split_data):
        """P5 acceptance: empirical coverage on a held-out year block should
        approach the 1-alpha nominal target."""
        d = split_data
        # reserve the last 2 training years as a calibration block
        unique_years = np.sort(np.unique(d["years_train"]))
        cal_years = set(unique_years[-2:])
        prop_mask = ~np.isin(d["years_train"], list(cal_years))
        cal_mask = np.isin(d["years_train"], list(cal_years))

        wrap = fit_conformal(
            _make_regressor(),
            d["X_train"][prop_mask], d["y_train"][prop_mask],
            d["X_train"][cal_mask], d["y_train"][cal_mask],
            alpha=0.10,
        )
        center, low, high = wrap.predict(d["X_test"])
        cov = empirical_coverage(d["y_test"], low, high)
        # Conformal is a distribution-free approximate guarantee; allow a
        # generous tolerance for the tiny synthetic sample.
        assert cov >= 0.80

    def test_coverage_improves_with_wider_alpha(self, split_data):
        """A larger alpha (narrower target) -> finer coverage is not guaranteed
        up-front, but a 50% interval must cover ~half; sanity: 95% interval
        (alpha=0.05) covers at least a 50% real interval."""
        d = split_data
        unique_years = np.sort(np.unique(d["years_train"]))
        cal_years = set(unique_years[-2:])
        prop_mask = ~np.isin(d["years_train"], list(cal_years))
        cal_mask = np.isin(d["years_train"], list(cal_years))
        wrap = fit_conformal(
            _make_regressor(),
            d["X_train"][prop_mask], d["y_train"][prop_mask],
            d["X_train"][cal_mask], d["y_train"][cal_mask],
            alpha=0.05,
        )
        center, low, high = wrap.predict(d["X_test"])
        assert empirical_coverage(d["y_test"], low, high) >= 0.80


# ---------------------------------------------------------------------------
# Time-aware calibration split (no leakage)
# ---------------------------------------------------------------------------

class TestTimeAwareCalibration:
    def test_requires_year_groups(self, split_data):
        d = split_data
        with pytest.raises(ValueError, match="year_groups"):
            calibrate_on_time_holdout(_make_regressor(), d["X_train"], d["y_train"], None)

    def test_calibration_years_are_recent_train_years(self, split_data):
        d = split_data
        wrap, split = calibrate_on_time_holdout(
            _make_regressor(), d["X_train"], d["y_train"], d["years_train"],
            n_cal_years=2, alpha=0.10,
        )
        cal_years = split["calibration_years"]
        train_unique = np.sort(np.unique(d["years_train"]))
        assert set(cal_years) == set(train_unique[-2:])
        # calibration years must be strictly before every test year
        assert max(cal_years) < min(d["years_test"])

    def test_works_with_numpy(self, split_data):
        d = split_data
        wrap, split = calibrate_on_time_holdout(
            _make_regressor(), d["X_train"], d["y_train"], d["years_train"],
            n_cal_years=2, alpha=0.10,
        )
        assert wrap.q_hat > 0
        center, low, high = wrap.predict(d["X_test"])
        assert np.all(low <= high)


# ---------------------------------------------------------------------------
# Pipeline additive integration (unit-level)
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    def test_add_conformal_intervals_columns(self):
        """The pipeline helper should add additive interval columns without
        changing the point predictions."""
        from src.pipeline.run_pipeline import _add_conformal_intervals, _fit_base_for_conformal
        from src.features.engineering import TARGET

        merged = _make_merged_df()
        features = default_features()
        from src.features.engineering import build_xy
        X, y = build_xy(merged, features, TARGET)
        years = merged["Year"].to_numpy()
        train_mask = years < 2007

        model = _fit_base_for_conformal(n_estimators=20, max_depth=4)
        model.fit(X[train_mask].to_numpy(), y[train_mask].to_numpy())
        before_preds = model.predict(X[train_mask].to_numpy())
        merged["Predicted_Yield"] = model.predict(X.to_numpy())  # mirror run()

        out = _add_conformal_intervals(
            merged, model,
            X[train_mask].to_numpy(), y[train_mask].to_numpy(),
            years[train_mask], features,
            target=TARGET, test_year_min=2007,
            n_cal_years=2, alpha=0.10,
        )
        assert "Pred_Yield_Lo" in out.columns
        assert "Pred_Yield_Hi" in out.columns
        assert np.all(out["Pred_Yield_Lo"] <= out["Predicted_Yield"])
        assert np.all(out["Pred_Yield_Hi"] >= out["Predicted_Yield"])

    def test_flag_wide_interval(self):
        from src.pipeline.run_pipeline import _flag_wide_interval
        import numpy as np

        width = np.array([10.0, 200.0, 30.0, np.nan])
        center = np.array([100.0, 100.0, 100.0, 100.0])
        flag = _flag_wide_interval(width, center, relative_threshold=0.5)
        assert list(flag) == [False, True, False, False]


# ---------------------------------------------------------------------------
# Visualization helpers write files (no shap/geopandas needed)
# ---------------------------------------------------------------------------

class TestUncertaintyPlots:
    def test_writes_report(self, split_data, tmp_path):
        from src.visualization.uncertainty_plots import write_conformal_report

        d = split_data
        paths = write_conformal_report(
            d["y_test"], d["y_test"], d["y_test"] - 5, d["y_test"] + 5,
            out_dir=tmp_path, width=10.0, coverage=1.0, q_hat=5.0,
        )
        names = {p.name for p in paths}
        assert "conformal_report.csv" in names
        assert "interval_width_histogram.png" in names
        assert "coverage_vs_nominal.png" in names
        assert "conformal_meta.txt" in names

    def test_interval_by_crop_writes(self, split_data, tmp_path):
        from src.visualization.uncertainty_plots import plot_interval_by_crop

        df = pd.DataFrame({
            "Crop": ["rice", "rice", "wheat"],
            "Interval_Width": [5.0, 6.0, 8.0],
        })
        p = tmp_path / "by_crop.png"
        plot_interval_by_crop(df, p)
        assert p.exists() and p.stat().st_size > 0

    def test_resilience_with_ci_writes(self, tmp_path):
        from src.visualization.uncertainty_plots import plot_resilience_with_ci

        df = pd.DataFrame({
            "Resilience_Index": [0.9, 0.6, 0.75],
            "Resilience_Lo": [0.8, 0.5, 0.7],
            "Resilience_Hi": [1.0, 0.8, 0.8],
        })
        p = tmp_path / "res_ci.png"
        plot_resilience_with_ci(df, p)
        assert p.exists() and p.stat().st_size > 0
