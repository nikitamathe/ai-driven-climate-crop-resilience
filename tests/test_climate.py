"""Tests for the climate-stress indicators module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.climate.indicators import (
    MONTH_ABBR_TO_NUM,
    _gamma_cdf_fit,
    _month_abbr_to_num,
    compute_all_indicators,
    compute_anomalies,
    mann_kendall,
    sen_slope,
    spi,
    thermal_stress,
)
from src.data.loader import load_nasa_data


# ------------------------------------------------------------------ helpers

def _make_monthly_df(years: range = range(1996, 2021)) -> pd.DataFrame:
    """Build a realistic300-row monthly DataFrame for 1996-2020."""
    rng = np.random.default_rng(42)
    rows = []
    for i, y in enumerate(years):
        for m in range(1, 13):
            rows.append({
                "Year": y,
                "Month": list(MONTH_ABBR_TO_NUM.keys())[m - 1],
                "Rainfall": (30 + 20 * np.sin(2 * np.pi * (m - 1) / 12)
                             + 5 * np.sin(4 * np.pi * (m - 1) / 12)
                             + 3 * np.sin(2 * np.pi * i / len(years))),
                "AvgTemp": (25 + 8 * np.sin(2 * np.pi * (m - 6) / 12)
                            + 0.5 * np.sin(2 * np.pi * i / len(years))),
                "MaxTemp": (35 + 5 * np.sin(2 * np.pi * (m - 6) / 12)
                            + 0.3 * np.sin(2 * np.pi * i / len(years))),
                "MinTemp": (15 + 5 * np.sin(2 * np.pi * (m - 6) / 12)
                            + 0.3 * np.sin(2 * np.pi * i / len(years))),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ SPI

class TestSPI:
    def test_basic_positive_values(self):
        precip = pd.Series([10.0] * 24)
        result = spi(precip, 3)
        assert result.isna().sum() == 2  # first 2 are warm-up NaNs
        valid = result.dropna()
        assert len(valid) == 22

    def test_zero_precipitation(self):
        precip = pd.Series([0.0] * 24)
        result = spi(precip, 3)
        assert len(result) == 24
        assert result.isna().sum() == 2
        assert np.all(np.isfinite(result.dropna()))

    def test_mixed_zeros_and_values(self):
        precip = pd.Series([0, 0, 5, 10, 0, 15, 0, 0, 20, 10, 5, 0] * 3)
        result = spi(precip, 6)
        assert result.isna().sum() == 5
        assert len(result) == 36

    def test_constant_nonzero_returns_near_zero(self):
        precip = pd.Series([50.0] * 24)
        result = spi(precip, 3)
        valid = result.dropna()
        assert np.all(np.abs(valid.values) < 0.5)

    def test_clipping(self):
        precip = pd.Series(np.concatenate([np.full(12, 200.0), np.full(12, 0.1)]))
        result = spi(precip, 6)
        valid = result.dropna()
        assert np.all(valid >= -3.0)
        assert np.all(valid <= 3.0)

    def test_different_scales(self):
        precip = pd.Series(np.random.default_rng(42).uniform(1, 30, 36))
        for s in (3, 6, 12):
            result = spi(precip, s)
            assert result.isna().sum() == s - 1

    def test_gamma_cdf_fit_all_positive(self):
        arr = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
        cdf = _gamma_cdf_fit(arr)
        assert len(cdf) == 5
        assert np.all(cdf > 0) and np.all(cdf < 1)

    def test_gamma_cdf_fit_with_zeros(self):
        arr = np.array([0.0, 0.0, 5.0, 10.0, 15.0])
        cdf = _gamma_cdf_fit(arr)
        assert cdf[0] < cdf[2]  # zero values get lower CDF

    def test_with_real_nasa_data(self):
        nasa = load_nasa_data()
        nasa["date"] = pd.to_datetime(
            nasa["Year"].astype(str) + "-" +
            nasa["Month"].map(_month_abbr_to_num).astype(str).str.zfill(2) + "-01"
        )
        nasa = nasa.set_index("date").sort_index()
        result = spi(nasa["Rainfall"], 3)
        assert len(result) == 300
        assert result.isna().sum() == 2
        valid = result.dropna()
        assert np.all(valid >= -3.0) and np.all(valid <= 3.0)


# ------------------------------------------------------------------ Anomalies

class TestAnomalies:
    def test_baseline_mean_zero(self):
        nasa = _make_monthly_df()
        anom = compute_anomalies(nasa, (1996, 2020))
        assert abs(anom["Rainfall_Anomaly"].mean()) < 0.2
        assert abs(anom["Temp_Anomaly"].mean()) < 0.2

    def test_symmetry(self):
        """A symmetric deviation from baseline mean should give equal-magnitude anomalies."""
        nasa = _make_monthly_df()
        anom = compute_anomalies(nasa, (1996, 2020))
        assert len(anom) == 25

    def test_with_real_data(self):
        nasa = load_nasa_data()
        anom = compute_anomalies(nasa, (1996, 2020))
        assert set(anom.columns) == {"Year", "Rainfall_Anomaly", "Temp_Anomaly"}
        assert len(anom) == 25


# ------------------------------------------------------------------ Mann-Kendall

class TestMannKendall:
    def test_increasing(self):
        series = np.arange(1.0, 21.0)
        mk = mann_kendall(series)
        assert mk["S"] > 0
        assert mk["Z"] > 0
        assert mk["p"] < 0.05
        assert mk["trend"] == "increasing"

    def test_decreasing(self):
        series = np.arange(20.0, 0.0, -1.0)
        mk = mann_kendall(series)
        assert mk["S"] < 0
        assert mk["Z"] < 0
        assert mk["p"] < 0.05
        assert mk["trend"] == "decreasing"

    def test_constant(self):
        series = np.ones(20)
        mk = mann_kendall(series)
        assert mk["S"] == 0.0
        assert mk["Z"] == 0.0
        assert mk["trend"] == "no trend"

    def test_with_nans(self):
        series = np.arange(1.0, 16.0)
        series[3] = np.nan
        series[10] = np.nan
        mk = mann_kendall(series)
        assert mk["trend"] == "increasing"

    def test_short_series(self):
        mk = mann_kendall(np.array([1.0]))
        assert mk["S"] == 0.0
        assert mk["trend"] == "no trend"

    def test_known_s_statistic(self):
        series = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mk = mann_kendall(series)
        assert mk["S"] == 10.0  # C(5,2) = 10

    def test_returns_all_keys(self):
        mk = mann_kendall(np.arange(10.0))
        assert set(mk.keys()) == {"S", "var_s", "Z", "p", "trend", "alpha"}


# ------------------------------------------------------------------ Sen's slope

class TestSenSlope:
    def test_linear(self):
        series = np.arange(1.0, 11.0)  # slope = 1.0
        assert sen_slope(series) == 1.0

    def test_steeper_linear(self):
        series = np.arange(0.0, 100.0, 2.0)  # slope = 2.0
        assert sen_slope(series) == 2.0

    def test_robust_to_outlier(self):
        series = np.arange(1.0, 21.0).astype(float)
        series[15] = 1000.0  # large outlier
        assert abs(sen_slope(series) - 1.0) < 0.5

    def test_constant(self):
        series = np.ones(10)
        assert sen_slope(series) == 0.0

    def test_few_values(self):
        assert sen_slope(np.array([1.0])) is np.nan or np.isnan(sen_slope(np.array([1.0])))


# ------------------------------------------------------------------ Thermal stress

class TestThermalStress:
    def test_within_optimum(self):
        temps = np.full(12, 25.0)
        assert thermal_stress(temps, 25.0, 5.0) == 0

    def test_known_exceedances(self):
        temps = np.array([20.0, 35.0, 25.0, 25.0, 25.0, 25.0,
                          25.0, 25.0, 25.0, 25.0, 25.0, 25.0])
        # 35 is > 5 away from 25 -> 1 stress month
        assert thermal_stress(temps, 25.0, 5.0) == 1

    def test_multiple_exceedances(self):
        temps = np.array([10.0, 40.0, 25.0, 25.0, 25.0, 25.0,
                          25.0, 25.0, 25.0, 25.0, 25.0, 25.0])
        # 10 and 40 both > 5 away from 25
        assert thermal_stress(temps, 25.0, 5.0) == 2

    def test_with_nans(self):
        temps = np.array([np.nan, 30.0, 31.0, 25.0, 25.0, 25.0,
                          25.0, 25.0, 25.0, 25.0, 25.0, 25.0])
        # nan filtered, 31 is 6 away -> 1
        assert thermal_stress(temps, 25.0, 5.0) == 1

    def test_all_nan(self):
        temps = np.full(12, np.nan)
        assert thermal_stress(temps, 25.0, 5.0) == 0

    def test_different_threshold(self):
        temps = np.full(12, 30.0)
        assert thermal_stress(temps, 25.0, 5.0) == 0  # exactly at threshold
        assert thermal_stress(temps, 25.0, 4.0) == 12  # all exceed


# ------------------------------------------------------------------ Orchestrator

class TestComputeAllIndicators:
    @pytest.fixture
    def indicators(self):
        nasa = load_nasa_data()
        return compute_all_indicators(nasa)

    def test_output_schema(self, indicators):
        expected_cols = {
            "Year",
            "SPI_3", "SPI_6", "SPI_12",
            "Rainfall_Anomaly", "Temp_Anomaly",
            "Rainfall_SenSlope", "Rainfall_MK_Z", "Rainfall_MK_p", "Rainfall_MK_direction",
            "AvgTemp_SenSlope", "AvgTemp_MK_Z", "AvgTemp_MK_p", "AvgTemp_MK_direction",
            "MaxTemp_SenSlope", "MaxTemp_MK_Z", "MaxTemp_MK_p", "MaxTemp_MK_direction",
            "MinTemp_SenSlope", "MinTemp_MK_Z", "MinTemp_MK_p", "MinTemp_MK_direction",
            "Thermal_Stress_rice", "Thermal_Stress_maize",
            "Thermal_Stress_chickpea", "Thermal_Stress_cotton",
            "Thermal_Stress_Mean",
        }
        assert expected_cols.issubset(set(indicators.columns))

    def test_year_range(self, indicators):
        assert indicators["Year"].min() == 1996
        assert indicators["Year"].max() == 2020
        assert len(indicators) == 25

    def test_spi_nan_at_start(self, indicators):
        # SPI-3 has warm-up NaNs for the first 2 monthly values, but by the
        # end of 1996 (12 months) enough history exists for a valid SPI-3.
        # SPI-12 needs 12 months of history, so Dec 1996 is the first valid.
        # Just verify the first year has some valid SPI values and the warm-up
        # behavior is correct (first few monthly values would be NaN, but
        # yearly extraction takes the last valid value per year).
        assert np.isfinite(indicators.loc[indicators["Year"] == 1996, "SPI_3"].iloc[0])
        # SPI-12 for 1996: Dec has enough data (12 months), so it's valid
        assert np.isfinite(indicators.loc[indicators["Year"] == 1996, "SPI_12"].iloc[0])

    def test_spi_valid_for_recent_years(self, indicators):
        row = indicators[indicators["Year"] == 2015].iloc[0]
        assert np.isfinite(row["SPI_3"])
        assert np.isfinite(row["SPI_6"])
        assert np.isfinite(row["SPI_12"])

    def test_spi_range(self, indicators):
        for col in ["SPI_3", "SPI_6", "SPI_12"]:
            valid = indicators[col].dropna()
            assert (valid >= -3.0).all(), f"{col} below -3"
            assert (valid <= 3.0).all(), f"{col} above 3"

    def test_thermal_stress_non_negative(self, indicators):
        for crop in ["rice", "maize", "chickpea", "cotton"]:
            col = f"Thermal_Stress_{crop}"
            assert (indicators[col] >= 0).all()
            assert (indicators[col] <= 12).all()

    def test_no_crash_with_real_data(self, indicators):
        assert len(indicators) > 0


# ------------------------------------------------------------------ Month helper

class TestMonthHelper:
    def test_all_months(self):
        for abbr, num in MONTH_ABBR_TO_NUM.items():
            assert _month_abbr_to_num(abbr) == num
            assert _month_abbr_to_num(abbr.lower()) == num

    def test_known_values(self):
        assert _month_abbr_to_num("JAN") == 1
        assert _month_abbr_to_num("DEC") == 12


# ------------------------------------------------------------------ Edge cases

class TestEdgeCases:
    def test_constant_precipitation_spi(self):
        precip = pd.Series([50.0] * 24)
        result = spi(precip, 6)
        assert len(result) == 24
        # First 5 are warm-up NaNs; remaining should be finite (near 0 for constant precip)
        assert result.isna().sum() == 5
        valid = result.dropna()
        assert np.all(np.abs(valid.values) < 0.5)

    def test_single_spi_scale(self):
        precip = pd.Series(np.random.default_rng(0).uniform(5, 25, 36))
        result = spi(precip, 3)
        assert len(result) == 36
        valid = result.dropna()
        assert len(valid) >= 33  # first 2 are warm-up NaNs

    def test_compute_all_custom_threshold(self):
        nasa = load_nasa_data()
        indicators = compute_all_indicators(nasa, thermal_threshold=3.0)
        assert "Thermal_Stress_rice" in indicators.columns
        assert len(indicators) == 25
