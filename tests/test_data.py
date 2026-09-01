"""Tests for the data loading and aggregation layer."""

import numpy as np

from src.data.loader import (
    aggregate_nasa_yearly,
    load_crop_data,
    load_nasa_data,
    merge_datasets,
)


def test_load_crop_data_shape():
    df = load_crop_data()
    assert len(df) == 50_765
    assert "Yield_kg_per_ha" in df.columns
    assert df.index.name == "Dist Code"


def test_crop_climate_columns_are_static_vectors():
    # Audit finding: climate columns are per-crop constants, not measurements.
    df = load_crop_data()
    for col in ["Temperature_C", "Humidity_%", "pH", "Rainfall_mm"]:
        assert df[col].nunique() <= 6, f"{col} should be constant per crop"


def test_nutrient_requirement_ratios_are_constant_per_crop():
    # Audit finding: the nutrient-requirement columns are derived from the
    # target (Yield). Within each crop the ratio nutrient/Yield is constant
    # (e.g. rice N_req/Yield = 0.025, chickpea = 0.018, ...). This is why
    # these columns must be excluded from ML features.
    df = load_crop_data()
    for crop, group in df.groupby("Crop"):
        for req_col in ["N_req_kg_per_ha", "P_req_kg_per_ha", "K_req_kg_per_ha"]:
            ratio = group[req_col] / group["Yield_kg_per_ha"]
            assert np.allclose(ratio, ratio.iloc[0]), (
                f"{req_col} / Yield ratio is not constant for crop={crop!r}"
            )


def test_total_nutrient_columns_are_requirement_times_area():
    # Audit finding: Total_N/P/K_kg are computed as requirement x Area_ha.
    # Combined with the ratio test above, the six nutrient columns are fully
    # target-derived and must never enter the model.
    df = load_crop_data()
    for req_col, total_col in [
        ("N_req_kg_per_ha", "Total_N_kg"),
        ("P_req_kg_per_ha", "Total_P_kg"),
        ("K_req_kg_per_ha", "Total_K_kg"),
    ]:
        expected = (df[req_col] * df["Area_ha"]).to_numpy()
        assert np.allclose(df[total_col].to_numpy(), expected), (
            f"{total_col} != {req_col} * Area_ha"
        )


def test_load_nasa_data_shape():
    df = load_nasa_data()
    assert len(df) == 300
    assert {"Year", "Month", "Rainfall", "AvgTemp"}.issubset(df.columns)


def test_aggregate_nasa_yearly():
    nasa = load_nasa_data()
    yearly = aggregate_nasa_yearly(nasa)
    assert yearly["Year"].nunique() == 25
    first_year = yearly[yearly["Year"] == yearly["Year"].min()].iloc[0]
    month_sum = nasa[nasa["Year"] == yearly["Year"].min()]["Rainfall"].sum()
    assert abs(first_year["Rainfall"] - month_sum) < 1e-6


def test_merge_datasets_drops_nasa_rainfall():
    crop = load_crop_data()
    nasa = aggregate_nasa_yearly(load_nasa_data())
    merged = merge_datasets(crop, nasa)
    assert "Rainfall" not in merged.columns
    assert "Rainfall_mm" in merged.columns
    assert len(merged) > 0