"""Tests for E07 data validation and provenance module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.validate import (
    CROP_EXPECTED_COLUMNS,
    CROP_EXPECTED_CROPS,
    CROP_EXPECTED_ROW_COUNT,
    CROP_YEAR_RANGE,
    LEAK_COLUMNS,
    NASA_EXPECTED_COLUMNS,
    NASA_EXPECTED_MONTHS,
    NASA_EXPECTED_ROW_COUNT,
    NASA_YEAR_RANGE,
    NUTRIENT_COLUMNS,
    ValidationError,
    crop_provenance,
    nasa_provenance,
    validate_all,
    validate_crop_data,
    validate_nasa_data,
    validate_no_leakage_in_features,
)
from src.data.loader import load_crop_data, load_nasa_data
from src.features.engineering import LEAK_COLUMNS as ENG_LEAK_COLUMNS


# ------------------------------------------------------------------ helpers

def _make_crop_df(**overrides) -> pd.DataFrame:
    """Build a minimal valid crop DataFrame matching expected row count."""
    n = CROP_EXPECTED_ROW_COUNT
    n_cycle = 1000
    _crops = ["chickpea", "cotton", "maize", "rice"]
    crops = np.tile(_crops, n // 4 + 1)[:n]
    data = {
        "Year": np.full(n, 2000),
        "State Code": np.tile(np.arange(1, 21), n // 20 + 1)[:n],
        "State Name": np.tile([f"S{i}" for i in range(1, 21)], n // 20 + 1)[:n],
        "Dist Name": np.tile([f"Dist{i}" for i in range(n_cycle)], n // n_cycle + 1)[:n],
        "Crop": crops,
        "Area_ha": np.full(n, 100.0),
        "Yield_kg_per_ha": np.full(n, 2000.0),
        "N_req_kg_per_ha": np.full(n, 50.0),      # 0.025 × 2000 for rice
        "P_req_kg_per_ha": np.full(n, 24.0),      # 0.012 × 2000
        "K_req_kg_per_ha": np.full(n, 44.0),      # 0.022 × 2000 for rice
        "Total_N_kg": np.full(n, 5000.0),
        "Total_P_kg": np.full(n, 2400.0),
        "Total_K_kg": np.full(n, 4400.0),
        "Temperature_C": np.full(n, 25),
        "Humidity_%": np.full(n, 80),
        "pH": np.full(n, 6.5),
        "Rainfall_mm": np.full(n, 1200),
        "Wind_Speed_m_s": np.full(n, 2.0),
        "Solar_Radiation_MJ_m2_day": np.full(n, 18),
    }
    data.update(overrides)
    df = pd.DataFrame(data)
    df.index.name = "Dist Code"
    return df


def _make_nasa_df(**overrides) -> pd.DataFrame:
    """Build a minimal valid NASA POWER DataFrame (300 rows)."""
    rows = []
    for y in range(1996, 2021):
        for m in NASA_EXPECTED_MONTHS:
            rows.append({"Year": y, "Month": m, "Rainfall": 50.0, "AvgTemp": 25.0, "MaxTemp": 35.0, "MinTemp": 15.0})
    df = pd.DataFrame(rows)
    for col, val in overrides.items():
        df[col] = val
    return df


# ------------------------------------------------------------------ real data

class TestValidateRealCropData:
    def test_valid(self):
        df = load_crop_data()
        warnings = validate_crop_data(df)
        assert isinstance(warnings, list)

    def test_provenance(self):
        p = crop_provenance()
        assert p["rows"] == 50_765
        assert len(p["columns"]) == 19
        assert "known_limitations" in p


class TestValidateRealNasaData:
    def test_valid(self):
        df = load_nasa_data()
        warnings = validate_nasa_data(df)
        assert any("single spatial point" in w for w in warnings)

    def test_provenance(self):
        p = nasa_provenance()
        assert p["rows"] == 300
        assert p["temporal_coverage"] == "Monthly, 1996–2020"


class TestValidateAll:
    def test_valid(self):
        crop = load_crop_data()
        nasa = load_nasa_data()
        result = validate_all(crop, nasa, features=["Area_ha", "pH"])
        assert "crop_provenance" in result
        assert "nasa_provenance" in result
        assert "warnings" in result

    def test_leakage_guard_raises(self):
        crop = load_crop_data()
        nasa = load_nasa_data()
        with pytest.raises(ValidationError, match="leakage_detector"):
            validate_all(crop, nasa, features=["Area_ha", "N_req_kg_per_ha"])


# ------------------------------------------------------------------ columns

class TestCropColumns:
    def test_missing_column(self):
        df = _make_crop_df()
        df = df.drop(columns=["Year"])
        with pytest.raises(ValidationError, match="missing columns.*Year"):
            validate_crop_data(df)

    def test_extra_column_passes(self):
        """Extra columns are allowed (schema is minimum required)."""
        df = _make_crop_df()
        df["Extra"] = 1
        validate_crop_data(df)

    def test_wrong_index_name(self):
        df = _make_crop_df()
        df.index.name = "wrong"
        with pytest.raises(ValidationError, match="index name"):
            validate_crop_data(df)


# ------------------------------------------------------------------ dtypes

class TestCropDtypes:
    def test_string_in_numeric_column(self):
        df = _make_crop_df(Year=[str(2000)] * CROP_EXPECTED_ROW_COUNT)
        with pytest.raises(ValidationError, match="dtype"):
            validate_crop_data(df)


# ------------------------------------------------------------------ nulls

class TestCropNulls:
    def test_null_in_required_column(self):
        df = _make_crop_df()
        df.loc[0, "Yield_kg_per_ha"] = np.nan
        with pytest.raises(ValidationError, match="nulls"):
            validate_crop_data(df)

    def test_null_in_optional_column_passes(self):
        """Columns not in the required list can have nulls."""
        df = _make_crop_df()
        df["Extra"] = np.nan
        validate_crop_data(df)


# ------------------------------------------------------------------ year range

class TestCropYearRange:
    def test_year_too_low(self):
        df = _make_crop_df(Year=[1965] * CROP_EXPECTED_ROW_COUNT)
        with pytest.raises(ValidationError, match="range"):
            validate_crop_data(df)

    def test_year_too_high(self):
        df = _make_crop_df(Year=[2018] * CROP_EXPECTED_ROW_COUNT)
        with pytest.raises(ValidationError, match="range"):
            validate_crop_data(df)


# ------------------------------------------------------------------ row count

class TestCropRowCount:
    def test_wrong_count(self):
        df = _make_crop_df().iloc[:100]
        with pytest.raises(ValidationError, match="expected 50765 rows"):
            validate_crop_data(df)


# ------------------------------------------------------------------ crops

class TestCropValues:
    def test_unknown_crop(self):
        df = _make_crop_df(Crop=["wheat"] * CROP_EXPECTED_ROW_COUNT)
        with pytest.raises(ValidationError, match="crops"):
            validate_crop_data(df)


# ------------------------------------------------------------------ duplicates

class TestNasaCompleteness:
    def test_incomplete_year(self):
        rows = []
        for y in range(1996, 2021):
            for m in NASA_EXPECTED_MONTHS[:11]:  # missing one month
                rows.append({"Year": y, "Month": m, "Rainfall": 50.0, "AvgTemp": 25.0, "MaxTemp": 35.0, "MinTemp": 15.0})
        df = pd.DataFrame(rows)
        with pytest.raises(ValidationError, match="incomplete years"):
            validate_nasa_data(df)

    def test_extra_month(self):
        df = _make_nasa_df()
        extra = pd.DataFrame([{"Year": 2020, "Month": "JAN", "Rainfall": 0, "AvgTemp": 0, "MaxTemp": 0, "MinTemp": 0}])
        df = pd.concat([df, extra], ignore_index=True)
        with pytest.raises(ValidationError, match="incomplete years"):
            validate_nasa_data(df)


# ------------------------------------------------------------------ NASA year range

class TestNasaYearRange:
    def test_year_out_of_range(self):
        rows = []
        for y in range(1995, 2021):
            for m in NASA_EXPECTED_MONTHS:
                rows.append({"Year": y, "Month": m, "Rainfall": 50.0, "AvgTemp": 25.0, "MaxTemp": 35.0, "MinTemp": 15.0})
        df = pd.DataFrame(rows)
        with pytest.raises(ValidationError, match="range"):
            validate_nasa_data(df)


# ------------------------------------------------------------------ NASA months

class TestNasaMonths:
    def test_wrong_months(self):
        rows = []
        for y in range(1996, 2021):
            for m in range(1, 13):
                rows.append({"Year": y, "Month": str(m), "Rainfall": 50.0, "AvgTemp": 25.0, "MaxTemp": 35.0, "MinTemp": 15.0})
        df = pd.DataFrame(rows)
        with pytest.raises(ValidationError, match="months"):
            validate_nasa_data(df)


# ------------------------------------------------------------------ leakage guard

class TestLeakageGuard:
    def test_clean_features(self):
        validate_no_leakage_in_features(["Area_ha", "pH", "Temperature_C"])

    def test_leaky_feature_detected(self):
        with pytest.raises(ValidationError, match="leakage_detector"):
            validate_no_leakage_in_features(["Area_ha", "N_req_kg_per_ha"])

    def test_multiple_leaky(self):
        with pytest.raises(ValidationError, match="N_req_kg_per_ha.*Total_N_kg"):
            validate_no_leakage_in_features(["N_req_kg_per_ha", "Total_N_kg", "Area_ha"])

    def test_empty_features(self):
        validate_no_leakage_in_features([])


# ------------------------------------------------------------------ corruption fixtures

class TestCorruptedFixtures:
    def test_empty_crop_df(self):
        df = pd.DataFrame()
        with pytest.raises(ValidationError):
            validate_crop_data(df)

    def test_empty_nasa_df(self):
        df = pd.DataFrame()
        with pytest.raises(ValidationError):
            validate_nasa_data(df)

    def test_crop_wrong_columns(self):
        df = pd.DataFrame({"A": [1], "B": [2]})
        with pytest.raises(ValidationError, match="missing columns"):
            validate_crop_data(df)

    def test_nasa_null_year(self):
        rows = []
        for y in range(1996, 2021):
            for m in NASA_EXPECTED_MONTHS:
                rows.append({"Year": y, "Month": m, "Rainfall": 50.0, "AvgTemp": 25.0, "MaxTemp": 35.0, "MinTemp": 15.0})
        df = pd.DataFrame(rows)
        df.loc[0, "Year"] = np.nan
        with pytest.raises(ValidationError, match="nulls"):
            validate_nasa_data(df)


# ------------------------------------------------------------------ provenance content

class TestProvenanceContent:
    def test_crop_has_all_fields(self):
        p = crop_provenance()
        for key in ["name", "source_file", "format", "rows", "year_range",
                     "crops", "columns", "schema_validated", "known_limitations",
                     "audit_invariants"]:
            assert key in p, f"Missing provenance key: {key}"

    def test_nasa_has_all_fields(self):
        p = nasa_provenance()
        for key in ["name", "source_file", "format", "rows", "year_range",
                     "months", "columns", "schema_validated", "known_limitations",
                     "spatial_coverage", "temporal_coverage"]:
            assert key in p, f"Missing provenance key: {key}"

    def test_nasa_flags_single_point(self):
        p = nasa_provenance()
        assert "single" in p["spatial_coverage"].lower()
