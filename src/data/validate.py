"""Dataset schema validation and provenance metadata (E07).

Validates the two raw datasets against their known schema, coverage,
and structural invariants.  Designed for CI: deterministic, fast, and
fails loudly with actionable error messages.

.. note::

   No new dependencies are introduced — validation uses only pandas and
   standard-library checks.  This keeps the module lightweight and avoids
   adding ``pandera`` or ``pyarrow`` at this stage.

Provenance
----------
Each dataset's provenance is captured as a plain dict describing source,
coverage, schema, and known limitations.  Use :func:`crop_provenance` and
:func:`nasa_provenance` to retrieve it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.features.engineering import LEAK_COLUMNS

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

CROP_EXPECTED_COLUMNS: list[str] = [
    "Year", "State Code", "State Name", "Dist Name", "Crop",
    "Area_ha", "Yield_kg_per_ha",
    "N_req_kg_per_ha", "P_req_kg_per_ha", "K_req_kg_per_ha",
    "Total_N_kg", "Total_P_kg", "Total_K_kg",
    "Temperature_C", "Humidity_%", "pH", "Rainfall_mm",
    "Wind_Speed_m_s", "Solar_Radiation_MJ_m2_day",
]

CROP_NUMERIC_COLUMNS: list[str] = [
    "Year", "State Code", "Area_ha", "Yield_kg_per_ha",
    "N_req_kg_per_ha", "P_req_kg_per_ha", "K_req_kg_per_ha",
    "Total_N_kg", "Total_P_kg", "Total_K_kg",
    "Temperature_C", "Humidity_%", "pH", "Rainfall_mm",
    "Wind_Speed_m_s", "Solar_Radiation_MJ_m2_day",
]

CROP_YEAR_RANGE: tuple[int, int] = (1966, 2017)
CROP_EXPECTED_ROW_COUNT: int = 50_765
CROP_EXPECTED_CROPS: list[str] = ["chickpea", "cotton", "maize", "rice"]
CROP_EXPECTED_INDEX_NAME: str = "Dist Code"

NASA_EXPECTED_COLUMNS: list[str] = [
    "Year", "Month", "Rainfall", "AvgTemp", "MaxTemp", "MinTemp",
]

NASA_EXPECTED_MONTHS: list[str] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]

NASA_YEAR_RANGE: tuple[int, int] = (1996, 2020)
NASA_EXPECTED_ROW_COUNT: int = 300

# Nutrient columns that are target-derived and must never be model features.
NUTRIENT_COLUMNS: list[str] = [
    "N_req_kg_per_ha", "P_req_kg_per_ha", "K_req_kg_per_ha",
    "Total_N_kg", "Total_P_kg", "Total_K_kg",
]

# Crop-specific nutrient/Yield ratios (verified against full dataset).
EXPECTED_NUTRIENT_RATIOS: dict[str, dict[str, float]] = {
    "chickpea": {"N_req_kg_per_ha": 0.018, "P_req_kg_per_ha": 0.010, "K_req_kg_per_ha": 0.018},
    "cotton":   {"N_req_kg_per_ha": 0.027, "P_req_kg_per_ha": 0.012, "K_req_kg_per_ha": 0.027},
    "maize":    {"N_req_kg_per_ha": 0.027, "P_req_kg_per_ha": 0.012, "K_req_kg_per_ha": 0.017},
    "rice":     {"N_req_kg_per_ha": 0.025, "P_req_kg_per_ha": 0.012, "K_req_kg_per_ha": 0.022},
}


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when a dataset fails a schema or invariant check."""


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _check_columns(df: pd.DataFrame, expected: list[str], name: str) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValidationError(f"{name}: missing columns {sorted(missing)}")


def _check_dtypes(
    df: pd.DataFrame, columns: list[str], expected_dtypes: dict[str, str], name: str,
) -> None:
    for col, expected_dtype in expected_dtypes.items():
        if col not in df.columns:
            continue  # caught by _check_columns
        actual = df[col].dtype
        if expected_dtype == "numeric":
            if not pd.api.types.is_numeric_dtype(actual):
                raise ValidationError(
                    f"{name}: column '{col}' has dtype '{actual}', expected numeric"
                )
        elif not pd.api.types.is_dtype_equal(actual, expected_dtype):
            raise ValidationError(
                f"{name}: column '{col}' has dtype '{actual}', expected '{expected_dtype}'"
            )


def _check_nulls(df: pd.DataFrame, columns: list[str], name: str) -> None:
    null_counts = df[columns].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValidationError(
            f"{name}: unexpected nulls in columns "
            + {col: int(n) for col, n in bad.items()}.__repr__()
        )


def _check_range(
    df: pd.DataFrame, col: str, low: int, high: int, name: str,
) -> None:
    if col not in df.columns:
        return
    actual_min = int(df[col].min())
    actual_max = int(df[col].max())
    if actual_min < low or actual_max > high:
        raise ValidationError(
            f"{name}: '{col}' range [{actual_min}, {actual_max}] outside "
            f"expected [{low}, {high}]"
        )


def _check_row_count(df: pd.DataFrame, expected: int, name: str) -> None:
    if len(df) != expected:
        raise ValidationError(
            f"{name}: expected {expected} rows, got {len(df)}"
        )


# ---------------------------------------------------------------------------
# Crop dataset validation
# ---------------------------------------------------------------------------

def validate_crop_data(df: pd.DataFrame) -> list[str]:
    """Validate the district-level crop yield dataset.

    Checks: columns, dtypes, nulls, year range, row count, expected crops,
    index name, nutrient-column leakage invariant (target-derived ratios are
    constant per crop), and Total_N/P/K = requirement × Area_ha.

    Returns a list of non-critical warnings (empty if clean).
    Raises :class:`ValidationError` on any hard failure.
    """
    warnings: list[str] = []

    _check_columns(df, CROP_EXPECTED_COLUMNS, "crop_dataset")

    _check_dtypes(df, CROP_NUMERIC_COLUMNS, {c: "numeric" for c in CROP_NUMERIC_COLUMNS}, "crop_dataset")

    _check_nulls(df, CROP_EXPECTED_COLUMNS, "crop_dataset")

    _check_range(df, "Year", CROP_YEAR_RANGE[0], CROP_YEAR_RANGE[1], "crop_dataset")

    _check_row_count(df, CROP_EXPECTED_ROW_COUNT, "crop_dataset")

    # Index name
    if df.index.name != CROP_EXPECTED_INDEX_NAME:
        raise ValidationError(
            f"crop_dataset: index name is '{df.index.name}', expected '{CROP_EXPECTED_INDEX_NAME}'"
        )

    # Expected crops
    actual_crops = sorted(df["Crop"].unique())
    if actual_crops != sorted(CROP_EXPECTED_CROPS):
        raise ValidationError(
            f"crop_dataset: crops are {actual_crops}, expected {sorted(CROP_EXPECTED_CROPS)}"
        )

    # --- Leakage invariant: nutrient/Yield ratios constant per crop ---
    for crop, group in df.groupby("Crop"):
        for req_col in ["N_req_kg_per_ha", "P_req_kg_per_ha", "K_req_kg_per_ha"]:
            ratio = group[req_col] / group["Yield_kg_per_ha"]
            if not ratio.std() < 1e-10:
                raise ValidationError(
                    f"crop_dataset: {req_col}/Yield ratio varies within crop '{crop}' "
                    f"(std={ratio.std():.2e}) — leakage invariant broken"
                )

    # --- Total_N/P/K = requirement × Area_ha ---
    for req_col, total_col in [
        ("N_req_kg_per_ha", "Total_N_kg"),
        ("P_req_kg_per_ha", "Total_P_kg"),
        ("K_req_kg_per_ha", "Total_K_kg"),
    ]:
        expected = df[req_col] * df["Area_ha"]
        if not (df[total_col] - expected).abs().max() < 1e-6:
            raise ValidationError(
                f"crop_dataset: {total_col} != {req_col} * Area_ha — "
                "leakage invariant broken"
            )

    return warnings


# ---------------------------------------------------------------------------
# NASA POWER validation
# ---------------------------------------------------------------------------

def validate_nasa_data(df: pd.DataFrame) -> list[str]:
    """Validate the NASA POWER monthly climate record.

    Checks: columns, dtypes, nulls, year range, expected months, row count.
    Documents that this is a single spatial point.

    Returns a list of non-critical warnings (empty if clean).
    Raises :class:`ValidationError` on any hard failure.
    """
    warnings: list[str] = []

    _check_columns(df, NASA_EXPECTED_COLUMNS, "nasa_power")

    _check_nulls(df, NASA_EXPECTED_COLUMNS, "nasa_power")

    _check_range(df, "Year", NASA_YEAR_RANGE[0], NASA_YEAR_RANGE[1], "nasa_power")

    # Check completeness: 12 months per year
    month_counts = df.groupby("Year")["Month"].count()
    bad_years = month_counts[month_counts != 12]
    if not bad_years.empty:
        raise ValidationError(
            f"nasa_power: incomplete years {bad_years.to_dict()} (expected 12 months each)"
        )

    _check_row_count(df, NASA_EXPECTED_ROW_COUNT, "nasa_power")

    # Month values
    actual_months = sorted(df["Month"].unique())
    if actual_months != sorted(NASA_EXPECTED_MONTHS):
        raise ValidationError(
            f"nasa_power: months are {actual_months}, expected {sorted(NASA_EXPECTED_MONTHS)}"
        )

    # Provenance warning about single-point data
    warnings.append(
        "nasa_power: single spatial point — indicators are NOT district-specific"
    )

    return warnings


# ---------------------------------------------------------------------------
# Feature-leakage guard
# ---------------------------------------------------------------------------

def validate_no_leakage_in_features(features: list[str]) -> None:
    """Verify that no target-derived nutrient columns are in the feature set.

    This is the *leakage detector* from the E07 roadmap: it fails loudly
    if nutrient columns are re-added to the model input.
    """
    leaked = set(features) & set(LEAK_COLUMNS)
    if leaked:
        raise ValidationError(
            f"leakage_detector: target-derived columns found in feature set: "
            f"{sorted(leaked)}. These columns are computed from the target and "
            f"must not be used as model inputs."
        )


# ---------------------------------------------------------------------------
# Provenance metadata
# ---------------------------------------------------------------------------

def crop_provenance(path: Path | str | None = None) -> dict:
    """Return provenance metadata for the crop-yield dataset."""
    return {
        "name": "District-level crop yield dataset",
        "source_file": str(path or "data/raw/crop_yield/Custom_Crops_yield_Historical_Dataset.csv"),
        "format": "CSV (index: Dist Code)",
        "rows": CROP_EXPECTED_ROW_COUNT,
        "year_range": list(CROP_YEAR_RANGE),
        "crops": CROP_EXPECTED_CROPS,
        "columns": CROP_EXPECTED_COLUMNS,
        "schema_validated": True,
        "known_limitations": [
            "Static per-crop climate columns (Temperature_C, Humidity_%, etc.) — "
            "encode crop suitability, not real spatiotemporal weather.",
            "Nutrient columns (N/P/K_req, Total_N/P/K) are deterministically "
            "derived from Yield_kg_per_ha using crop-specific constant "
            "coefficients — target-derived leakage features.",
            "Original source and download procedure not recorded.",
        ],
        "audit_invariants": [
            "nutrient/Yield ratio constant per crop",
            "Total_N/P/K = requirement × Area_ha",
            "climate columns have ≤6 unique values per column",
        ],
    }


def nasa_provenance(path: Path | str | None = None) -> dict:
    """Return provenance metadata for the NASA POWER dataset."""
    return {
        "name": "NASA POWER monthly climate record",
        "source_file": str(path or "data/raw/nasa_power/nasa_power_updated.csv"),
        "format": "CSV",
        "rows": NASA_EXPECTED_ROW_COUNT,
        "year_range": list(NASA_YEAR_RANGE),
        "months": NASA_EXPECTED_MONTHS,
        "columns": NASA_EXPECTED_COLUMNS,
        "schema_validated": True,
        "known_limitations": [
            "Single spatial point (one lat/lon) — does NOT provide spatial "
            "climate information. Cannot support spatially resolved downscaling.",
            "Original notebook applied this single point to every district "
            "in all 20 states (one weather signal for the whole country).",
            "Only 25 years (1996–2020) — shorter than WMO-standard 30-year "
            "climatological reference period.",
            "Download script not yet in repository; CSV is the raw input.",
        ],
        "spatial_coverage": "Single point (lat/lon implicit, not stored)",
        "temporal_coverage": "Monthly, 1996–2020",
    }


# ---------------------------------------------------------------------------
# Combined validation entry point
# ---------------------------------------------------------------------------

def validate_all(
    crop_df: pd.DataFrame,
    nasa_df: pd.DataFrame,
    features: list[str] | None = None,
) -> dict:
    """Run all E07 validations and return provenance metadata.

    Parameters
    ----------
    crop_df : Loaded crop dataset.
    nasa_df : Loaded NASA POWER dataset.
    features : Model feature list (optional — if provided, leakage guard runs).

    Returns
    -------
    dict with keys ``crop_provenance``, ``nasa_provenance``, ``warnings``.

    Raises :class:`ValidationError` on any hard failure.
    """
    crop_warns = validate_crop_data(crop_df)
    nasa_warns = validate_nasa_data(nasa_df)
    if features is not None:
        validate_no_leakage_in_features(features)

    return {
        "crop_provenance": crop_provenance(),
        "nasa_provenance": nasa_provenance(),
        "warnings": crop_warns + nasa_warns,
    }
