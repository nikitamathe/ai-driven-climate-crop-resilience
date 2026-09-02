"""Feature engineering for crop-yield prediction.

Phase 0 audit finding
---------------------
The original notebook trained a Random Forest on features that included
``N_req_kg_per_ha``, ``P_req_kg_per_ha``, ``K_req_kg_per_ha`` and their
``Total_*_kg`` products. Those columns are *deterministically derived from
the target* using crop-specific constant coefficients (e.g. rice
N/Yield = 0.025, maize N/Yield = 0.027, cotton N/Yield = 0.027, chickpea
N/Yield = 0.018):

    N_req_kg_per_ha = Yield_kg_per_ha * <crop-specific coefficient>
    Total_N_kg      = N_req_kg_per_ha * Area_ha

These are target-derived leakage features and must not be used as model
inputs. This leaks the target variable into the model and inflates the
reported R^2 (0.97). The improved pipeline therefore excludes those columns.
"""

from __future__ import annotations

import pandas as pd

TARGET = "Yield_kg_per_ha"

# Derived from the target -> must never enter the model.
LEAK_COLUMNS = [
    "N_req_kg_per_ha",
    "P_req_kg_per_ha",
    "K_req_kg_per_ha",
    "Total_N_kg",
    "Total_P_kg",
    "Total_K_kg",
]

# Static per-crop environment constants in the raw dataset.
CROP_CLIMATE_COLUMNS = [
    "Temperature_C",
    "Humidity_%",
    "Rainfall_mm",
    "Wind_Speed_m_s",
    "Solar_Radiation_MJ_m2_day",
]

# Yearly climate signal merged in from the NASA POWER record.
NASA_CLIMATE_COLUMNS = ["AvgTemp", "MaxTemp", "MinTemp"]

OTHER_COLUMNS = ["Area_ha", "pH"]


def default_features(include_leaky: bool = False) -> list[str]:
    """Return the feature set used by the model.

    By default the leaky nutrient columns are excluded. Passing
    ``include_leaky=True`` reproduces the original (audit-flagged) feature list.
    """
    features = OTHER_COLUMNS + CROP_CLIMATE_COLUMNS + NASA_CLIMATE_COLUMNS
    if include_leaky:
        features = LEAK_COLUMNS + features
    return features


def build_xy(
    merged_df: pd.DataFrame,
    features: list[str] | None = None,
    target: str = TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a merged frame into feature matrix ``X`` and target ``y``."""
    features = features or default_features()
    missing = [c for c in features if c not in merged_df.columns]
    if missing:
        raise ValueError(f"Columns not present in data: {missing}")
    return merged_df[features], merged_df[target]


def temporal_split(
    merged_df: pd.DataFrame,
    features: list[str],
    target: str,
    cutoff_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train/test split by year.

    Data from years ``>= cutoff_year`` forms the test set, everything before
    it forms the train set. This avoids the temporal leakage present in the
    original random split (where the same year appeared on both sides).
    """
    train = merged_df[merged_df["Year"] < cutoff_year]
    test = merged_df[merged_df["Year"] >= cutoff_year]
    X_train, y_train = build_xy(train, features, target)
    X_test, y_test = build_xy(test, features, target)
    return X_train, X_test, y_train, y_test