"""Tests for feature engineering and the split logic."""

import pandas as pd

from src.features.engineering import (
    LEAK_COLUMNS,
    default_features,
    temporal_split,
)


def _sample_frame():
    rows = []
    for year in (2010, 2012, 2015):
        for crop in ("rice", "maize"):
            rows.append(
                {
                    "Year": year,
                    "Area_ha": 100,
                    "Temperature_C": 25,
                    "Humidity_%": 80,
                    "pH": 6.5,
                    "Rainfall_mm": 1200,
                    "Wind_Speed_m_s": 2.0,
                    "Solar_Radiation_MJ_m2_day": 18,
                    "AvgTemp": 20 + year - 2010,
                    "MaxTemp": 30,
                    "MinTemp": 10,
                    "N_req_kg_per_ha": 400,
                    "Yield_kg_per_ha": 4000,
                }
            )
    return pd.DataFrame(rows)


def test_default_features_never_include_leaky_columns():
    features = default_features()
    for leak in LEAK_COLUMNS:
        assert leak not in features


def test_nutrient_columns_are_not_in_improved_features():
    features = default_features()
    assert set(features) == {"Area_ha", "pH", "Temperature_C", "Humidity_%",
                             "Rainfall_mm", "Wind_Speed_m_s", "Solar_Radiation_MJ_m2_day",
                             "AvgTemp", "MaxTemp", "MinTemp"}


def test_temporal_split_respects_cutoff():
    df = _sample_frame()
    features = default_features()
    X_train, X_test, y_train, y_test = temporal_split(
        df, features, "Yield_kg_per_ha", cutoff_year=2015
    )
    assert len(X_train) == 4
    assert len(X_test) == 2
    assert (df.loc[X_test.index, "Year"] >= 2015).all()
    assert (df.loc[X_train.index, "Year"] < 2015).all()


def test_temporal_split_matches_y_labels():
    df = _sample_frame()
    features = default_features()
    _, _, y_train, y_test = temporal_split(df, features, "Yield_kg_per_ha", 2015)
    assert (y_train.index == df[df["Year"] < 2015].index).all()
    assert (y_test.index == df[df["Year"] >= 2015].index).all()