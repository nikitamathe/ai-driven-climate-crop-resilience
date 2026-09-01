"""Data loading and aggregation helpers.

The original notebook merged the district-level crop dataset with a NASA POWER
record aggregated from monthly to yearly. These helpers reproduce that logic
while giving every function an explicit path and schema contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_crop_data(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load the district-level crop yield dataset.

    The CSV uses the first column (``Dist Code``) as the row index.
    """
    path = path or RAW_DIR / "crop_yield" / "Custom_Crops_yield_Historical_Dataset.csv"
    return pd.read_csv(path, index_col=0)


def load_nasa_data(path: Path | None = None) -> pd.DataFrame:
    """Load the NASA POWER monthly climate record."""
    path = path or RAW_DIR / "nasa_power" / "nasa_power_updated.csv"
    return pd.read_csv(path)


def aggregate_nasa_yearly(nasa_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the monthly NASA POWER record to yearly climate averages.

    Rainfall is summed; temperatures are averaged.
    """
    return (
        nasa_df.groupby("Year")
        .agg(
            Rainfall=("Rainfall", "sum"),
            AvgTemp=("AvgTemp", "mean"),
            MaxTemp=("MaxTemp", "mean"),
            MinTemp=("MinTemp", "mean"),
        )
        .reset_index()
    )


def merge_datasets(crop_df: pd.DataFrame, nasa_yearly: pd.DataFrame) -> pd.DataFrame:
    """Merge crop records with aggregated NASA climate data on ``Year``.

    The NASA ``Rainfall`` column is dropped after the merge because the crop
    dataset already carries a (static, per-crop) ``Rainfall_mm`` column.
    """
    merged = crop_df.merge(nasa_yearly, on="Year", how="inner")
    return merged.drop(columns=["Rainfall"])