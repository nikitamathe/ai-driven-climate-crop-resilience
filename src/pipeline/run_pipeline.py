"""End-to-end crop resilience mapping pipeline.

Run from the project root:

    python -m src.pipeline.run_pipeline

The pipeline (re)generates ``data/processed/final_crop_resilience_district_year.csv``
using a corrected methodology:

* nutrient columns that are derived from the target are excluded,
* the train/test split is temporal (by year) instead of a random split.

The original (audit-flagged) output is preserved alongside it as
``final_crop_resilience_district_year_original.csv``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.climate.indicators import write_climate_indicators
from src.data.loader import (
    PROCESSED_DIR,
    aggregate_nasa_yearly,
    load_crop_data,
    load_nasa_data,
    merge_datasets,
)
from src.features.engineering import build_xy, default_features, temporal_split
from src.models.evaluate import resilience_class, resilience_index, report_metrics
from src.models.train import train_random_forest
from src.visualization.plots import (
    plot_anomaly_heatmap,
    plot_climate_trend_table,
    plot_feature_importance,
    plot_resilience_distribution,
    plot_spi_timeseries,
    plot_yield_vs_rainfall,
)

def run(
    cutoff_year: int = 2014,
    output_csv: Path | None = None,
    out_dir: Path | None = None,
) -> dict:
    """Execute the full pipeline and return key outputs."""
    crop_df = load_crop_data()
    nasa_df = load_nasa_data()
    nasa_yearly = aggregate_nasa_yearly(nasa_df)
    merged_df = merge_datasets(crop_df, nasa_yearly)

    features = default_features()
    target = "Yield_kg_per_ha"

    X_train, X_test, y_train, y_test = temporal_split(
        merged_df, features, target, cutoff_year
    )
    print(f"Train rows: {len(X_train)} (years < {cutoff_year})")
    print(f"Test  rows: {len(X_test)} (years >= {cutoff_year})")

    model = train_random_forest(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = report_metrics(y_test, y_pred)

    merged_df["Predicted_Yield"] = model.predict(build_xy(merged_df, features)[0])
    merged_df["Resilience_Index"] = resilience_index(
        merged_df["Yield_kg_per_ha"], merged_df["Predicted_Yield"]
    )
    merged_df["Resilience_Class"] = merged_df["Resilience_Index"].apply(
        resilience_class
    )

    summary = (
        merged_df.groupby(["Year", "State Name", "Dist Name", "Crop"])
        .agg(
            Min_Temp=("Temperature_C", lambda x: x.quantile(0.05)),
            Max_Temp=("Temperature_C", lambda x: x.quantile(0.95)),
            Avg_Temp=("Temperature_C", "mean"),
            Rainfall=("Rainfall_mm", "mean"),
            Actual_Yield=("Yield_kg_per_ha", "mean"),
            Predicted_Yield=("Predicted_Yield", "mean"),
            Resilience_Index=("Resilience_Index", "mean"),
        )
        .reset_index()
    )
    summary["Resilience_Class"] = summary["Resilience_Index"].apply(resilience_class)
    summary = summary.round(
        {
            "Min_Temp": 1,
            "Max_Temp": 1,
            "Avg_Temp": 1,
            "Rainfall": 0,
            "Actual_Yield": 0,
            "Predicted_Yield": 0,
            "Resilience_Index": 2,
        }
    )

    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_csv or (out_dir / "final_crop_resilience_district_year.csv")
    summary.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({len(summary)} rows)")

    imgs = out_dir  # re-use processed dir for plots to keep docs/images clean
    plot_yield_vs_rainfall(summary, imgs / "yield_vs_rainfall.png")
    plot_feature_importance(model, features, imgs / "feature_importance.png")
    plot_resilience_distribution(summary, imgs / "resilience_distribution.png")

    # --- E06 climate indicators (separate from model output) ---
    indicators_path = out_dir / "climate_indicators.csv"
    indicators_df = write_climate_indicators(nasa_df, indicators_path)
    print(f"Wrote {indicators_path} ({len(indicators_df)} rows, E06 indicators)")
    plot_spi_timeseries(indicators_df, imgs / "spi_timeseries.png")
    plot_anomaly_heatmap(indicators_df, imgs / "anomaly_heatmap.png")
    plot_climate_trend_table(indicators_df, imgs / "climate_trend_table.png")

    print(f"Wrote plots to {imgs}")

    return {
        "model": model,
        "metrics": metrics,
        "summary": summary,
        "climate_indicators": indicators_df,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cutoff-year", type=int, default=2014,
        help="First year of the test period (train on earlier years)",
    )
    args = parser.parse_args(argv)
    run(cutoff_year=args.cutoff_year)
    return 0


if __name__ == "__main__":
    sys.exit(main())