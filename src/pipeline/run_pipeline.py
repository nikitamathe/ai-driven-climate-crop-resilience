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
from src.data.validate import validate_all
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


def _fit_base_for_conformal(n_estimators: int = 300, max_depth: int = 15):
    """Create an unfitted RandomForestRegressor matching the pipeline baseline."""
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        random_state=42, n_jobs=-1,
    )


def _add_conformal_intervals(
    merged_df,
    model,
    X_train,
    y_train,
    year_groups_train,
    features,
    *,
    target: str = "Yield_kg_per_ha",
    test_year_min: int = 2014,
    n_cal_years: int = 3,
    alpha: float = 0.10,
):
    """Additive E05 overlay: prediction-interval columns around point predictions.

    This preserves the exact baseline point model (``model``) and therefore the
    reported metrics. The conformal half-width ``q_hat`` is estimated from a
    time-aware proper/calibration split of the *training* data using an
    independent RF scaffold; it is then applied as a symmetric band around the
    existing point predictions. This is an *approximate* overlay for pipeline
    display; the statistically rigorous single-model conformal procedure lives
    in :func:`src.models.conformal.calibrate_on_time_holdout`.
    """
    import numpy as np

    from src.models.conformal import ConformalRegressor, empirical_coverage

    years = np.sort(np.unique(np.asarray(year_groups_train)))
    if len(years) <= n_cal_years:
        return merged_df

    cal_years = set(years[-n_cal_years:])
    prop_mask = ~np.isin(year_groups_train, list(cal_years))
    cal_mask = np.isin(year_groups_train, list(cal_years))
    X_prop = X_train[prop_mask]
    y_prop = np.asarray(y_train)[prop_mask]
    X_cal = X_train[cal_mask]
    y_cal = np.asarray(y_train)[cal_mask]

    base = _fit_base_for_conformal()
    wrap = ConformalRegressor(base, alpha=alpha)
    wrap.fit(X_prop, y_prop)
    wrap.calibrate(X_cal, y_cal)

    X_all, _ = build_xy(merged_df, features, target)
    center = np.asarray(model.predict(X_all), dtype=float)
    merged_df["Pred_Yield_Lo"] = center - wrap.q_hat
    merged_df["Pred_Yield_Hi"] = center + wrap.q_hat

    # Optional diagnostic: empirical coverage on the temporal test rows.
    test_mask = merged_df["Year"] >= test_year_min
    if test_mask.any():
        y_true = merged_df.loc[test_mask, target].to_numpy(dtype=float)
        lo = merged_df.loc[test_mask, "Pred_Yield_Lo"].to_numpy(dtype=float)
        hi = merged_df.loc[test_mask, "Pred_Yield_Hi"].to_numpy(dtype=float)
        cov = empirical_coverage(y_true, lo, hi)
        print(f"[conformal] test coverage = {cov:.4f}, q_hat = {wrap.q_hat:.2f}")
    return merged_df


def _flag_wide_interval(width, center, relative_threshold: float = 0.5) -> "pd.Series":
    """Return a boolean Series flagging rows whose interval is wide vs. center.

    ``relative_threshold`` is the width/center ratio above which an interval is
    considered "wide" -- many such rows means the associated vulnerability label
    carries low confidence. The returned Series preserves the input index.
    """
    import numpy as np
    import pandas as pd

    width = pd.Series(width).astype(float)
    center = pd.Series(center).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = width / center.replace(0, np.nan)
    ratio = ratio.where(ratio.notna() & np.isfinite(ratio))
    flag = ratio > relative_threshold
    return flag.fillna(False).astype(bool).reset_index(drop=True)


def _train_model(model_name: str, X_train, y_train, **params):
    """Train the named model. ``random_forest`` is the baseline; ``xgboost``
    is the E03 alternative. Only these two are wired into the pipeline."""
    if model_name == "random_forest":
        return train_random_forest(X_train, y_train, **params)
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        params = dict(params)
        params.setdefault("random_state", 42)
        params.setdefault("n_estimators", 300)
        params.setdefault("verbosity", 0)
        params.setdefault("n_jobs", -1)
        return XGBRegressor(**params).fit(X_train, y_train)
    raise ValueError(f"Unsupported model: {model_name!r}")

def run(
    cutoff_year: int = 2014,
    output_csv: Path | None = None,
    out_dir: Path | None = None,
    model_params: dict | None = None,
    model: str = "random_forest",
    conformal: bool = True,
) -> dict:
    """Execute the full pipeline and return key outputs.

    ``model_params`` optionally overrides the model hyper-parameters
    (n_estimators, max_depth, random_state for RF; corresponding names for
    XGBoost). ``model`` selects the estimator: ``"random_forest"`` (default)
    or ``"xgboost"`` (E03). ``conformal`` toggles the additive E05 prediction-
    interval columns (default on; point predictions and metrics are unchanged).
    """
    model_params = model_params or {}

    crop_df = load_crop_data()
    nasa_df = load_nasa_data()

    # --- E07 validation: fail loudly before any further processing ---
    features = default_features()
    validation = validate_all(crop_df, nasa_df, features=features)
    if validation["warnings"]:
        for w in validation["warnings"]:
            print(f"[validation] {w}")

    nasa_yearly = aggregate_nasa_yearly(nasa_df)
    merged_df = merge_datasets(crop_df, nasa_yearly)

    target = "Yield_kg_per_ha"

    X_train, X_test, y_train, y_test = temporal_split(
        merged_df, features, target, cutoff_year
    )
    print(f"Train rows: {len(X_train)} (years < {cutoff_year})")
    print(f"Test  rows: {len(X_test)} (years >= {cutoff_year})")

    model_obj = _train_model(model, X_train, y_train, **model_params)
    y_pred = model_obj.predict(X_test)
    metrics = report_metrics(y_test, y_pred)

    merged_df["Predicted_Yield"] = model_obj.predict(build_xy(merged_df, features)[0])

    # --- E05: additive split-conformal prediction intervals (point preds unchanged) ---
    if conformal:
        year_groups_train = merged_df.loc[
            merged_df["Year"] < cutoff_year, "Year"
        ].to_numpy()
        merged_df = _add_conformal_intervals(
            merged_df, model_obj, X_train, y_train,
            year_groups_train, features,
            target=target, test_year_min=cutoff_year,
        )
        merged_df["Pred_Yield_Width"] = (
            merged_df["Pred_Yield_Hi"] - merged_df["Pred_Yield_Lo"]
        )
        merged_df["has_wide_interval"] = _flag_wide_interval(
            merged_df["Pred_Yield_Width"], merged_df["Predicted_Yield"]
        ).to_numpy()

    merged_df["Resilience_Index"] = resilience_index(
        merged_df["Yield_kg_per_ha"], merged_df["Predicted_Yield"]
    )
    merged_df["Resilience_Class"] = merged_df["Resilience_Index"].apply(
        resilience_class
    )

    agg_map = {
        "Min_Temp": ("Temperature_C", lambda x: x.quantile(0.05)),
        "Max_Temp": ("Temperature_C", lambda x: x.quantile(0.95)),
        "Avg_Temp": ("Temperature_C", "mean"),
        "Rainfall": ("Rainfall_mm", "mean"),
        "Actual_Yield": ("Yield_kg_per_ha", "mean"),
        "Predicted_Yield": ("Predicted_Yield", "mean"),
        "Resilience_Index": ("Resilience_Index", "mean"),
    }
    if conformal:
        agg_map["Pred_Yield_Lo"] = ("Pred_Yield_Lo", "mean")
        agg_map["Pred_Yield_Hi"] = ("Pred_Yield_Hi", "mean")
        agg_map["Pred_Yield_Width"] = ("Pred_Yield_Width", "mean")
        agg_map["has_wide_interval"] = ("has_wide_interval", "mean")

    # NOTE: pass agg_map via **kwargs. Passing a dict *variable* positionally
    # with (col, func) values trips a pandas 2.2.x handling quirk, whereas the
    # inline keyword form is correct.
    summary = (
        merged_df.groupby(["Year", "State Name", "Dist Name", "Crop"])
        .agg(**agg_map)
        .reset_index()
    )
    summary["Resilience_Class"] = summary["Resilience_Index"].apply(resilience_class)
    round_map = {
        "Min_Temp": 1,
        "Max_Temp": 1,
        "Avg_Temp": 1,
        "Rainfall": 0,
        "Actual_Yield": 0,
        "Predicted_Yield": 0,
        "Resilience_Index": 2,
    }
    if conformal:
        round_map.update({
            "Pred_Yield_Lo": 0,
            "Pred_Yield_Hi": 0,
            "Pred_Yield_Width": 0,
            "has_wide_interval": 2,
        })
    summary = summary.round(round_map)

    out_dir = out_dir or PROCESSED_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_csv or (out_dir / "final_crop_resilience_district_year.csv")
    summary.to_csv(output_csv, index=False)
    print(f"Wrote {output_csv} ({len(summary)} rows)")

    imgs = out_dir  # re-use processed dir for plots to keep docs/images clean
    plot_yield_vs_rainfall(summary, imgs / "yield_vs_rainfall.png")
    plot_feature_importance(model_obj, features, imgs / "feature_importance.png")
    plot_resilience_distribution(summary, imgs / "resilience_distribution.png")

    # --- E05 uncertainty: interval width by crop (additive) ---
    if conformal:
        from src.visualization.uncertainty_plots import plot_interval_by_crop

        plot_interval_by_crop(
            summary, imgs / "interval_width_by_crop.png",
            crop_col="Crop", width_col="Pred_Yield_Width",
        )

    # --- E06 climate indicators (separate from model output) ---
    indicators_path = out_dir / "climate_indicators.csv"
    indicators_df = write_climate_indicators(nasa_df, indicators_path)
    print(f"Wrote {indicators_path} ({len(indicators_df)} rows, E06 indicators)")
    plot_spi_timeseries(indicators_df, imgs / "spi_timeseries.png")
    plot_anomaly_heatmap(indicators_df, imgs / "anomaly_heatmap.png")
    plot_climate_trend_table(indicators_df, imgs / "climate_trend_table.png")

    print(f"Wrote plots to {imgs}")

    return {
        "model": model_obj,
        "metrics": metrics,
        "summary": summary,
        "climate_indicators": indicators_df,
        "validation": validation,
        "conformal": bool(conformal and "Pred_Yield_Lo" in merged_df.columns),
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