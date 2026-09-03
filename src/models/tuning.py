"""Hyperparameter optimization for RF and XGBoost (E03).

Uses ``RandomizedSearchCV`` with a year-grouped time-aware CV splitter so
that tuning signal never leaks across the time axis.  Each search evaluates
a fixed number of random parameter combinations on expanding-window folds
where the test set is always strictly *later* in time than the training set.

Model comparison
----------------
``compare_models()`` trains four models under identical year-grouped CV and
reports R², RMSE, and MAE for each:

1. Default Random Forest (baseline)
2. Tuned Random Forest
3. Default XGBoost
4. Tuned XGBoost

Tuned models are obtained by running ``RandomizedSearchCV`` on the full
dataset first (to find the best hyperparameters), then re-evaluated under
the same year-grouped CV splits as the defaults.  This ensures all four
numbers are comparable.

Usage from Python::

    from src.models.tuning import compare_models
    results = compare_models(merged_df, seed=42)
    for name, m in results.items():
        print(f"{name}: R²={m['r2']:.4f} RMSE={m['rmse']:.2f}")

Usage as a standalone script::

    python -m src.models.tuning [--seed 42] [--n-iter 10] [--n-test-years 3]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from src.features.engineering import build_xy, default_features, TARGET
from src.models.cross_validation import YearGroupCV

# ---------------------------------------------------------------------------
# Search spaces
# ---------------------------------------------------------------------------

RF_SEARCH_SPACE: dict = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [5, 10, 15, 20, None],
    "min_samples_split": [2, 5, 10, 20],
}

XGB_SEARCH_SPACE: dict = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2, 0.3],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _evaluate_cv(model, X, y, year_groups, cv):
    """Evaluate a model under year-grouped CV, returning mean metrics."""
    r2_scores, rmse_scores, mae_scores = [], [], []
    for train_idx, test_idx in cv.split(X, groups=year_groups):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        r2_scores.append(r2_score(y_te, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae_scores.append(mean_absolute_error(y_te, y_pred))
    return {
        "r2": float(np.mean(r2_scores)),
        "rmse": float(np.mean(rmse_scores)),
        "mae": float(np.mean(mae_scores)),
    }


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    year_groups: np.ndarray | None = None,
    n_iter: int = 15,
    cv_n_test_years: int = 3,
    random_state: int = 42,
) -> dict:
    """Tune Random Forest with year-grouped CV.

    Parameters
    ----------
    X : Feature matrix for model fitting.
    y : Target vector.
    year_groups : Array of year values, one per row of ``X``.  Required for
        time-aware CV; the splitter groups rows by year so validation years
        never leak into training years.
    n_iter : Number of random parameter combinations to try.
    cv_n_test_years : Number of years held out per CV fold.
    random_state : Seed for reproducibility.

    Returns
    -------
    dict with ``model`` (retrained on all data with best params),
    ``best_params``, and ``search_results`` (DataFrame of all trials).
    """
    from sklearn.ensemble import RandomForestRegressor

    if year_groups is None:
        raise ValueError("year_groups must be provided for time-aware CV")

    cv = YearGroupCV(n_test_years=cv_n_test_years)
    base = RandomForestRegressor(random_state=random_state, n_jobs=-1)

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=RF_SEARCH_SPACE,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X, y, groups=year_groups)

    # Retrain best model on all data for downstream use
    best = RandomForestRegressor(**search.best_params_, random_state=random_state, n_jobs=-1)
    best.fit(X, y)

    return {
        "model": best,
        "best_params": dict(search.best_params_),
        "search_results": pd.DataFrame(search.cv_results_),
    }


def tune_xgboost(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    year_groups: np.ndarray | None = None,
    n_iter: int = 15,
    cv_n_test_years: int = 3,
    random_state: int = 42,
) -> dict:
    """Tune XGBoost with year-grouped CV.

    Parameters are analogous to :func:`tune_random_forest`.

    Returns
    -------
    dict with ``model``, ``best_params``, and ``search_results``.
    """
    from xgboost import XGBRegressor

    if year_groups is None:
        raise ValueError("year_groups must be provided for time-aware CV")

    cv = YearGroupCV(n_test_years=cv_n_test_years)
    base = XGBRegressor(
        random_state=random_state,
        n_estimators=300,
        verbosity=0,
        n_jobs=-1,
    )

    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=XGB_SEARCH_SPACE,
        n_iter=n_iter,
        cv=cv,
        scoring="r2",
        random_state=random_state,
        n_jobs=-1,
        error_score="raise",
    )
    search.fit(X, y, groups=year_groups)

    best = XGBRegressor(
        **search.best_params_,
        random_state=random_state,
        n_estimators=300,
        verbosity=0,
        n_jobs=-1,
    )
    best.fit(X, y)

    return {
        "model": best,
        "best_params": dict(search.best_params_),
        "search_results": pd.DataFrame(search.cv_results_),
    }


# ---------------------------------------------------------------------------
# Model comparison
# ---------------------------------------------------------------------------

def compare_models(
    merged_df: pd.DataFrame,
    features: list[str] | None = None,
    *,
    n_iter: int = 15,
    cv_n_test_years: int = 3,
    random_state: int = 42,
) -> dict[str, dict]:
    """Compare default and tuned RF/XGBoost under identical year-grouped CV.

    All four models are evaluated using the same ``YearGroupCV`` splitter.
    Default models are trained fresh in each fold.  Tuned models use
    hyperparameters found by ``RandomizedSearchCV`` on the full dataset,
    then re-evaluated under the same fold structure.

    Parameters
    ----------
    merged_df : Full merged DataFrame (must contain ``Year`` column).
    features : Feature columns (defaults to :func:`default_features`).
    n_iter : Random search iterations per tuned model.
    cv_n_test_years : Test window size for the CV splitter.
    random_state : Seed for reproducibility.

    Returns
    -------
    Nested dict ``{model_name: {r2, rmse, mae, best_params}}``.
    """
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    features = features or default_features()
    X, y = build_xy(merged_df, features, TARGET)
    year_groups = merged_df["Year"].to_numpy()
    cv = YearGroupCV(n_test_years=cv_n_test_years)

    results: dict[str, dict] = {}

    # --- Default Random Forest ---
    default_rf = RandomForestRegressor(
        n_estimators=300, max_depth=15, random_state=random_state, n_jobs=-1,
    )
    rf_metrics = _evaluate_cv(default_rf, X, y, year_groups, cv)
    results["default_rf"] = {**rf_metrics, "best_params": {}, "model": default_rf}

    # --- Default XGBoost ---
    default_xgb = XGBRegressor(
        n_estimators=300, random_state=random_state, verbosity=0, n_jobs=-1,
    )
    xgb_metrics = _evaluate_cv(default_xgb, X, y, year_groups, cv)
    results["default_xgb"] = {**xgb_metrics, "best_params": {}, "model": default_xgb}

    # --- Tuned Random Forest ---
    rf_tuned = tune_random_forest(
        X, y,
        year_groups=year_groups,
        n_iter=n_iter,
        cv_n_test_years=cv_n_test_years,
        random_state=random_state,
    )
    tuned_rf_model = rf_tuned["model"]
    tuned_rf_metrics = _evaluate_cv(tuned_rf_model, X, y, year_groups, cv)
    results["tuned_rf"] = {**tuned_rf_metrics, "best_params": rf_tuned["best_params"], "model": tuned_rf_model}

    # --- Tuned XGBoost ---
    xgb_tuned = tune_xgboost(
        X, y,
        year_groups=year_groups,
        n_iter=n_iter,
        cv_n_test_years=cv_n_test_years,
        random_state=random_state,
    )
    tuned_xgb_model = xgb_tuned["model"]
    tuned_xgb_metrics = _evaluate_cv(tuned_xgb_model, X, y, year_groups, cv)
    results["tuned_xgb"] = {**tuned_xgb_metrics, "best_params": xgb_tuned["best_params"], "model": tuned_xgb_model}

    # --- Summary table ---
    print("\n=== Model Comparison (year-grouped CV) ===")
    print(f"{'Model':<20s} {'R²':>8s} {'RMSE':>10s} {'MAE':>10s}")
    print("-" * 52)
    for name, m in results.items():
        print(f"{name:<20s} {m['r2']:>8.4f} {m['rmse']:>10.2f} {m['mae']:>10.2f}")

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    """CLI entry point for running model comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-iter", type=int, default=15, help="Search iterations per model")
    parser.add_argument("--n-test-years", type=int, default=3, help="Test window in years")
    args = parser.parse_args(argv)

    from src.data.loader import (
        aggregate_nasa_yearly, load_crop_data, load_nasa_data, merge_datasets,
    )

    crop_df = load_crop_data()
    nasa_df = load_nasa_data()
    nasa_yearly = aggregate_nasa_yearly(nasa_df)
    merged_df = merge_datasets(crop_df, nasa_yearly)

    results = compare_models(
        merged_df,
        n_iter=args.n_iter,
        cv_n_test_years=args.n_test_years,
        random_state=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
