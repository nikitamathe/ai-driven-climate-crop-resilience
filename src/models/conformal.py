"""Split-conformal prediction intervals (E05).

Adds **prediction intervals** -- a 90% credible/prediction band -- around the
point yield predictions of the existing tree models. A resilience class
computed from a bare point value carries no confidence; a district whose
interval is very wide should not be labeled "Vulnerable" with the same
certainty as one whose interval is narrow. This module surfaces that
uncertainty.

Approach
--------
Split-conformal prediction (regression) around the best point model:

1. Split the training data (time-aware) into a *proper training* set and a
   *calibration* set.
2. Fit the base regressor on the proper-training rows.
3. Compute non-conformity scores on the calibration rows,
   ``s_i = |y_i - yhat_i|`` (absolute residuals).
4. Take ``q_hat`` = the ``ceil((n+1)*(1-alpha))/n`` empirical quantile of the
   scores.
5. For a test prediction ``yhat`` the interval is ``[yhat-q_hat, yhat+q_hat]``.

This yields **distribution-free marginal coverage**: for the whole test set,
on average, about ``(1-alpha)`` of the true values fall inside the interval to
a given (non-exchangeable data) approximation.

Scientific caveat (must be stated in the README)
------------------------------------------------
Coverage is **marginal**, not conditional: it holds on average over the test
set, not for any single district/crop/year. A pure temporal test split also
violates strict exchangeability with the calibration years; we mitigate by
calibrating on the most-recent training years (time-adjacent to the test
window) and reporting the *empirical* coverage on held-out test years rather
than claiming the nominal rate. Uncertainty is a confidence diagnostic, not a
causal statement.

No new dependencies
-------------------
Pure scikit-learn + numpy. ``sklearn_quantile`` / ``scikit-garden`` are
deliberately NOT used (they add a fragile supply-chain dependency); the
absolute-residual split-conformal path is exact and dependency-free.

Usage from Python::

    from src.models.conformal import fit_conformal
    wrap = fit_conformal(base_model, X_proper, y_proper, X_cal, y_cal, alpha=0.10)
    center, low, high = wrap.predict(X_test)

Usage as a standalone script::

    python -m src.models.conformal --alpha 0.10 --cal-years 3
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


class ConformalRegressor:
    """Split-conformal wrapper that extends a base regressor with intervals.

    Parameters
    ----------
    base_model : fitted or unfitted scikit-learn regressor with ``fit``/``predict``.
    alpha : float, coverage miscoverage rate (``1-alpha`` = target coverage).
    """

    def __init__(self, base_model, alpha: float = 0.10) -> None:
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.base_model = base_model
        self.alpha = float(alpha)
        self.q_hat: float | None = None
        self.calibration_scores: np.ndarray | None = None

    def fit(self, X_proper, y_proper):
        """Fit the base regressor on the proper-training rows only."""
        self.base_model.fit(X_proper, y_proper)
        return self

    def calibrate(self, X_cal, y_cal) -> None:
        """Compute non-conformity scores and the conformal quantile ``q_hat``.

        ``X_cal``/``y_cal`` must come from a *time-adjacent* calibration block
        that is strictly outside the test years to avoid temporal leakage.
        """
        n = len(X_cal)
        if n == 0:
            raise ValueError("calibration set is empty")
        residuals = np.abs(np.asarray(y_cal) - self.base_model.predict(X_cal))
        self.calibration_scores = residuals
        # Conformal quantile with a small finite-sample correction.
        q_index = int(np.ceil((n + 1) * (1 - self.alpha)))
        q_index = min(q_index, n)
        self.q_hat = float(np.sort(residuals)[q_index - 1])
        return self

    def predict(self, X):
        """Return ``(center, low, high)`` prediction bands for ``X``."""
        if self.q_hat is None:
            raise RuntimeError(
                "ConformalRegressor must be calibrated before predict()"
            )
        center = np.asarray(self.base_model.predict(X), dtype=float)
        low = center - self.q_hat
        high = center + self.q_hat
        return center, low, high


def fit_conformal(
    base_model,
    X_proper,
    y_proper,
    X_cal,
    y_cal,
    *,
    alpha: float = 0.10,
) -> ConformalRegressor:
    """Fit and calibrate a split-conformal model in one call.

    Parameters
    ----------
    base_model : regressor to wrap.
    X_proper, y_proper : proper-training rows the base model is fit on.
    X_cal, y_cal : conformal calibration rows (recent, out-of-test-sample).
    alpha : miscoverage rate (default 0.10 -> 90% intervals).

    Returns
    -------
    Calibrated :class:`ConformalRegressor`.
    """
    wrap = ConformalRegressor(base_model, alpha=alpha)
    wrap.fit(X_proper, y_proper)
    wrap.calibrate(X_cal, y_cal)
    return wrap


def calibrate_on_time_holdout(
    base_model,
    X: np.ndarray | "pd.DataFrame",
    y: np.ndarray,
    year_groups: np.ndarray,
    *,
    n_cal_years: int = 3,
    alpha: float = 0.10,
):
    """Fit + calibrate conformal using a time-aware calibration split.

    Uses :class:`src.models.cross_validation.YearGroupCV` indirectly: the most
    recent ``n_cal_years`` of the *training* period become the calibration set,
    and all earlier training years become the proper-training set. This keeps
    the calibration block time-adjacent to (and disjoint from) the test window,
    so validation years never leak into the model or the calibration scores.

    Parameters
    ----------
    base_model : regressor to wrap.
    X, y : full *training* feature matrix and target.
    year_groups : year per row of ``X``.
    n_cal_years : number of most-recent training years reserved for calibration.
    alpha : miscoverage rate.

    Returns
    -------
    Calibrated :class:`ConformalRegressor` and the ``(X_proper, X_cal)`` split
    metadata as ``(wrapper, split)`` where ``split`` records calibration years.
    """
    from src.models.cross_validation import YearGroupCV

    if year_groups is None:
        raise ValueError("year_groups must be provided for time-aware calibration")

    years = np.sort(np.unique(np.asarray(year_groups)))
    if len(years) <= n_cal_years:
        raise ValueError(
            f"need more than {n_cal_years} distinct years for a calibration split"
        )

    cal_years = set(years[-n_cal_years:])
    proper_mask = ~np.isin(year_groups, list(cal_years))
    cal_mask = np.isin(year_groups, list(cal_years))

    X_proper = X[proper_mask]
    y_proper = np.asarray(y)[proper_mask]
    X_cal = X[cal_mask]
    y_cal = np.asarray(y)[cal_mask]

    wrap = fit_conformal(
        base_model, X_proper, y_proper, X_cal, y_cal, alpha=alpha
    )
    split = {"calibration_years": sorted(cal_years), "n_cal": len(X_cal)}
    return wrap, split


def empirical_coverage(y_true, low, high) -> float:
    """Fraction of true values inside the ``[low, high]`` intervals."""
    y_true = np.asarray(y_true, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    inside = (y_true >= low) & (y_true <= high)
    return float(inside.mean()) if len(y_true) else float("nan")


def mean_interval_width(low, high) -> float:
    """Mean width of the prediction intervals, ``mean(high - low)``."""
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    return float(np.mean(high - low)) if len(low) else float("nan")


def _build_data_and_holdout(cutoff_year: int = 2014):
    """Load data and produce training + test splits for the conformal CLI."""
    from src.data.loader import (
        aggregate_nasa_yearly, load_crop_data, load_nasa_data, merge_datasets,
    )
    from src.features.engineering import TARGET, default_features, temporal_split

    crop_df = load_crop_data()
    nasa_df = load_nasa_data()
    nasa_yearly = aggregate_nasa_yearly(nasa_df)
    merged_df = merge_datasets(crop_df, nasa_yearly)

    features = default_features()
    X_train, X_test, y_train, y_test = temporal_split(
        merged_df, features, TARGET, cutoff_year
    )
    year_groups_train = merged_df.loc[
        merged_df["Year"] < cutoff_year, "Year"
    ].to_numpy()
    return X_train, y_train.to_numpy(), X_test, y_test.to_numpy(), year_groups_train


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: build a conformal model and report coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-year", type=int, default=2014)
    parser.add_argument("--cal-years", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--out-dir", type=str, default="data/processed/conformal")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=15)
    args = parser.parse_args(argv)

    from sklearn.ensemble import RandomForestRegressor

    X_train, y_train, X_test, y_test, year_groups_train = _build_data_and_holdout(
        args.cutoff_year
    )

    base = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
        n_jobs=-1,
    )
    wrap, split = calibrate_on_time_holdout(
        base, X_train.to_numpy(), y_train, year_groups_train,
        n_cal_years=args.cal_years, alpha=args.alpha,
    )

    center, low, high = wrap.predict(np.asarray(X_test))
    cov = empirical_coverage(y_test, low, high)
    width = mean_interval_width(low, high)

    print("\n=== E05 split-conformal coverage (test years) ===")
    print(f"q_hat       = {wrap.q_hat:.2f}")
    print(f"cal years   = {split['calibration_years']} (n={split['n_cal']})")
    print(f"empirical coverage = {cov:.4f}  (target 1-alpha = {1 - args.alpha:.2f})")
    print(f"mean interval width = {width:.2f}")

    from src.visualization.uncertainty_plots import write_conformal_report

    paths = write_conformal_report(
        y_test, center, low, high,
        X_test, out_dir=args.out_dir,
        width=width, coverage=cov, q_hat=wrap.q_hat,
    )
    for p in paths:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
