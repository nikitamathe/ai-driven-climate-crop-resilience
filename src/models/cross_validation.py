"""Year-grouped cross-validation splitter (E03).

This module provides a scikit-learn-compatible CV splitter that groups rows
by their ``Year`` column, ensuring that *all rows from the same year* appear
exclusively in either the training or the test fold — never both.  This
prevents temporal leakage during hyperparameter tuning and model comparison.

The splitter is designed to work with ``RandomizedSearchCV`` (and any other
sklearn meta-estimator that accepts a ``cv`` parameter) via the standard
``split(X, y, groups)`` interface.

Example::

    from src.models.cross_validation import YearGroupCV
    splitter = YearGroupCV(n_test_years=3)
    for train_idx, test_idx in splitter.split(merged_df):
        X_train, y_train = build_xy(merged_df.iloc[train_idx])
        X_test,  y_test  = build_xy(merged_df.iloc[test_idx])
        ...
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class YearGroupCV:
    """Leave-one-year-out or expanding-window CV grouped by year.

    Parameters
    ----------
    n_test_years : int, default=1
        Number of most-recent years held out as the test set in each fold.
        With ``n_test_years=1`` each fold tests on a single year, training on
        all earlier years.  With ``n_test_years>1`` the last *k* years form the
        test set.

    minimum_train_years : int, default=3
        Minimum number of distinct years required in the training set.  Folds
        that would leave fewer than this many training years are skipped.
    """

    def __init__(self, n_test_years: int = 1, minimum_train_years: int = 3) -> None:
        if n_test_years < 1:
            raise ValueError(f"n_test_years must be >= 1, got {n_test_years}")
        if minimum_train_years < 1:
            raise ValueError(
                f"minimum_train_years must be >= 1, got {minimum_train_years}"
            )
        self.n_test_years = n_test_years
        self.minimum_train_years = minimum_train_years

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D102
        """Return the number of splitting iterations."""
        years = self._extract_years(X, groups)
        n_unique_years = len(np.unique(years))
        n_folds = n_unique_years - self.minimum_train_years - self.n_test_years + 1
        return max(n_folds, 0)

    def split(self, X=None, y=None, groups=None):  # noqa: D102
        """Generate (train_index, test_index) pairs.

        Parameters
        ----------
        X : unused (kept for sklearn API compatibility).
        y : unused (kept for sklearn API compatibility).
        groups : array-like of year values, required.

        Yields
        ------
        train_idx : ndarray of int
            Row indices for the training set.
        test_idx : ndarray of int
            Row indices for the test set.
        """
        years = self._extract_years(X, groups)
        unique_years = np.sort(np.unique(years))

        # Expanding-window: each fold tests on the last n_test_years years
        # available, with training on everything earlier.
        n_total = len(unique_years)
        # Number of folds = n_total - minimum_train_years - n_test_years + 1
        n_folds = n_total - self.minimum_train_years - self.n_test_years + 1

        for i in range(max(n_folds, 0)):
            test_start = self.minimum_train_years + i
            test_end = test_start + self.n_test_years
            test_years = set(unique_years[test_start:test_end])
            train_years = set(unique_years[:test_start])

            train_idx = np.where(
                np.isin(years, sorted(train_years))
            )[0]
            test_idx = np.where(
                np.isin(years, sorted(test_years))
            )[0]

            if len(train_idx) == 0 or len(test_idx) == 0:
                continue

            yield train_idx, test_idx

    def _extract_years(self, X, groups) -> np.ndarray:
        """Extract the Year column from *X* if present, else from *groups*.

        ``RandomizedSearchCV`` passes ``groups`` through to the splitter when
        the caller supplies them to ``fit(X, y, groups=...)``.  If groups are
        not supplied, fall back to reading the ``Year`` column from a
        DataFrame ``X`` (only meaningful in a direct-use scenario).
        """
        if groups is not None:
            return np.asarray(groups)
        if isinstance(X, pd.DataFrame):
            if "Year" in X.columns:
                return X["Year"].to_numpy()
            raise ValueError(
                "X is a DataFrame but has no 'Year' column. "
                "Pass years via the 'groups' parameter."
            )
        raise TypeError(
            f"X must be a pandas DataFrame with a 'Year' column, got {type(X).__name__}"
        )
