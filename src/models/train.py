"""Random Forest yield model training."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 300,
    max_depth: int = 15,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Train a Random Forest regressor on yield features."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model