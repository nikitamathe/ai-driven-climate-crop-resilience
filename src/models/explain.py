"""SHAP-based yield-model explanation (E04).

This module explains the *tuned* Random Forest and XGBoost models from E03
using `shap.TreeExplainer`. It computes:

* per-feature SHAP values on a held-out sample,
* mean |SHAP| attributions (a magnitude ranking of how much each feature
  moves the prediction away from the expected value),
* scikit-learn permutation importance on the same holdout, for a direct
  attribution-vs-perturbation honesty comparison,
* leakage-safe attributions: `tree_path_dependent` SHAP (no interventional
  background reference set) computed on the held-out temporal test set, so
  no test-year information can encode future signal.

Scientific honesty
------------------
SHAP values are **attributions**, not **causal** statements. They describe
how each feature contributed to a specific model's predictions — they do not
establish cause-and-effect relationships between climate variables and yield.
This module (and the README) states that explicitly.

``shap`` is imported **lazily** inside the functions that need it, so the core
pipeline and the E01-E03/E06-E10 tests never require it to be installed. Only
this module's functions (and :file:`tests/test_shap.py`) depend on SHAP.

Usage from Python::

    from src.models.explain import explain_models
    result = explain_models(models, X_test, y_test)

Usage as a standalone script::

    python -m src.models.explain --out-dir data/processed/shap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.engineering import default_features, temporal_split

# Features we must produce dependence plots for (per E04 roadmap).
DEPENDENCE_FEATURES = ("AvgTemp", "Rainfall_mm")


def _require_shap():
    """Import and return the ``shap`` module, raising a clear error if absent."""
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - exercised only w/o shap
        raise ImportError(
            "SHAP is required for model explainability (E04). "
            "Install it with: pip install -r requirements.txt"
        ) from exc
    return shap


def compute_shap(model, X: pd.DataFrame) -> dict:
    """Compute TreeExplainer SHAP values for ``model`` on ``X``.

    Uses ``feature_perturbation="tree_path_dependent"`` — the standard,
    exact mode for tree ensembles and the only mode fully supported by
    XGBoost >= 3.x. Because this mode explains each prediction from the
    model's own training-tree path statistics, it needs **no background
    reference distribution**, which also means no temporal-leakage surface:
    attributions never draw on holdout data.

    Parameters
    ----------
    model : fitted tree-based estimator (RandomForestRegressor / XGBRegressor).
    X : DataFrame of rows to explain.

    Returns
    -------
    dict with ``values`` (ndarray of shape ``(n_samples, n_features)``),
    ``expected_value`` (scalar baseline), and ``feature_names`` (list).
    """
    shap = _require_shap()

    import numpy as np

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")

    raw = explainer.shap_values(np.asarray(X))

    # For single-output regressors shap_values returns a 2-D array directly;
    # guard against any 3-D multi-output edge case by selecting the first output.
    if isinstance(raw, list):
        raw = raw[0]

    expected = explainer.expected_value
    if isinstance(expected, (list, np.ndarray)) and np.ndim(expected) > 0:
        expected = expected[0]

    return {
        "values": np.asarray(raw, dtype=float),
        "expected_value": float(np.asarray(expected).reshape(-1)[0]),
        "feature_names": list(X.columns),
    }


def mean_abs_shap(shap_result: dict, feature_names: list[str] | None = None) -> dict[str, float]:
    """Return per-feature mean |SHAP| attribution magnitudes.

    Higher values indicate a feature that, on average, moves predictions more
    strongly (in either direction) away from the expected value. This is an
    *attribution* ranking, not a causal or performance claim.
    """
    features = feature_names or shap_result["feature_names"]
    magnitudes = np.mean(np.abs(shap_result["values"]), axis=0)
    return {name: float(mag) for name, mag in zip(features, magnitudes)}


def permutation_importance(
    model, X: pd.DataFrame, y: np.ndarray, *, n_repeats: int = 5, random_state: int = 42
) -> dict[str, float]:
    """Return per-feature scikit-learn permutation importance on ``X``/``y``.

    Uses the default test R² scoring on the provided sample. For the honesty
    comparison this sample must be the same holdout used for SHAP.
    """
    from sklearn.inspection import permutation_importance as sk_perm

    result = sk_perm(
        model,
        np.asarray(X),
        np.asarray(y),
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )
    return {
        name: float(imp)
        for name, imp in zip(X.columns, result["importances_mean"])
    }


def explain_models(
    models: dict[str, object],
    X_explain: pd.DataFrame,
    y_explain: np.ndarray | None = None,
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> dict[str, dict]:
    """Explain each model with SHAP + permutation importance on a clean holdout.

    Parameters
    ----------
    models : ``{name: fitted_model}``, e.g. ``{"tuned_rf": ..., "tuned_xgb": ...}``.
    X_explain : holdout sample (e.g. the temporal test set) to explain.
    y_explain : optional holdout target, required for permutation importance.
    n_repeats : permutation importance repetitions.
    random_state : seed for permutation importance.

    Returns
    -------
    ``{model_name: {"shap": ..., "mean_abs_shap": {...}, "permutation_importance": {...}}}``

    Attributions use ``tree_path_dependent`` SHAP (no interventional background),
    and permutation importance uses the same held-out ``X_explain``/``y_explain``,
    so no test-year information shapes the explanation.
    """
    results: dict[str, dict] = {}
    for name, model in models.items():
        shap_res = compute_shap(model, X_explain)
        mean_abs = mean_abs_shap(shap_res)
        perm = (
            permutation_importance(
                model, X_explain, y_explain,
                n_repeats=n_repeats, random_state=random_state,
            )
            if y_explain is not None
            else {}
        )
        results[name] = {
            "shap": shap_res,
            "mean_abs_shap": mean_abs,
            "permutation_importance": perm,
        }
    return results


def consistency_report(results: dict[str, dict], top_n: int = 5) -> str:
    """Return a short text summary of top-feature agreement across models.

    Compares the mean |SHAP| rankings of each explained model. This is the
    E04 "consistency of top features across RF/XGB" evaluation.
    """
    lines: list[str] = []
    lines.append(f"Top-{top_n} features by mean |SHAP| per model:")
    ranked: dict[str, list[str]] = {}
    for name, res in results.items():
        ranked[name] = _top_features(res["mean_abs_shap"], top_n)
        lines.append(f"  {name}: {', '.join(ranked[name])}")

    names = list(ranked.keys())
    if len(names) >= 2:
        a, b = set(ranked[names[0]]), set(ranked[names[1]])
        overlap = sorted(a & b)
        lines.append(f"Top-{top_n} overlap between models: {', '.join(overlap) or 'none'}")
    return "\n".join(lines)


def _top_features(scores: dict[str, float], top_n: int) -> list[str]:
    return [f for f, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)][:top_n]


def _to_frame(results: dict[str, dict]) -> pd.DataFrame:
    """Build a tidy DataFrame of mean|SHAP| and permutation importance."""
    rows = []
    for model_name, res in results.items():
        for feature in res["mean_abs_shap"]:
            rows.append({
                "model": model_name,
                "feature": feature,
                "mean_abs_shap": res["mean_abs_shap"][feature],
                "permutation_importance": res["permutation_importance"].get(feature, np.nan),
            })
    return pd.DataFrame(rows)


def write_shap_outputs(
    results: dict[str, dict],
    out_dir: Path | str,
    feature_names: list[str] | None = None,
) -> list[Path]:
    """Write SHAP CSV + plots and return the created file paths.

    Produces, for each explained model: a beeswarm summary plot, dependence
    plots for temperature (``AvgTemp``) and rainfall (``Rainfall_mm``), and a
    mean-|SHAP|-vs-permutation-importance bar chart. Also writes a tidy
    ``feature_shap_summary.csv`` and a ``consistency_report.txt``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.visualization.shap_plots import (
        plot_beeswarm,
        plot_dependence,
        plot_mean_abs_shap_vs_permutation,
    )

    written: list[Path] = []

    for name, res in results.items():
        shap_res = res["shap"]
        base = out_dir / name
        base.mkdir(parents=True, exist_ok=True)

        beeswarm = base / "beeswarm.png"
        plot_beeswarm(shap_res, beeswarm)
        written.append(beeswarm)

        for feat in DEPENDENCE_FEATURES:
            if feat in shap_res["feature_names"]:
                dep = base / f"dependence_{feat.lower().replace('%', 'pct')}.png"
                plot_dependence(shap_res, feat, dep)
                written.append(dep)

        bar = base / "shap_vs_permutation_importance.png"
        plot_mean_abs_shap_vs_permutation(res, bar, feature_names or shap_res["feature_names"])
        written.append(bar)

    summary_df = _to_frame(results)
    summary_path = out_dir / "feature_shap_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    written.append(summary_path)

    report_path = out_dir / "consistency_report.txt"
    report_path.write_text(consistency_report(results) + "\n", encoding="utf-8")
    written.append(report_path)

    return written


def _build_models_and_holdout(cutoff_year: int = 2014):
    """Load data, build the tuned RF + XGB models (E03) and a leak-safe holdout.

    This is the self-contained entry used by the CLI: it reproduces the tuned
    models from `src.models.tuning` and evaluates attributions on the temporal
    test set (years >= cutoff), consistent with the reported metrics.
    """
    from src.data.loader import (
        aggregate_nasa_yearly, load_crop_data, load_nasa_data, merge_datasets,
    )
    from src.models.tuning import tune_random_forest, tune_xgboost
    from src.features.engineering import TARGET

    crop_df = load_crop_data()
    nasa_df = load_nasa_data()
    nasa_yearly = aggregate_nasa_yearly(nasa_df)
    merged_df = merge_datasets(crop_df, nasa_yearly)

    features = default_features()
    X_train, X_test, y_train, y_test = temporal_split(
        merged_df, features, TARGET, cutoff_year
    )

    year_groups = merged_df["Year"].to_numpy()
    train_mask = merged_df["Year"] < cutoff_year
    year_groups_train = year_groups[train_mask]

    rf = tune_random_forest(
        X_train, y_train, year_groups=year_groups_train,
        n_iter=15, cv_n_test_years=3, random_state=42,
    )["model"]
    xgb = tune_xgboost(
        X_train, y_train, year_groups=year_groups_train,
        n_iter=15, cv_n_test_years=3, random_state=42,
    )["model"]

    models = {"tuned_rf": rf, "tuned_xgb": xgb}
    return models, X_test, y_test.to_numpy()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for SHAP explainability."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff-year", type=int, default=2014)
    parser.add_argument("--out-dir", type=str, default="data/processed/shap")
    parser.add_argument("--n-repeats", type=int, default=5)
    args = parser.parse_args(argv)

    models, X_test, y_test = _build_models_and_holdout(args.cutoff_year)
    results = explain_models(models, X_test, y_test, n_repeats=args.n_repeats)

    print("\n=== SHAP explainability (E04) ===")
    print(consistency_report(results))

    written = write_shap_outputs(results, args.out_dir)
    print(f"Wrote {len(written)} SHAP artifacts to {args.out_dir}")
    for p in written:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
