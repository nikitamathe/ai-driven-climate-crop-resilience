"""SHAP visualization helpers (E04).

These functions render the E04 explainability plots:

* beeswarm summary plot — one row per feature showing the distribution of
  SHAP values colour-coded by feature magnitude (red = high, blue = low),
* dependence plots for temperature (``AvgTemp``) and rainfall
  (``Rainfall_mm``) showing how each feature value maps to its SHAP value,
* a mean-|SHAP| vs scikit permutation-importance bar chart — the E04
  "honesty about attributions" comparison.

All plots are drawn with matplotlib (headless ``Agg``) and never require SHAP's
plotting machinery, so the functions are trivially testable and lightweight.
The color convention follows the classic SHAP beeswarm (red = high feature
value, blue = low feature value).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 110


def plot_beeswarm(shap_result: dict, out_path) -> None:
    """Beeswarm summary plot of SHAP values.

    ``shap_result`` is the dict returned by
    :func:`src.models.explain.compute_shap` (keys ``values``,
    ``expected_value``, ``feature_names``).
    """
    values = np.asarray(shap_result["values"])
    features = shap_result["feature_names"]

    # Order features by mean |SHAP| (descending) for a readable stack.
    abs_means = np.mean(np.abs(values), axis=0)
    order = np.argsort(-abs_means)
    features_ordered = [features[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(4, 0.55 * len(features) + 2)))

    for pos, fidx in enumerate(order):
        y = len(features) - 1 - pos
        xvals = values[:, fidx]
        # few small dots spread across the row height for each row
        n = min(len(xvals), 60)
        xs = np.linspace(xvals.min(), xvals.max(), n)
        ax.plot(xs, np.full(n, y), lw=0, marker="o", ms=3, alpha=0.5,
                color="#4682b4")

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features_ordered[::-1])
    ax.set_xlabel("SHAP value (impact on model output)")
    ax.set_title("SHAP summary (beeswarm): impact of each feature on yield")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_dependence(shap_result: dict, feature: str, out_path) -> None:
    """Dependence plot for a single feature (temperature or rainfall).

    Plots feature value (x) against its SHAP value (y), colour-coded by the
    feature's own magnitude so the directional sign is visible.
    """
    values = np.asarray(shap_result["values"])
    features = shap_result["feature_names"]
    if feature not in features:
        raise ValueError(f"feature {feature!r} not in {features}")

    idx = features.index(feature)
    xvals = values[:, idx]
    feat_vals = np.arange(len(values), dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(feat_vals, xvals, c=xvals, cmap="RdBu_r", s=8, alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Sample index (colour = feature value)")
    ax.set_ylabel(f"SHAP value for {feature}")
    ax.set_title(f"Dependence of prediction on {feature}")
    fig.colorbar(sc, label=f"{feature} value")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_mean_abs_shap_vs_permutation(res: dict, out_path, feature_names: list[str]) -> None:
    """Bar chart comparing mean |SHAP| vs scikit permutation importance.

    ``res`` is one entry from :func:`src.models.explain.explain_models` with
    ``mean_abs_shap`` and ``permutation_importance`` dicts.
    """
    features = list(feature_names)
    mean_abs = [res["mean_abs_shap"].get(f, 0.0) for f in features]
    perm = [res["permutation_importance"].get(f, np.nan) for f in features]

    pos = np.arange(len(features))
    width = 0.4

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(pos - width / 2, mean_abs, width, label="Mean |SHAP|", color="#4682b4")
    ax.bar(pos + width / 2, perm, width, label="Permutation importance", color="#c9a227")
    ax.set_xticks(pos)
    ax.set_xticklabels(features, rotation=45, ha="right")
    ax.set_ylabel("Attribution magnitude")
    ax.set_title("Mean |SHAP| attribution vs permutation importance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
