"""E05 uncertainty visualization helpers.

Renders the split-conformal prediction-interval diagnostics:

* a per-row / per-crop breakdown of prediction intervals,
* interval-width-by-crop bars,
* a "Resilience with 90% CI" bar chart showing point resilience with its
  prediction band,
* a coverage-vs-nominal diagnostic used by the conformal CLI.

All plots draw with matplotlib (headless ``Agg``) and never import geopandas
or SHAP, so importing this module never touches the E01-E04/E06-E10 optional
dependency paths.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 110


def _as_float_arrays(*arrays):
    return [np.asarray(a, dtype=float) for a in arrays]


def plot_resilience_with_ci(
    summary_df, out_path, index_col="Resilience_Index",
    low_col="Resilience_Lo", high_col="Resilience_Hi",
) -> None:
    """Bar chart of resilience index with its 90% prediction band.

    ``summary_df`` must carry point resilience (``index_col``) plus optional
    ``low_col``/``high_col`` interval bounds. Rows without bounds are drawn
    without an error bar (backward compatible with E01-E04 summaries).
    """
    if low_col in summary_df.columns and high_col in summary_df.columns:
        lo = summary_df[low_col].to_numpy(dtype=float)
        hi = summary_df[high_col].to_numpy(dtype=float)
        center = summary_df[index_col].to_numpy(dtype=float)
        lo = np.clip(lo, center - 2.0, center)
        hi = np.clip(hi, center, center + 2.0)
        yerr = np.vstack([center - lo, hi - center])
        has_err = True
    else:
        center = summary_df[index_col].to_numpy(dtype=float)
        yerr = None
        has_err = False

    order = np.argsort(-center)
    xs = np.arange(len(center))
    fig, ax = plt.subplots(figsize=(max(8, len(center) * 0.25), 5))
    ax.bar(xs, center[order], color="#6a5acd", yerr=yerr[:, order] if has_err else None,
           capsize=2, error_kw={"elinewidth": 0.8})
    ax.axhline(0.7, color="black", linewidth=0.6, linestyle="--", label="Vulnerable threshold")
    ax.set_xlabel("District-Year-Crop rows (sorted)")
    ax.set_ylabel("Resilience Index")
    ax.set_title("Resilience with 90% CI ({0})".format(
        "intervals shown" if has_err else "point values only"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_interval_by_crop(interval_df, out_path, crop_col="Crop",
                          width_col="Interval_Width") -> None:
    """Bar chart of mean prediction-interval width by crop."""
    if crop_col not in interval_df.columns or width_col not in interval_df.columns:
        # no crop/width info -> emit a single-label fallback chart
        means = [("all", interval_df[width_col].mean())] if width_col in interval_df else []
        labels = [m[0] for m in means]
        values = [m[1] for m in means]
    else:
        grouped = interval_df.groupby(crop_col)[width_col].mean().sort_values()
        labels = list(grouped.index)
        values = list(grouped.values)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.5 * len(labels) + 2)))
    ax.barh(labels, values, color="#4682b4")
    ax.set_xlabel("Mean interval width (kg/ha)")
    ax.set_ylabel("Crop")
    ax.set_title("Prediction interval width by crop")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_coverage_diagnostic(coverage: float, nominal: float, out_path) -> None:
    """Bar comparing empirical vs nominal test coverage."""
    labels = ["Empirical coverage", "Nominal (1-alpha)"]
    values = [coverage, nominal]
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, values, color=["#2e8b57", "#c9a227"])
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=9)
    ax.set_ylim(0, max(1.0, max(values) * 1.15))
    ax.set_ylabel("Coverage")
    ax.set_title("Empirical vs nominal test coverage (90% CI)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_interval_hist(low, high, out_path) -> None:
    """Histogram of prediction-interval widths."""
    widths = np.asarray(high, dtype=float) - np.asarray(low, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(widths, bins=40, color="#4682b4", edgecolor="white")
    ax.axvline(np.mean(widths), color="#c0392b", linestyle="--",
               label=f"mean = {np.mean(widths):.0f}")
    ax.set_xlabel("Interval width (kg/ha)")
    ax.set_ylabel("Count")
    ax.set_title("Prediction interval width distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_conformal_report(
    y_true,
    center,
    low,
    high,
    X_explain=None,
    *,
    out_dir,
    width: float,
    coverage: float,
    q_hat: float,
    nominal_coverage: float = 0.90,
) -> list:
    """Write CSV + plots for a conformal run and return the created paths."""
    from pathlib import Path

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    center, low, high, y_true = _as_float_arrays(center, low, high, y_true)
    written = []

    # --- tidy CSV (per-row predictions + intervals + coverage flag) ---
    import pandas as pd

    covered = (y_true >= low) & (y_true <= high)
    rows = {
        "prediction": center,
        "ci_low": low,
        "ci_high": high,
        "covered": covered,
    }
    if X_explain is not None:
        df = pd.DataFrame(X_explain)
        for col in ("Crop", "State Name", "Dist Name"):
            if col in df.columns:
                rows[col] = df[col].to_numpy()
    report_df = pd.DataFrame(rows)
    csv_path = out_dir / "conformal_report.csv"
    report_df.to_csv(csv_path, index=False)
    written.append(csv_path)

    # --- interval width histogram ---
    hist_path = out_dir / "interval_width_histogram.png"
    plot_interval_hist(low, high, hist_path)
    written.append(hist_path)

    # --- coverage vs nominal ---
    cov_path = out_dir / "coverage_vs_nominal.png"
    plot_coverage_diagnostic(coverage, nominal_coverage, cov_path)
    written.append(cov_path)

    # --- interval width by crop (if a Crop column is present) ---
    if "Crop" in report_df.columns:
        by_crop = report_df.copy()
        by_crop["Interval_Width"] = by_crop["ci_high"] - by_crop["ci_low"]
        crop_path = out_dir / "interval_width_by_crop.png"
        plot_interval_by_crop(by_crop, crop_path)
        written.append(crop_path)

    # --- a small consistency_report.txt ---
    meta_path = out_dir / "conformal_meta.txt"
    meta_path.write_text(
        f"q_hat = {q_hat:.2f}\n"
        f"empirical_coverage = {coverage:.4f}\n"
        f"nominal_coverage = {nominal_coverage:.2f}\n"
        f"mean_interval_width = {width:.2f}\n",
        encoding="utf-8",
    )
    written.append(meta_path)

    return written
