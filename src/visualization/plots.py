"""Plot generation for the crop-resilience workflow."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["figure.dpi"] = 110


def plot_yield_vs_rainfall(df, out_path) -> None:
    """Scatter of yield vs rainfall coloured by resilience class."""
    fig, ax = plt.subplots()
    for cls, color in {
        "Highly Resilient": "#2e8b57",
        "Moderately Resilient": "#c9a227",
        "Vulnerable": "#b22222",
    }.items():
        sub = df[df["Resilience_Class"] == cls]
        if len(sub):
            ax.scatter(sub["Rainfall"], sub["Actual_Yield"], s=8, label=cls, color=color)
    ax.set_xlabel("Rainfall (mm)")
    ax.set_ylabel("Actual Yield (kg/ha)")
    ax.set_title("Yield vs Rainfall by Resilience Class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_feature_importance(model, features, out_path, top: int = 10) -> None:
    """Bar chart of the most important yield predictors."""
    import pandas as pd

    importance = pd.Series(model.feature_importances_, index=features).sort_values(
        ascending=False
    )
    fig, ax = plt.subplots()
    importance.head(top).plot(kind="bar", ax=ax, color="#4682b4")
    ax.set_title("Top Factors Affecting Crop Yield")
    ax.set_ylabel("Feature importance")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_resilience_distribution(df, out_path) -> None:
    """Bar chart of resilience class counts."""
    fig, ax = plt.subplots()
    df["Resilience_Class"].value_counts().plot(kind="bar", ax=ax, color="#6a5acd")
    ax.set_xlabel("Resilience Class")
    ax.set_ylabel("Number of Crop Records")
    ax.set_title("Distribution of Crop Resilience Classes")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# E06 climate indicator visualizations
# ---------------------------------------------------------------------------

def plot_spi_timeseries(indicators_df, out_path) -> None:
    """Line plot of SPI-3, SPI-6 and SPI-12 over time."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, color, label in [
        ("SPI_3", "#e74c3c", "SPI-3"),
        ("SPI_6", "#3498db", "SPI-6"),
        ("SPI_12", "#2ecc71", "SPI-12"),
    ]:
        valid = indicators_df[["Year", col]].dropna()
        ax.plot(valid["Year"], valid[col], label=label, color=color, linewidth=1.2)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.axhline(-1, color="gray", linewidth=0.5, linestyle=":")
    ax.axhline(1, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("Year")
    ax.set_ylabel("SPI")
    ax.set_title("Standardized Precipitation Index (single-point NASA POWER record)")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_anomaly_heatmap(indicators_df, out_path) -> None:
    """Heatmap of yearly rainfall and temperature anomalies."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, col, label, cmap in [
        (axes[0], "Rainfall_Anomaly", "Rainfall anomaly (σ)", "BrBG"),
        (axes[1], "Temp_Anomaly", "Temperature anomaly (σ)", "RdBu_r"),
    ]:
        valid = indicators_df[["Year", col]].dropna()
        data = valid[col].values.reshape(1, -1)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=-3, vmax=3)
        ax.set_ylabel(label)
        ax.set_yticks([])
        years = valid["Year"].values
        step = max(1, len(years) // 10)
        ax.set_xticks(range(0, len(years), step))
        ax.set_xticklabels(years[::step], rotation=45, fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.6, label="z-score")
    fig.suptitle("Climate Anomalies (single-point NASA POWER record)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_climate_trend_table(indicators_df, out_path) -> None:
    """Tabular visualization of Mann-Kendall trend statistics."""
    import pandas as pd

    variables = [
        ("Rainfall", "Rainfall"),
        ("AvgTemp", "Avg. Temperature"),
        ("MaxTemp", "Max. Temperature"),
        ("MinTemp", "Min. Temperature"),
    ]
    rows = []
    for col, label in variables:
        direction = indicators_df[f"{col}_MK_direction"].iloc[-1]
        z = indicators_df[f"{col}_MK_Z"].iloc[-1]
        p = indicators_df[f"{col}_MK_p"].iloc[-1]
        slope = indicators_df[f"{col}_SenSlope"].iloc[-1]
        rows.append({
            "Variable": label,
            "Sen Slope": f"{slope:.4f}",
            "MK Z": f"{z:.3f}",
            "MK p-value": f"{p:.4f}" if np.isfinite(p) else "N/A",
            "Trend": direction,
        })
    table_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for i, row in enumerate(table_df.itertuples(index=False)):
        if row.Trend == "increasing":
            table[i + 1, 4].set_facecolor("#d4edda")
        elif row.Trend == "decreasing":
            table[i + 1, 4].set_facecolor("#f8d7da")
    ax.set_title(
        "Mann-Kendall Trend Analysis (single-point NASA POWER record, 1996–2020)",
        fontsize=10, pad=20,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)