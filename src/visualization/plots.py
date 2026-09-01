"""Plot generation for the crop-resilience workflow."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless-safe

import matplotlib.pyplot as plt

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