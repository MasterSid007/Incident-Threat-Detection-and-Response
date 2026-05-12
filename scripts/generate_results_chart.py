"""Generate the README results comparison chart from committed metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "docs" / "evaluation_metrics.json"
OUTPUT_PATH = ROOT / "screenshots" / "comparison.png"


def percent(value: float) -> float:
    return value * 100


def main() -> None:
    with METRICS_PATH.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    approaches = metrics["approaches"]
    labels = [item["approach"] for item in approaches]
    good_metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    fpr_values = [percent(item["fpr"]) for item in approaches]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#d7dde8",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "font.family": "DejaVu Sans",
            "font.size": 10,
        }
    )

    fig, ax = plt.subplots(figsize=(12.5, 7.0))

    metrics_to_plot = metric_labels + ["False Positive Rate"]
    y = np.arange(len(metrics_to_plot))
    height = 0.22
    colors = {
        "Rules-Only": "#64748b",
        "ML Behavioral": "#16a34a",
        "Combined": "#0ea5e9",
    }

    for idx, approach in enumerate(approaches):
        values = [percent(approach[key]) for key in good_metrics] + [percent(approach["fpr"])]
        offset = (idx - 1) * height
        bars = ax.barh(
            y + offset,
            values,
            height,
            label=labels[idx],
            color=colors.get(labels[idx], "#334155"),
            edgecolor="none",
        )
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=4, color="#0f172a")

    ax.set_title(
        "Behavioral ML cuts false positives without sacrificing recall",
        loc="left",
        fontsize=17,
        fontweight="bold",
        color="#0f172a",
        pad=18,
    )
    ax.text(
        0,
        1.02,
        "Comparison on 200,000 RBA authentication events with a temporal train/test split",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#64748b",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(metrics_to_plot)
    ax.invert_yaxis()
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score (%)")
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    ax.axvline(3.5, color="#16a34a", linewidth=1, linestyle="--", alpha=0.35)
    ax.annotate(
        "ML FPR: 3.5%",
        xy=(3.5, metrics_to_plot.index("False Positive Rate") - height),
        xytext=(13, metrics_to_plot.index("False Positive Rate") - 0.48),
        arrowprops=dict(arrowstyle="-", color="#16a34a", lw=1.2),
        color="#166534",
        fontsize=9,
        ha="left",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
