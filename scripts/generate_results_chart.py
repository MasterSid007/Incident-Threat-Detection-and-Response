"""Generate the README results comparison chart from committed metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METRICS_PATH = ROOT / "docs" / "evaluation_metrics.json"
OUTPUT_PATH = ROOT / "screenshots" / "results_comparison.png"


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

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_main, ax_fpr) = plt.subplots(
        1,
        2,
        figsize=(13.5, 5.8),
        gridspec_kw={"width_ratios": [3.2, 1]},
    )

    x = np.arange(len(metric_labels))
    width = 0.23
    colors = ["#2563eb", "#16a34a", "#7c3aed"]

    for idx, approach in enumerate(approaches):
        values = [percent(approach[key]) for key in good_metrics]
        ax_main.bar(
            x + (idx - 1) * width,
            values,
            width,
            label=labels[idx],
            color=colors[idx],
            edgecolor="none",
        )

    ax_main.set_title("Detection Performance: Rules vs ML vs Combined", weight="bold", pad=14)
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(metric_labels)
    ax_main.set_ylim(0, 105)
    ax_main.set_ylabel("Score (%)")
    ax_main.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=3, frameon=True)
    ax_main.grid(axis="y", alpha=0.25)
    ax_main.grid(axis="x", visible=False)

    for container in ax_main.containers:
        ax_main.bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)

    ax_fpr.bar(labels, fpr_values, color=colors, edgecolor="none")
    ax_fpr.set_title("False Positive Rate", weight="bold", pad=14)
    ax_fpr.set_ylim(0, 60)
    ax_fpr.set_ylabel("FPR (%)")
    ax_fpr.tick_params(axis="x", rotation=30)
    ax_fpr.grid(axis="y", alpha=0.25)
    ax_fpr.grid(axis="x", visible=False)

    for container in ax_fpr.containers:
        ax_fpr.bar_label(container, fmt="%.1f%%", fontsize=8, padding=2)

    fig.suptitle(
        "ML Behavioral Detection reached 95.4% accuracy and reduced FPR from 49.9% to 3.5%",
        fontsize=13,
        weight="bold",
        y=1.03,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUTPUT_PATH, dpi=180, bbox_inches="tight")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
