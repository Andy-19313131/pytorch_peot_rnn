#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Create evaluation figures from Task E CSV outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def save_metric_comparison(metrics_path: str, output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rows = read_csv_rows(metrics_path)
    if not rows:
        return False

    metrics = ["format_correctness", "rhyme_consistency", "distinct_1", "distinct_2", "unique_ratio"]
    strategies = [row["strategy"] for row in rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4), sharey=False)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        values = [float(row.get(metric, 0) or 0) for row in rows]
        ax.bar(strategies, values, color="#4C78A8")
        ax.set_title(metric)
        ax.set_ylim(0, max(1.0, max(values) * 1.15 if values else 1.0))
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def save_loss_curves(log_dir: str, output_path: str, ppl_output_path: str) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    curve_files = glob.glob(os.path.join(log_dir, "*curves.csv"))
    if not curve_files:
        return False

    fig_loss, ax_loss = plt.subplots(figsize=(7, 4))
    fig_ppl, ax_ppl = plt.subplots(figsize=(7, 4))
    for path in curve_files:
        rows = read_csv_rows(path)
        epochs = [int(row["epoch"]) for row in rows]
        train_loss = [float(row["train_loss"]) for row in rows]
        valid_loss = [float(row["valid_loss"]) for row in rows]
        label = os.path.basename(path).replace("_curves.csv", "")
        ax_loss.plot(epochs, train_loss, label=f"{label} train")
        ax_loss.plot(epochs, valid_loss, linestyle="--", label=f"{label} valid")
        ax_ppl.plot(epochs, [pow(2.718281828, value) for value in valid_loss], label=label)

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend(fontsize=8)
    ax_loss.set_title("Training and validation loss")
    fig_loss.tight_layout()

    ax_ppl.set_xlabel("epoch")
    ax_ppl.set_ylabel("perplexity")
    ax_ppl.legend(fontsize=8)
    ax_ppl.set_title("Validation perplexity")
    fig_ppl.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig_loss.savefig(output_path, dpi=180)
    fig_ppl.savefig(ppl_output_path, dpi=180)
    plt.close(fig_loss)
    plt.close(fig_ppl)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Visualize Task E outputs")
    parser.add_argument("--results_dir", default="results")
    args = parser.parse_args()

    metrics_path = os.path.join(args.results_dir, "metrics", "all_metrics.csv")
    figures_dir = os.path.join(args.results_dir, "figures")

    made_any = False
    if os.path.exists(metrics_path):
        made_any = save_metric_comparison(
            metrics_path,
            os.path.join(figures_dir, "metric_comparison.png"),
        ) or made_any
    made_any = save_loss_curves(
        os.path.join(args.results_dir, "logs"),
        os.path.join(figures_dir, "loss_curve.png"),
        os.path.join(figures_dir, "perplexity_curve.png"),
    ) or made_any

    if not made_any:
        print("No figures were created. Install matplotlib or provide metrics/training curve CSV files.")
    else:
        print(f"Figures written under {figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

