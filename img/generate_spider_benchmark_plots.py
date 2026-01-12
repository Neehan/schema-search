#!/usr/bin/env python3
"""Generate Spider benchmark comparison plots from JSON results."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.style.use("seaborn-v0_8-whitegrid")


BENCHMARK_SPECS = [
    ("spider_benchmark_without_reranker.json", "spider_benchmark_without_reranker.png"),
    ("spider_benchmark_with_reranker.json", "spider_benchmark_with_reranker.png"),
]
PRECISION_KS = [1, 3, 5]
RECALL_KS = [1, 3, 5]
PERCENT_PREFIXES = ("precision_at_", "recall_at_")
PERCENT_METRICS = {"mrr"}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#f8f9fb",
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 16,
        "axes.linewidth": 1.0,
        "axes.titlesize": 16,
        "font.size": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "grid.alpha": 0.25,
    }
)


def _load_results(json_path: Path):
    with open(json_path) as f:
        return json.load(f)


def _collect_metric(stats: dict, metric_name: str) -> float:
    metric = stats.get(metric_name)
    if isinstance(metric, dict):
        value = float(metric.get("mean", 0.0))
    else:
        value = float(metric or 0.0)
    if metric_name == "latency":
        return value * 1000.0
    if metric_name in PERCENT_METRICS or metric_name.startswith(PERCENT_PREFIXES):
        return value * 100.0
    return value


def _plot_grouped_bars(
    ax, ks, strategies, strategy_stats, colors, title, metric_prefix
):
    x = np.arange(len(ks))
    bar_width = 0.6 / len(strategies)
    total_width = bar_width * len(strategies)

    for idx, strategy in enumerate(strategies):
        stats = strategy_stats[strategy]
        values = [_collect_metric(stats, f"{metric_prefix}_{k}") for k in ks]
        offsets = x - total_width / 2 + idx * bar_width + bar_width / 2
        ax.bar(
            offsets,
            values,
            width=bar_width,
            color=colors[idx],
            edgecolor="#222222",
            linewidth=0.75,
            label=strategy.upper(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_xlabel("k", fontsize=16)
    ax.set_ylabel(f"{title} (%)", fontsize=16)
    ax.tick_params(axis="both", length=0)


def _plot_dual_metric(ax, strategies, strategy_stats, colors, metric_names, ylabels):
    mrr_values = [
        _collect_metric(strategy_stats[strategy], metric_names[0])
        for strategy in strategies
    ]
    latency_values = [
        _collect_metric(strategy_stats[strategy], metric_names[1])
        for strategy in strategies
    ]

    x = np.arange(len(metric_names))  # positions for metrics
    bar_width = 0.6 / len(strategies)
    total_width = bar_width * len(strategies)

    ax_mrr = ax
    ax_latency = ax.twinx()

    for idx, strategy in enumerate(strategies):
        offsets = x - total_width / 2 + idx * bar_width + bar_width / 2

        ax_mrr.bar(
            offsets[0],
            mrr_values[idx],
            width=bar_width,
            color=colors[idx],
            edgecolor="#222222",
            linewidth=0.75,
        )

        ax_latency.bar(
            offsets[1],
            latency_values[idx],
            width=bar_width,
            color=colors[idx],
            edgecolor="#222222",
            linewidth=0.75,
        )

    ax_mrr.set_xticks(x)
    ax_mrr.set_xticklabels(ylabels)
    ax_mrr.set_ylabel(ylabels[0], fontsize=16)
    ax_latency.set_ylabel(ylabels[1], fontsize=16)
    ax_mrr.set_ylim(0, 105)
    latency_max = max(latency_values) if latency_values else 1.0
    ax_latency.set_ylim(0, latency_max * 1.2)
    ax_mrr.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
    ax_latency.grid(False)

    ax_mrr.tick_params(axis="both", length=0)
    ax_latency.tick_params(axis="y", length=0)

    return ax_mrr, ax_latency


def generate_plot(json_filename: str, output_filename: str) -> None:
    json_path = Path(__file__).with_name(json_filename)
    if not json_path.exists():
        print(f"Skipping missing benchmark JSON: {json_filename}")
        return

    data = _load_results(json_path)
    strategy_stats = data.get("strategies", {})
    if not strategy_stats:
        print(f"No strategy data found in {json_filename}.")
        return

    preferred_order = ["fuzzy", "bm25", "semantic", "hybrid"]
    strategies = [s for s in preferred_order if s in strategy_stats]
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    colors = [palette[i % len(palette)] for i in range(len(strategies))]
    legend_handles = [
        Patch(facecolor=colors[i], edgecolor="#222222", label=strategies[i].upper())
        for i in range(len(strategies))
    ]

    fig = plt.figure(figsize=(18, 5), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.2])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    _plot_grouped_bars(
        axes[0],
        PRECISION_KS,
        strategies,
        strategy_stats,
        colors,
        "Precision@k",
        "precision_at",
    )
    axes[0].set_xlabel("k")

    _plot_grouped_bars(
        axes[1],
        RECALL_KS,
        strategies,
        strategy_stats,
        colors,
        "Recall@k",
        "recall_at",
    )
    axes[1].set_xlabel("k")

    _plot_dual_metric(
        axes[2],
        strategies,
        strategy_stats,
        colors,
        ("mrr", "latency"),
        ("MRR (%)", "Latency (ms)"),
    )

    fig.subplots_adjust(top=0.82, left=0.06, right=0.94)
    fig.subplots_adjust(top=0.82, left=0.06, right=0.94)
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],  # type: ignore
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=len(strategies),
        frameon=False,
        fontsize=16,
    )

    output_path = Path(__file__).with_name(output_filename)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    for json_filename, output_filename in BENCHMARK_SPECS:
        generate_plot(json_filename, output_filename)


if __name__ == "__main__":
    main()
