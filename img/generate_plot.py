import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

scenarios = {
    "With Reranker": {
        "strategies": ["Hybrid", "Semantic", "BM25", "Fuzzy"],
        "scores": [49, 49, 40, 45],
        "latencies": [793, 513, 491, 501],
    },
    "Without Reranker": {
        "strategies": ["Hybrid", "Semantic", "BM25", "Fuzzy"],
        "scores": [38, 44, 0, 23],
        "latencies": [47, 26, 17, 16],
    },
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for idx, (title, data) in enumerate(scenarios.items()):
    ax = ax1 if idx == 0 else ax2

    x = np.arange(len(data["strategies"]))
    width = 0.25

    color_acc = "#3b82f6"
    color_lat = "#f97316"

    ax_twin = ax.twinx()

    bars1 = ax.bar(
        x - width / 2,
        data["scores"],
        width,
        label="Score",
        color=color_acc,
        edgecolor="white",
        linewidth=1,
    )
    bars2 = ax_twin.bar(
        x + width / 2,
        data["latencies"],
        width,
        label="Latency",
        color=color_lat,
        edgecolor="white",
        linewidth=1,
    )

    ax.set_xlabel("Strategy", fontsize=13, fontweight="600")
    ax.set_ylabel("Score (out of 50)", fontsize=13, fontweight="600", color=color_acc)
    ax_twin.set_ylabel("Latency (ms)", fontsize=13, fontweight="600", color=color_lat)

    ax.tick_params(axis="y", labelcolor=color_acc, labelsize=11)
    ax_twin.tick_params(axis="y", labelcolor=color_lat, labelsize=11)
    ax.tick_params(axis="x", labelsize=12)

    ax.set_ylim(0, 55)
    max_lat = max(data["latencies"])
    ax_twin.set_ylim(0, max_lat * 1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(data["strategies"], fontsize=12)

    for bar, score in zip(bars1, data["scores"]):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 1,
            f"{score}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="600",
        )

    for bar, latency in zip(bars2, data["latencies"]):
        height = bar.get_height()
        offset = max_lat * 0.03
        ax_twin.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            f"{latency}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="600",
        )

    ax.set_title(title, fontsize=14, fontweight="700", pad=12)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

    if idx == 1:
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(
            lines1 + lines2,
            labels1 + labels2,
            loc="upper right",
            fontsize=11,
            framealpha=0.9,
        )

plt.tight_layout()
plt.savefig("strategy_comparison.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Plot saved to strategy_comparison.png")
