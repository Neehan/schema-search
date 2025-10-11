import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

strategies = ["Semantic", "BM25", "Fuzzy", "Hybrid"]

data_without_reranker = {
    "scores": [44, 41, 23, 42],
    "latencies": [216, 16, 16, 104],
    "memory": [466.27, 403.53, 403.56, 480.00],
}

data_with_reranker = {
    "scores": [49, 44, 45, 44],
    "latencies": [801, 760, 579, 652],
    "memory": [549.11, 560.14, 560.77, 591.86],
}

color_score = "#3b82f6"
color_latency = "#f97316"
color_without = "#3b82f6"
color_with = "#f97316"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5))

x = np.arange(len(strategies))
width = 0.18

# Plot 1: Without Reranker (Score + Latency)
ax1_twin = ax1.twinx()

bars1_1 = ax1.bar(
    x - width / 2,
    data_without_reranker["scores"],
    width,
    label="Score",
    color=color_score,
    edgecolor="white",
    linewidth=1,
)

bars1_2 = ax1_twin.bar(
    x + width / 2,
    data_without_reranker["latencies"],
    width,
    label="Latency",
    color=color_latency,
    edgecolor="white",
    linewidth=1,
)

ax1.set_xlabel("Strategy", fontsize=13, fontweight="600")
ax1.set_ylabel("Score (out of 50)", fontsize=13, fontweight="600", color=color_score)
ax1_twin.set_ylabel("Latency (ms)", fontsize=13, fontweight="600", color=color_latency)
ax1.set_title("Without Reranker", fontsize=14, fontweight="700", pad=12)

ax1.tick_params(axis="y", labelcolor=color_score, labelsize=11)
ax1_twin.tick_params(axis="y", labelcolor=color_latency, labelsize=11)
ax1.tick_params(axis="x", labelsize=12)

ax1.set_ylim(0, 55)
ax1_twin.set_ylim(0, max(data_without_reranker["latencies"]) * 1.2)
ax1.set_xticks(x)
ax1.set_xticklabels(strategies, fontsize=12)
ax1.set_xlim(-0.5, len(strategies) - 0.5)
ax1.grid(axis="y", alpha=0.3, linewidth=0.5)

for bar, score in zip(bars1_1, data_without_reranker["scores"]):
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 1,
        f"{score}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

for bar, latency in zip(bars1_2, data_without_reranker["latencies"]):
    ax1_twin.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + max(data_without_reranker["latencies"]) * 0.03,
        f"{latency}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    fontsize=11,
    framealpha=0.9,
)

# Plot 2: With Reranker (Score + Latency)
ax2_twin = ax2.twinx()

bars2_1 = ax2.bar(
    x - width / 2,
    data_with_reranker["scores"],
    width,
    label="Score",
    color=color_score,
    edgecolor="white",
    linewidth=1,
)

bars2_2 = ax2_twin.bar(
    x + width / 2,
    data_with_reranker["latencies"],
    width,
    label="Latency",
    color=color_latency,
    edgecolor="white",
    linewidth=1,
)

ax2.set_xlabel("Strategy", fontsize=13, fontweight="600")
ax2.set_ylabel("Score (out of 50)", fontsize=13, fontweight="600", color=color_score)
ax2_twin.set_ylabel("Latency (ms)", fontsize=13, fontweight="600", color=color_latency)
ax2.set_title("With Reranker", fontsize=14, fontweight="700", pad=12)

ax2.tick_params(axis="y", labelcolor=color_score, labelsize=11)
ax2_twin.tick_params(axis="y", labelcolor=color_latency, labelsize=11)
ax2.tick_params(axis="x", labelsize=12)

ax2.set_ylim(0, 55)
ax2_twin.set_ylim(0, max(data_with_reranker["latencies"]) * 1.2)
ax2.set_xticks(x)
ax2.set_xticklabels(strategies, fontsize=12)
ax2.set_xlim(-0.5, len(strategies) - 0.5)
ax2.grid(axis="y", alpha=0.3, linewidth=0.5)

for bar, score in zip(bars2_1, data_with_reranker["scores"]):
    ax2.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 1,
        f"{score}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

for bar, latency in zip(bars2_2, data_with_reranker["latencies"]):
    ax2_twin.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + max(data_with_reranker["latencies"]) * 0.03,
        f"{latency}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    fontsize=11,
    framealpha=0.9,
)

# Plot 3: Memory Footprint
bars3_1 = ax3.bar(
    x - width / 2,
    data_without_reranker["memory"],
    width,
    label="Without Reranker",
    color=color_without,
    edgecolor="white",
    linewidth=1,
)

bars3_2 = ax3.bar(
    x + width / 2,
    data_with_reranker["memory"],
    width,
    label="With Reranker",
    color=color_with,
    edgecolor="white",
    linewidth=1,
)

ax3.set_xlabel("Strategy", fontsize=13, fontweight="600")
ax3.set_ylabel("Peak Memory (MB)", fontsize=13, fontweight="600")
ax3.set_title("Memory Footprint", fontsize=14, fontweight="700", pad=12)
ax3.set_xticks(x)
ax3.set_xticklabels(strategies, fontsize=12)
ax3.set_xlim(-0.5, len(strategies) - 0.5)
ax3.set_ylim(0, max(data_with_reranker["memory"]) * 1.3)
ax3.grid(axis="y", alpha=0.3, linewidth=0.5)
ax3.tick_params(axis="both", labelsize=11)
ax3.legend(fontsize=11, loc="upper right", framealpha=0.9)

for bar, mem in zip(bars3_1, data_without_reranker["memory"]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + max(data_with_reranker["memory"]) * 0.02,
        f"{mem:.0f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

for bar, mem in zip(bars3_2, data_with_reranker["memory"]):
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + max(data_with_reranker["memory"]) * 0.02,
        f"{mem:.0f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="600",
    )

plt.tight_layout()
plt.savefig(
    "img/strategy_comparison.png", dpi=300, bbox_inches="tight", facecolor="white"
)
print("✅ Plot saved to img/strategy_comparison.png")
