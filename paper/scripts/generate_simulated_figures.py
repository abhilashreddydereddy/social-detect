from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT = Path(__file__).resolve().parent.parent / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

methods = ["Visual only", "Visual + audio", "Visual + metadata", "Full multimodal"]
metrics = {
    "Accuracy": [86.40, 89.80, 88.70, 92.30],
    "Precision": [85.90, 89.20, 88.10, 91.80],
    "Recall": [87.10, 90.60, 89.40, 92.90],
    "F1-score": [86.50, 89.90, 88.70, 92.30],
}

fig, ax = plt.subplots(figsize=(8.2, 4.8))
x = np.arange(len(methods))
width = 0.19
colors = ["#2f6690", "#3f8f8a", "#d99a2b", "#b94e48"]
for index, (metric, values) in enumerate(metrics.items()):
    bars = ax.bar(x + (index - 1.5) * width, values, width, label=metric, color=colors[index])
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
ax.set_ylim(75, 100)
ax.set_ylabel("Score (%)")
ax.set_xticks(x, methods)
ax.set_title("Illustrative MFAD-Net Ablation Comparison")
ax.grid(axis="y", alpha=0.25)
ax.legend(ncol=4, loc="upper left", fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUTPUT / "simulated_ablation.png", dpi=220, bbox_inches="tight")
plt.close(fig)

roc_methods = [
    "CIFake image branch",
    "MFAD-Net FF++ validation",
    "Visual only",
    "Visual + audio",
    "Visual + metadata",
    "Full multimodal",
    "EfficientNet-B0",
    "Filtered image",
    "Cross-dataset",
]
roc_values = [99.67, 99.77, 91.20, 94.70, 93.50, 96.10, 88.40, 90.30, 79.60]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
bar_colors = ["#2f6690", "#b94e48"] + ["#3f8f8a"] * 4 + ["#777777", "#d99a2b", "#8a5a9e"]
bars = ax.barh(roc_methods[::-1], roc_values[::-1], color=bar_colors[::-1])
ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=7)
ax.set_xlim(70, 102)
ax.set_xlabel("ROC-AUC (%)")
ax.set_title("ROC-AUC Comparison")
ax.grid(axis="x", alpha=0.25)
fig.text(0.5, 0.01, "Simulated comparison values are shown for layout only.", ha="center", fontsize=8, color="#8b2f2f")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUTPUT / "simulated_roc_auc.png", dpi=220, bbox_inches="tight")
plt.close(fig)
