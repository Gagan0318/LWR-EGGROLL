"""02 — Sensitivity Pilot: input vs hidden vs output accuracy across architectures."""
import json, glob, os, numpy as np, matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/dissertation/eggroll-diss/results/arch_variation")
FIG  = os.path.expanduser("~/dissertation/eggroll-diss/figures")
os.makedirs(FIG, exist_ok=True)

archs = [("narrow_2h", "Narrow (2h)"), ("standard_3h", "Standard (3h)"), ("deep_4h", "Deep (4h)")]
layers = [("input_only", "Input only"), ("hidden_only", "Hidden only"), ("output_only", "Output only")]
colors = ["#8ecf9e", "#8eb8e0", "#e8a0a0"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

for ax_i, (arch_dir, arch_label) in enumerate(archs):
    means, stds, labels = [], [], []
    for layer_dir, layer_label in layers:
        path = f"{ROOT}/{arch_dir}/sensitivity/{layer_dir}"
        accs = []
        for f in sorted(glob.glob(f"{path}/*.json")):
            d = json.load(open(f))
            accs.append(d["best_test_acc"])
        means.append(np.mean(accs) * 100 if accs else 0)
        stds.append(np.std(accs) * 100 if accs else 0)
        labels.append(layer_label)

    x = np.arange(len(layers))
    bars = axes[ax_i].bar(x, means, yerr=stds, capsize=5, color=colors,
                           edgecolor="white", width=0.6)
    for j, bar in enumerate(bars):
        axes[ax_i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[j] + 0.3,
                        f"{means[j]:.1f}%", ha="center", va="bottom", fontsize=9)
    axes[ax_i].set_xticks(x)
    axes[ax_i].set_xticklabels(labels, fontsize=9)
    axes[ax_i].set_title(arch_label, fontsize=11, fontweight="bold")
    axes[ax_i].set_ylim(65, 95)

axes[0].set_ylabel("Peak Test Accuracy (%)", fontsize=10)
fig.suptitle("Sensitivity Pilot: Isolated Layer Accuracy by Architecture (MNIST, n=5)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG}/02_sensitivity_pilot.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG}/02_sensitivity_pilot.pdf", bbox_inches="tight")
print(f"Saved to {FIG}/02_sensitivity_pilot.png")
plt.show()
