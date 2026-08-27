"""03 — Architecture Variation: LWR vs Vanilla vs Reversed across network depths."""
import json, glob, os, numpy as np, matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/dissertation/eggroll-diss/results/arch_variation")
FIG  = os.path.expanduser("~/dissertation/eggroll-diss/figures")
os.makedirs(FIG, exist_ok=True)

archs = [("narrow_2h", "Narrow\n(2 hidden)"),
         ("standard_3h", "Standard\n(3 hidden)"),
         ("deep_4h", "Deep\n(4 hidden)")]
configs = [("lwr_aligned", "LWR aligned", "#8ecf9e"),
           ("vanilla_r4", "Vanilla r=4", "#8eb8e0"),
           ("lwr_reversed", "Reversed", "#e8a0a0")]

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(archs))
w = 0.25

for i, (cfg_dir, cfg_label, color) in enumerate(configs):
    means, stds = [], []
    for arch_dir, _ in archs:
        path = f"{ROOT}/{arch_dir}/{cfg_dir}"
        accs = [json.load(open(f))["best_test_acc"]
                for f in sorted(glob.glob(f"{path}/*.json"))]
        means.append(np.mean(accs) * 100 if accs else 0)
        stds.append(np.std(accs) * 100 if accs else 0)

    bars = ax.bar(x + i * w, means, w, yerr=stds, capsize=4,
                  color=color, edgecolor="white", label=cfg_label)
    for j, bar in enumerate(bars):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + stds[j] + 0.3,
                f"{means[j]:.1f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(x + w)
ax.set_xticklabels([a[1] for a in archs], fontsize=10)
ax.set_ylabel("Peak Test Accuracy (%)", fontsize=11)
ax.set_title("LWR-EGGROLL Advantage Across Network Depths (MNIST, n=3)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_ylim(68, 92)

# Add advantage annotations
for j, (arch_dir, _) in enumerate(archs):
    aligned = np.mean([json.load(open(f))["best_test_acc"]
              for f in glob.glob(f"{ROOT}/{arch_dir}/lwr_aligned/*.json")]) * 100
    vanilla = np.mean([json.load(open(f))["best_test_acc"]
              for f in glob.glob(f"{ROOT}/{arch_dir}/vanilla_r4/*.json")]) * 100
    diff = aligned - vanilla
    ax.annotate(f"+{diff:.1f}pp", xy=(j + 0.125, max(aligned, vanilla) + 2),
                fontsize=9, fontweight="bold", color="#4caf6a", ha="center")

plt.tight_layout()
plt.savefig(f"{FIG}/03_architecture_variation.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG}/03_architecture_variation.pdf", bbox_inches="tight")
print(f"Saved to {FIG}/03_architecture_variation.png")
plt.show()
