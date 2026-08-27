"""01 — LWR vs Vanilla vs Reversed: headline comparison across datasets."""
import json, glob, os, numpy as np, matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/dissertation/eggroll-diss/results")
FIG  = os.path.expanduser("~/dissertation/eggroll-diss/figures")
os.makedirs(FIG, exist_ok=True)

def load_accs(pattern):
    accs = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        accs.append(d["best_test_acc"])
    return accs

# --- MNIST (from validation/ and lwr_allocation_controls/) ---
mnist = {
    "LWR (8,2,0)": load_accs(f"{ROOT}/validation/lwr_derived_r0/*.json"),
    "Vanilla r=4":  load_accs(f"{ROOT}/validation/uniform_r4/*.json") or
                    load_accs(f"{ROOT}/lwr_allocation_controls/lwr_4_4_4/*.json"),
    "Reversed (0,2,8)": load_accs(f"{ROOT}/lwr_allocation_controls/lwr_0_2_8/*.json"),
}

# --- Cross-dataset ---
datasets = {"Fashion-MNIST": "fashion", "KMNIST": "kmnist", "EMNIST-Digits": "emnist"}
cross = {}
for label, folder in datasets.items():
    base = f"{ROOT}/cross_dataset/{folder}"
    lwr_files = sorted(glob.glob(f"{base}/lwr/**/*.json", recursive=True))
    van_files = sorted(glob.glob(f"{base}/vanilla/**/*.json", recursive=True))

    # Try to separate aligned vs reversed from lwr files
    aligned, reversed_ = [], []
    for f in lwr_files:
        d = json.load(open(f))
        name = d.get("method", "") or d.get("experiment", "") or os.path.basename(os.path.dirname(f))
        if "0_2_8" in name or "reversed" in name.lower():
            reversed_.append(d["best_test_acc"])
        else:
            aligned.append(d["best_test_acc"])

    vanilla = [json.load(open(f))["best_test_acc"] for f in van_files]

    cross[label] = {
        "LWR aligned": aligned if aligned else [np.nan],
        "Vanilla": vanilla if vanilla else [np.nan],
        "Reversed": reversed_ if reversed_ else [np.nan],
    }

# --- Plot ---
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
colors = {"LWR (8,2,0)": "#8ecf9e", "LWR aligned": "#8ecf9e",
          "Vanilla r=4": "#8eb8e0", "Vanilla": "#8eb8e0",
          "Reversed (0,2,8)": "#e8a0a0", "Reversed": "#e8a0a0"}

def plot_group(ax, data, title):
    x = np.arange(len(data))
    w = 0.25
    for i, (label, accs) in enumerate(data.items()):
        mean = np.mean(accs) * 100
        std  = np.std(accs) * 100
        color = list(colors.values())[i % 3]
        bar = ax.bar(x[0] + i * w, mean, w, yerr=std, capsize=4,
                     color=color, edgecolor="white", label=label)
        ax.text(x[0] + i * w, mean + std + 0.3, f"{mean:.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_ylabel("Test Accuracy (%)")
    ax.legend(fontsize=7, loc="lower left")

plot_group(axes[0], mnist, "MNIST")
for i, (label, data) in enumerate(cross.items()):
    plot_group(axes[i + 1], data, label)

fig.suptitle("LWR-EGGROLL vs Vanilla vs Reversed Allocation", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG}/01_lwr_headline_comparison.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG}/01_lwr_headline_comparison.pdf", bbox_inches="tight")
print(f"Saved to {FIG}/01_lwr_headline_comparison.png")
plt.show()
