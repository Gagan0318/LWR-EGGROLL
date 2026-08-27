"""LWR aligned vs vanilla r=4 vs reversed — grouped bar across 4 datasets.
Uses results/cross_dataset/{dataset}/lwr/ and /vanilla/ directories."""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs("figures", exist_ok=True)

DATASETS = [
    ("fashion", "Fashion-MNIST"),
    ("kmnist", "KMNIST"),
    ("emnist", "EMNIST-Digits"),
]

GROUPS = {
    "LWR aligned (8,4,0)": ["lwr_8_4_0", "lwr_8_2_0"],
    "Vanilla r=4": ["eggroll_r4", "vanilla_r4"],
    "Reversed (0,2,8)": ["lwr_0_2_8"],
}

all_data = defaultdict(lambda: defaultdict(list))

# ── also grab MNIST from root-level results or lwr_allocation_controls ──
# root results: results/lwr_allocation_controls/{method}/seed{n}.json
for method_dir in glob.glob("results/lwr_allocation_controls/*"):
    method = os.path.basename(method_dir)
    for f in glob.glob(os.path.join(method_dir, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        acc = r.get("best_test_acc", None)
        if acc is None: continue
        for grp, patterns in GROUPS.items():
            if any(p == method for p in patterns):
                all_data["MNIST"][grp].append(float(acc))
                break

# ── cross_dataset directories ──
for ds_dir, ds_label in DATASETS:
    base = f"results/cross_dataset/{ds_dir}"
    # LWR methods
    for method_dir in glob.glob(os.path.join(base, "lwr", "*")):
        method = os.path.basename(method_dir)
        for f in glob.glob(os.path.join(method_dir, "*.json")):
            with open(f) as fh:
                r = json.load(fh)
            acc = r.get("best_test_acc", None)
            if acc is None: continue
            for grp, patterns in GROUPS.items():
                if any(p == method for p in patterns):
                    all_data[ds_label][grp].append(float(acc))
                    break
    # Vanilla methods
    for method_dir in glob.glob(os.path.join(base, "vanilla", "*")):
        method = os.path.basename(method_dir)
        for f in glob.glob(os.path.join(method_dir, "*.json")):
            with open(f) as fh:
                r = json.load(fh)
            acc = r.get("best_test_acc", None)
            if acc is None: continue
            for grp, patterns in GROUPS.items():
                if any(p == method for p in patterns):
                    all_data[ds_label][grp].append(float(acc))
                    break

found = [d for d in ["MNIST"] + [l for _, l in DATASETS] if all_data[d]]
if not found:
    print("[!] No data found")
    exit(1)

for d in found:
    print(f"  {d}: " + ", ".join(f"{g}={np.mean(v)*100:.1f}%" for g, v in all_data[d].items()))

# ── plot ──
groups = list(GROUPS.keys())
colors = ['#2a78d6', '#898781', '#e24b4a']
hatches = ['', '', '///']

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(found))
n = len(groups)
w = 0.22

for i, grp in enumerate(groups):
    means = [np.mean(all_data[d].get(grp, [0]))*100 for d in found]
    errs  = [np.std(all_data[d].get(grp, [0]))*100 if len(all_data[d].get(grp, [])) > 1 else 0 for d in found]
    offset = (i - 1) * w
    ax.bar(x + offset, means, w, yerr=errs, capsize=3,
           label=grp, color=colors[i], hatch=hatches[i], edgecolor='white', linewidth=0.5)

ax.set_xlabel("Dataset")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Allocation direction matters — aligned vs uniform vs reversed")
ax.set_xticks(x)
ax.set_xticklabels(found, fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig("figures/lwr_vs_vanilla_reversed.png", dpi=200, bbox_inches='tight')
print("Saved → figures/lwr_vs_vanilla_reversed.png")
plt.close()
