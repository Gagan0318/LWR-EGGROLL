"""Phase 2 causal ablation — degradation per layer per dataset from pilot_3phase."""
import json, os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

PILOT_DIR = "results/pilot_3phase"
DATASETS = {
    "mnist": "MNIST",
    "fashion_mnist": "Fashion-MNIST",
    "kmnist": "KMNIST",
    "emnist_digits": "EMNIST-Digits",
}

# ── parse pilot_allocation.json per dataset ──
results = {}
for ds_dir, ds_label in DATASETS.items():
    path = os.path.join(PILOT_DIR, ds_dir, "pilot_allocation.json")
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        continue
    with open(path) as f:
        r = json.load(f)
    phase2 = r.get("phase2", [])
    if not phase2:
        print(f"  [SKIP] {ds_dir} has no phase2 data")
        continue
    degradation = {}
    for entry in phase2:
        layer = entry["layer"]
        degradation[layer] = entry["mean"] * 100  # to percentage points
    results[ds_label] = degradation
    print(f"  {ds_label}: {degradation}")

if not results:
    print("[!] No phase2 data found")
    exit(1)

# ── plot ──
datasets = list(results.keys())
layers = ["input", "hidden", "output"]
colors = {"input": "#e24b4a", "hidden": "#eda100", "output": "#1baf7a"}

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(datasets))
w = 0.22

for i, layer in enumerate(layers):
    vals = [results[d].get(layer, 0) for d in datasets]
    offset = (i - 1) * w
    ax.bar(x + offset, vals, w, label=f"{layer} layer", color=colors[layer],
           edgecolor='white', linewidth=0.5)

ax.set_xlabel("Dataset")
ax.set_ylabel("Accuracy degradation (pp)")
ax.set_title("Phase 2 causal ablation — degradation when layer dropped from r=4 to r=1")
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10)
ax.legend(fontsize=9, framealpha=0.9)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig("figures/phase2_degradation.png", dpi=200, bbox_inches='tight')
print("Saved → figures/phase2_degradation.png")
plt.close()
