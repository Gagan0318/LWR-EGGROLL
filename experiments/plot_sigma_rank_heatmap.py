"""σ × rank interaction heatmap from results/variance_rank/sig{X}_r{Y}/vsweep_*.json"""
import json, glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs("figures", exist_ok=True)

data = defaultdict(list)  # (sigma, rank) → [best_test_acc, ...]

for subdir in sorted(glob.glob("results/variance_rank/sig*_r*")):
    dirname = os.path.basename(subdir)
    m = re.match(r'sig([\d.]+)_r(\d+)', dirname)
    if not m: continue
    sigma, rank = float(m.group(1)), int(m.group(2))
    for f in glob.glob(os.path.join(subdir, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        acc = r.get("best_test_acc", None)
        if acc is not None:
            data[(sigma, rank)].append(float(acc))

if not data:
    print("[!] No data found in results/variance_rank/")
    exit(1)

sigmas = sorted(set(s for s, _ in data.keys()))
ranks = sorted(set(r for _, r in data.keys()))
print(f"σ values: {sigmas}")
print(f"Rank values: {ranks}")

matrix = np.full((len(sigmas), len(ranks)), np.nan)
for i, s in enumerate(sigmas):
    for j, r in enumerate(ranks):
        vals = data.get((s, r), [])
        if vals:
            matrix[i, j] = np.mean(vals) * 100

fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')

ax.set_xticks(range(len(ranks)))
ax.set_xticklabels([str(r) for r in ranks])
ax.set_yticks(range(len(sigmas)))
ax.set_yticklabels([str(s) for s in sigmas])
ax.set_xlabel("Perturbation rank")
ax.set_ylabel("Noise scale σ")
ax.set_title("σ × rank interaction — mean test accuracy (%)")

for i in range(len(sigmas)):
    for j in range(len(ranks)):
        if not np.isnan(matrix[i, j]):
            val = matrix[i, j]
            color = 'white' if val < np.nanmedian(matrix) else 'black'
            ax.text(j, i, f"{val:.1f}", ha='center', va='center', fontsize=8, color=color)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Accuracy (%)")
fig.tight_layout()
fig.savefig("figures/sigma_rank_heatmap.png", dpi=200, bbox_inches='tight')
print("Saved → figures/sigma_rank_heatmap.png")
plt.close()
