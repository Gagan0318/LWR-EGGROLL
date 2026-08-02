"""Plot the σ×rank sweep from experiments/sigma_rank_sweep.py.

Produces three figures for the Monday meeting:
  1. Accuracy heatmap  (σ × rank)
  2. Accuracy vs rank, one line per σ — shows the ordering flip
  3. Fitness variance vs rank at final gen — mechanism figure
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/sigma_rank_sweep/summary.json")   # <-- adjust
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

with RESULTS.open() as f:
    data = json.load(f)

# ---- adjust these three lines to match your actual JSON schema ----
sigmas = sorted(set(r["sigma"] for r in data["runs"]))
ranks = sorted(set(r["rank"] for r in data["runs"]))
runs = data["runs"]  # each has: sigma, rank, seed, best_test_acc, final_variance
# -------------------------------------------------------------------

def cell(sigma, rank, key):
    """Mean over seeds for one (sigma, rank) cell."""
    vals = [r[key] for r in runs if r["sigma"] == sigma and r["rank"] == rank]
    return np.mean(vals) if vals else np.nan

acc = np.array([[cell(s, r, "best_test_acc") for r in ranks] for s in sigmas])
var = np.array([[cell(s, r, "final_variance") for r in ranks] for s in sigmas])

# ---- Fig 1: accuracy heatmap ----
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(acc, aspect="auto", cmap="viridis", origin="lower")
ax.set_xticks(range(len(ranks)), ranks)
ax.set_yticks(range(len(sigmas)), sigmas)
ax.set_xlabel("rank")
ax.set_ylabel("σ")
ax.set_title("Best test accuracy (mean over 5 seeds)")
for i in range(len(sigmas)):
    for j in range(len(ranks)):
        ax.text(j, i, f"{acc[i, j]:.3f}", ha="center", va="center",
                color="white" if acc[i, j] < acc.mean() else "black",
                fontsize=8)
fig.colorbar(im, ax=ax, label="test acc")
fig.tight_layout()
fig.savefig(FIGDIR / "sigma_rank_accuracy_heatmap.png", dpi=150)

# ---- Fig 2: accuracy vs rank, one line per σ (shows the flip) ----
fig, ax = plt.subplots(figsize=(6, 4))
for i, s in enumerate(sigmas):
    ax.plot(ranks, acc[i, :], marker="o", label=f"σ={s}")
ax.set_xscale("log", base=2)
ax.set_xticks(ranks, ranks)
ax.set_xlabel("rank")
ax.set_ylabel("best test accuracy")
ax.set_title("σ×rank interaction — ordering flip")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "sigma_rank_accuracy_lines.png", dpi=150)

# ---- Fig 3: variance vs rank (mechanism figure) ----
fig, ax = plt.subplots(figsize=(6, 4))
for i, s in enumerate(sigmas):
    ax.plot(ranks, var[i, :], marker="s", label=f"σ={s}")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ranks, ranks)
ax.set_xlabel("rank")
ax.set_ylabel("population fitness variance (final gen)")
ax.set_title("Variance mechanism: variance ∝ 1/rank")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIGDIR / "sigma_rank_variance.png", dpi=150)

print("Wrote three figures to figures/")
for s in sigmas:
    best_rank = ranks[int(np.argmax([cell(s, r, "best_test_acc") for r in ranks]))]
    print(f"  σ={s}: best rank = {best_rank}")
