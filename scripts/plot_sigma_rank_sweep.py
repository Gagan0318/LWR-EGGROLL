"""Plot the σ×rank sweep from results/variance_rank/summary.json.

Produces three figures for the Monday meeting:
  1. Accuracy heatmap  (σ × rank)
  2. Accuracy vs rank, one line per σ — shows the σ×rank interaction
  3. Fitness variance vs rank at final gen — mechanism figure
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/variance_rank/summary.json")
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

with RESULTS.open() as f:
    data = json.load(f)

sigmas = data["sigmas"]           # e.g. [0.01, 0.03, 0.05, 0.1, 0.3]
ranks = data["ranks"]             # e.g. [1, 2, 4, 8, 16]
cells = data["summary"]           # dict keyed by "sig{s}_r{r}"

def key(s, r):
    return f"sig{s}_r{r}"

acc = np.array([[cells[key(s, r)]["acc_mean"] for r in ranks] for s in sigmas])
acc_std = np.array([[cells[key(s, r)]["acc_std"] for r in ranks] for s in sigmas])
var = np.array([[cells[key(s, r)]["variance_mean"] for r in ranks] for s in sigmas])

# ---- Fig 1: accuracy heatmap ----
fig, ax = plt.subplots(figsize=(7, 4.5))
im = ax.imshow(acc, aspect="auto", cmap="viridis", origin="lower")
ax.set_xticks(range(len(ranks)), ranks)
ax.set_yticks(range(len(sigmas)), sigmas)
ax.set_xlabel("rank")
ax.set_ylabel("σ")
ax.set_title("Best test accuracy (mean over 5 seeds)")
mid = (acc.max() + acc.min()) / 2
for i in range(len(sigmas)):
    for j in range(len(ranks)):
        ax.text(j, i, f"{acc[i, j]:.3f}", ha="center", va="center",
                color="white" if acc[i, j] < mid else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="test acc")
fig.tight_layout()
fig.savefig(FIGDIR / "sigma_rank_accuracy_heatmap.png", dpi=150)

# ---- Fig 2: accuracy vs rank, one line per σ ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, s in enumerate(sigmas):
    ax.errorbar(ranks, acc[i, :], yerr=acc_std[i, :],
                marker="o", capsize=3, label=f"σ={s}")
ax.set_xscale("log", base=2)
ax.set_xticks(ranks, ranks)
ax.set_xlabel("rank")
ax.set_ylabel("best test accuracy (mean ± std, n=5)")
ax.set_title("σ×rank interaction")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "sigma_rank_accuracy_lines.png", dpi=150)

# ---- Fig 3: variance vs rank (mechanism figure) ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for i, s in enumerate(sigmas):
    ax.plot(ranks, var[i, :], marker="s", label=f"σ={s}")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(ranks, ranks)
ax.set_xlabel("rank")
ax.set_ylabel("population fitness variance (final gen, log)")
ax.set_title("Variance mechanism: variance decreases with rank at every σ")
ax.legend()
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(FIGDIR / "sigma_rank_variance.png", dpi=150)

print("Wrote three figures to figures/")
for i, s in enumerate(sigmas):
    j_best = int(np.argmax(acc[i, :]))
    print(f"  σ={s}: best rank = {ranks[j_best]} @ {acc[i, j_best]:.4f}")
