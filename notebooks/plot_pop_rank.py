"""06 — Population × Rank Interaction: heatmap showing rank matters at low N, not at high N."""
import json, os, numpy as np, matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/dissertation/eggroll-diss/results/pop_rank_interaction")
FIG  = os.path.expanduser("~/dissertation/eggroll-diss/figures")
os.makedirs(FIG, exist_ok=True)

# Load the combined JSON
files = [f for f in os.listdir(ROOT) if f.endswith(".json")]
if len(files) == 1:
    data = json.load(open(os.path.join(ROOT, files[0])))
else:
    # Multiple files — try to reconstruct
    data = {}
    for f in files:
        d = json.load(open(os.path.join(ROOT, f)))
        key = os.path.splitext(f)[0]
        data[key] = d

pops = [256, 512, 1024, 4096]
ranks = [1, 4, 16]

# Build accuracy matrix
acc_matrix = np.zeros((len(pops), len(ranks)))
for i, pop in enumerate(pops):
    for j, rank in enumerate(ranks):
        key = f"pop{pop}_r{rank}"
        if key in data:
            val = data[key]
            if isinstance(val, dict):
                acc_matrix[i, j] = val.get("best_test_acc", val.get("mean_acc", 0)) * 100
            elif isinstance(val, list):
                acc_matrix[i, j] = np.mean(val) * 100
            else:
                acc_matrix[i, j] = float(val) * 100

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=acc_matrix.min() - 2,
               vmax=acc_matrix.max() + 2)

# Annotate cells
for i in range(len(pops)):
    for j in range(len(ranks)):
        text_color = "white" if acc_matrix[i, j] < (acc_matrix.min() + acc_matrix.max()) / 2 else "black"
        ax.text(j, i, f"{acc_matrix[i, j]:.1f}%", ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color)

ax.set_xticks(range(len(ranks)))
ax.set_xticklabels([f"r={r}" for r in ranks], fontsize=11)
ax.set_yticks(range(len(pops)))
ax.set_yticklabels([f"N={p}" for p in pops], fontsize=11)
ax.set_xlabel("Perturbation Rank", fontsize=12)
ax.set_ylabel("Population Size", fontsize=12)
ax.set_title("Population × Rank Interaction (MNIST, n=3)\nRank matters at low N, irrelevant at high N",
             fontsize=13, fontweight="bold")

# Add aggregate rank annotations
for i, pop in enumerate(pops):
    for j, rank in enumerate(ranks):
        nr = pop * rank
        ax.text(j, i + 0.35, f"Nr={nr}", ha="center", va="center",
                fontsize=7, color="gray", style="italic")

plt.colorbar(im, ax=ax, label="Peak Test Accuracy (%)", shrink=0.8)
plt.tight_layout()
plt.savefig(f"{FIG}/06_population_rank.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG}/06_population_rank.pdf", bbox_inches="tight")
print(f"Saved to {FIG}/06_population_rank.png")
plt.show()
