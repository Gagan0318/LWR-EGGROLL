"""04 — Per-Layer Rank Sweeps: saturation curves for input, hidden, output."""
import json, glob, os, numpy as np, matplotlib.pyplot as plt

ROOT = os.path.expanduser("~/dissertation/eggroll-diss/results")
FIG  = os.path.expanduser("~/dissertation/eggroll-diss/figures")
os.makedirs(FIG, exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Input rank sweep ---
input_path = f"{ROOT}/rank_study/input_sweep"
input_ranks, input_means, input_stds = [], [], []
for d in sorted(os.listdir(input_path)):
    full = os.path.join(input_path, d)
    if not os.path.isdir(full):
        continue
    accs = []
    rank_val = None
    for f in sorted(glob.glob(f"{full}/*.json")):
        data = json.load(open(f))
        accs.append(data["best_test_acc"])
        if rank_val is None:
            rank_val = data.get("input_rank", int(''.join(filter(str.isdigit, d)) or 0))
    if accs:
        input_ranks.append(rank_val)
        input_means.append(np.mean(accs) * 100)
        input_stds.append(np.std(accs) * 100)

order = np.argsort(input_ranks)
input_ranks = [input_ranks[i] for i in order]
input_means = [input_means[i] for i in order]
input_stds  = [input_stds[i] for i in order]

axes[0].errorbar(input_ranks, input_means, yerr=input_stds, marker="o", capsize=4,
                 color="#8ecf9e", linewidth=2, markersize=8)
axes[0].set_xlabel("Input Layer Rank", fontsize=10)
axes[0].set_ylabel("Peak Test Accuracy (%)", fontsize=10)
axes[0].set_title("Input Rank Sweep\n(hidden=2, output=0)", fontsize=11, fontweight="bold")
axes[0].set_xticks(input_ranks)

# --- Hidden rank sweep ---
hidden_path = f"{ROOT}/generalisation/hidden_sweep"
if not os.path.exists(hidden_path):
    hidden_path = f"{ROOT}/rank_study/hidden_sweep"  # fallback

hidden_ranks, hidden_means, hidden_stds = [], [], []
for d in sorted(os.listdir(hidden_path)):
    full = os.path.join(hidden_path, d)
    if not os.path.isdir(full):
        continue
    accs = []
    rank_val = None
    for f in sorted(glob.glob(f"{full}/*.json")):
        data = json.load(open(f))
        accs.append(data["best_test_acc"])
        if rank_val is None:
            rank_val = data.get("hidden_rank", int(''.join(filter(str.isdigit, d)) or 0))
    if accs:
        hidden_ranks.append(rank_val)
        hidden_means.append(np.mean(accs) * 100)
        hidden_stds.append(np.std(accs) * 100)

order = np.argsort(hidden_ranks)
hidden_ranks = [hidden_ranks[i] for i in order]
hidden_means = [hidden_means[i] for i in order]
hidden_stds  = [hidden_stds[i] for i in order]

axes[1].errorbar(hidden_ranks, hidden_means, yerr=hidden_stds, marker="s", capsize=4,
                 color="#8eb8e0", linewidth=2, markersize=8)
axes[1].set_xlabel("Hidden Layer Rank", fontsize=10)
axes[1].set_title("Hidden Rank Sweep\n(input=8, output=0)", fontsize=11, fontweight="bold")
axes[1].set_xticks(hidden_ranks)

# --- Output rank sweep ---
output_path = f"{ROOT}/rank_study/output_sweep"
output_ranks, output_means, output_stds = [], [], []
for d in sorted(os.listdir(output_path)):
    full = os.path.join(output_path, d)
    if not os.path.isdir(full):
        continue
    accs = []
    rank_val = None
    for f in sorted(glob.glob(f"{full}/*.json")):
        data = json.load(open(f))
        accs.append(data["best_test_acc"])
        if rank_val is None:
            rank_val = data.get("output_rank", int(''.join(filter(str.isdigit, d)) or 0))
    if accs:
        output_ranks.append(rank_val)
        output_means.append(np.mean(accs) * 100)
        output_stds.append(np.std(accs) * 100)

order = np.argsort(output_ranks)
output_ranks = [output_ranks[i] for i in order]
output_means = [output_means[i] for i in order]
output_stds  = [output_stds[i] for i in order]

axes[2].errorbar(output_ranks, output_means, yerr=output_stds, marker="D", capsize=4,
                 color="#e8a0a0", linewidth=2, markersize=8)
axes[2].set_xlabel("Output Layer Rank", fontsize=10)
axes[2].set_title("Output Rank Sweep\n(input=8, hidden=2)", fontsize=11, fontweight="bold")
axes[2].set_xticks(output_ranks)

fig.suptitle("Per-Layer Rank Saturation Curves (MNIST, n=3)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG}/04_rank_sweeps.png", dpi=200, bbox_inches="tight")
plt.savefig(f"{FIG}/04_rank_sweeps.pdf", bbox_inches="tight")
print(f"Saved to {FIG}/04_rank_sweeps.png")
plt.show()
