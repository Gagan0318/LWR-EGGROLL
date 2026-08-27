"""Brax Ant — convergence + summary bar. Run AFTER tonight's results land.
Checks results/brax_ant_mini/ and Google Drive symlink."""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs("figures", exist_ok=True)

# check multiple paths
SEARCH_PATHS = [
    "results/brax_ant_mini",
    "results/brax_ant",
    os.path.expanduser("~/drive/dissertation/results/brax_ant"),
    os.path.expanduser("~/drive/My Drive/dissertation/results/brax_ant"),
]

files = []
source = None
for p in SEARCH_PATHS:
    found = sorted(glob.glob(os.path.join(p, "**/*.json"), recursive=True))
    if found:
        files = found
        source = p
        break

if not files:
    print("[!] No Brax Ant results found. Checked:")
    for p in SEARCH_PATHS:
        print(f"    {p}")
    print("    Run this after tonight's experiment completes.")
    exit(1)

print(f"Found {len(files)} files in {source}/")

curves = defaultdict(list)
finals = defaultdict(list)

for f in files:
    if "summary" in os.path.basename(f) or "pilot" in os.path.basename(f): continue
    with open(f) as fh:
        r = json.load(fh)
    method = r.get("method", os.path.basename(f).rsplit("_seed", 1)[0])
    hist = r.get("history", [])
    best = r.get("best_fitness", r.get("best_reward", r.get("best_test_acc", None)))
    if best is None and hist:
        best = max(h[2] if len(h) >= 3 else h[-1] for h in hist)
    if hist:
        curves[method].append(hist)
    if best is not None:
        finals[method].append(float(best))
        print(f"  {method}: best={best:.1f}")

def get_color(label):
    l = label.lower()
    if 'lwr' in l and '8_4_0' in l: return '#2a78d6'
    if 'lwr' in l and '4_2_0' in l: return '#1baf7a'
    if 'lwr' in l: return '#4a3aa7'
    if 'r1' in l: return '#eda100'
    if 'r4' in l: return '#eb6834'
    return '#898781'

# PLOT 1: convergence
if curves:
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in sorted(curves.keys()):
        all_gens = set()
        for hist in curves[method]:
            for h in hist:
                all_gens.add(int(h[1] if len(h) >= 3 else h[0]))
        gen_grid = sorted(all_gens)
        seed_r = []
        for hist in curves[method]:
            gens = [int(h[1] if len(h) >= 3 else h[0]) for h in hist]
            rews = [float(h[2] if len(h) >= 3 else h[-1]) for h in hist]
            seed_r.append(np.interp(gen_grid, gens, rews))
        seed_r = np.array(seed_r)
        mean_r = np.mean(seed_r, axis=0)
        color = get_color(method)
        ax.plot(gen_grid, mean_r, label=method, color=color, linewidth=2)
        if seed_r.shape[0] > 1:
            ax.fill_between(gen_grid, mean_r - np.std(seed_r, axis=0),
                            mean_r + np.std(seed_r, axis=0), alpha=0.15, color=color)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Reward")
    ax.set_title("Brax Ant — reward convergence (deterministic RL)")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/brax_ant_convergence.png", dpi=200, bbox_inches='tight')
    print("Saved → figures/brax_ant_convergence.png")
    plt.close()

# PLOT 2: summary bar
if finals:
    methods = sorted(finals.keys(), key=lambda m: np.mean(finals[m]))
    means = [np.mean(finals[m]) for m in methods]
    stds  = [np.std(finals[m]) if len(finals[m]) > 1 else 0 for m in methods]
    colors = [get_color(m) for m in methods]
    fig, ax = plt.subplots(figsize=(9, max(3, len(methods)*0.7+1)))
    y = np.arange(len(methods))
    ax.barh(y, means, xerr=stds, capsize=4, color=colors, edgecolor='white', linewidth=0.5, height=0.5)
    for i in range(len(methods)):
        ax.text(means[i]+stds[i]+5, i, f"{means[i]:.1f}±{stds[i]:.1f}", va='center', fontsize=9, color='#52514e')
    ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=10)
    ax.set_xlabel("Best reward")
    ax.set_title("Brax Ant — final reward by method")
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/brax_ant_summary.png", dpi=200, bbox_inches='tight')
    print("Saved → figures/brax_ant_summary.png")
    plt.close()
