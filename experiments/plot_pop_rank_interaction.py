"""Population × rank interaction from results/pop_rank_interaction/pop{N}_r{R}/"""
import json, glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs("figures", exist_ok=True)

data = defaultdict(list)

for subdir in sorted(glob.glob("results/pop_rank_interaction/pop*_r*")):
    dirname = os.path.basename(subdir)
    m = re.match(r'pop(\d+)_r(\d+)', dirname)
    if not m: continue
    pop, rank = int(m.group(1)), int(m.group(2))
    for f in glob.glob(os.path.join(subdir, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        acc = r.get("best_test_acc", None)
        if acc is not None:
            data[(pop, rank)].append(float(acc))

if not data:
    print("[!] No data found in results/pop_rank_interaction/")
    exit(1)

pops = sorted(set(p for p, _ in data.keys()))
ranks = sorted(set(r for _, r in data.keys()))
print(f"Populations: {pops}")
print(f"Ranks: {ranks}")

fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
markers = ['o', 's', '^', 'D', 'v']

for i, rank in enumerate(ranks):
    means, errs = [], []
    for pop in pops:
        vals = data.get((pop, rank), [])
        means.append(np.mean(vals)*100 if vals else np.nan)
        errs.append(np.std(vals)*100 if len(vals) > 1 else 0)
    ax.errorbar(pops, means, yerr=errs, marker=markers[i % len(markers)], markersize=7,
                color=colors[i % len(colors)], label=f"rank {rank}", linewidth=2, capsize=3)

ax.set_xlabel("Population size N")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Population × rank interaction — rank matters at small N, irrelevant at large N")
ax.set_xscale('log', base=2)
ax.set_xticks(pops)
ax.set_xticklabels([str(p) for p in pops])
ax.legend(fontsize=9, framealpha=0.9, title="Rank")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/pop_rank_interaction.png", dpi=200, bbox_inches='tight')
print("Saved → figures/pop_rank_interaction.png")
plt.close()
