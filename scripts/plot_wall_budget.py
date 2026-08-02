"""Plot wall_budget_sweep results.

Three figures:
  1. Accuracy at 300s budget — bar chart per config
  2. Budget-per-gen vs accuracy — scatter with vanilla vs LWR markers
  3. Generations reached at budget — shows compute-cost differences

Updated 30 Jul 2026: reads generation counts from individual seed files
since the summary doesn't store them.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results/wall_budget/summary.json")
SEED_DIR = Path("results/wall_budget")
FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

with RESULTS.open() as f:
    data = json.load(f)

summary = data["summary"]
configs = list(summary.keys())
vanilla = [c for c in configs if summary[c]["method"] == "eggroll"]
lwr = [c for c in configs if summary[c]["method"] == "lwr"]

# Enrich summary with generation counts from seed files
for c in configs:
    gens = []
    config_dir = SEED_DIR / c
    if config_dir.is_dir():
        for seed_file in sorted(config_dir.glob("seed*.json")):
            with seed_file.open() as f:
                seed_data = json.load(f)
            # Try common key names for generation count
            g = (seed_data.get("converged_at_step") or seed_data.get("generations") or seed_data.get("gen")
                 or seed_data.get("n_generations") or seed_data.get("total_gens"))
            if g is not None:
                gens.append(g)
    summary[c]["gens_mean"] = float(np.mean(gens)) if gens else None
    summary[c]["gens_std"] = float(np.std(gens)) if gens else None

# ---- Fig 1: accuracy bar chart ----
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(configs))
accs = [summary[c]["acc_mean"] for c in configs]
stds = [summary[c]["acc_std"] for c in configs]
colors = ["steelblue" if summary[c]["method"] == "eggroll" else "coral" for c in configs]
ax.bar(x, accs, yerr=stds, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(configs, rotation=45, ha="right")
ax.set_ylabel("best test accuracy (mean ± std, n=3)")
ax.set_title(f"Best accuracy achieved within {data['wall_budget_seconds']:.0f}s wall budget")
ax.grid(alpha=0.3, axis="y")
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="steelblue", label="vanilla EGGROLL"),
    Patch(color="coral", label="LWR-EGGROLL"),
])
fig.tight_layout()
fig.savefig(FIGDIR / "wall_budget_accuracy.png", dpi=150)
print(f"Saved {FIGDIR / 'wall_budget_accuracy.png'}")

# ---- Fig 2: budget-per-gen vs accuracy scatter ----
fig, ax = plt.subplots(figsize=(7, 4.5))
for group, marker, color, label in [
    (vanilla, "o", "steelblue", "vanilla EGGROLL"),
    (lwr, "s", "coral", "LWR-EGGROLL"),
]:
    xs = [summary[c]["budget_per_gen"] for c in group]
    ys = [summary[c]["acc_mean"] for c in group]
    yerr = [summary[c]["acc_std"] for c in group]
    ax.errorbar(xs, ys, yerr=yerr, marker=marker, linestyle="",
                markersize=9, capsize=3, color=color, label=label)
    for c, xi, yi in zip(group, xs, ys):
        ax.annotate(c.replace("eggroll_", "").replace("lwr_", ""),
                    (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)
ax.set_xlabel("rank budget per generation")
ax.set_ylabel(f"best test accuracy at {data['wall_budget_seconds']:.0f}s")
ax.set_title("Compute efficiency: rank budget vs achieved accuracy")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(FIGDIR / "wall_budget_efficiency.png", dpi=150)
print(f"Saved {FIGDIR / 'wall_budget_efficiency.png'}")

# ---- Fig 3: generations reached ----
has_gens = all(summary[c]["gens_mean"] is not None for c in configs)
if has_gens:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    gens = [summary[c]["gens_mean"] for c in configs]
    gens_std = [summary[c]["gens_std"] for c in configs]
    ax.bar(x, gens, yerr=gens_std, color=colors, capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=45, ha="right")
    ax.set_ylabel("generations reached within budget (mean ± std)")
    ax.set_title(f"Compute cost: generations completed within {data['wall_budget_seconds']:.0f}s")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGDIR / "wall_budget_generations.png", dpi=150)
    print(f"Saved {FIGDIR / 'wall_budget_generations.png'}")
else:
    print("[info] Skipping generations figure — no generation counts found in seed files.")
    print("       Expected keys: 'generations', 'gen', 'n_generations', or 'total_gens'")
    # Show what keys are available in a sample seed file for debugging
    sample = next(SEED_DIR.glob("*/seed0.json"), None)
    if sample:
        with sample.open() as f:
            keys = list(json.load(f).keys())
        print(f"       Available keys in {sample.name}: {keys}")

# Summary
print(f"\nResults at {data['wall_budget_seconds']:.0f}s budget:")
for c in configs:
    s = summary[c]
    gens_str = f"  ({s['gens_mean']:.0f} gens)" if s.get("gens_mean") else ""
    print(f"  {c}: {s['acc_mean']:.4f} ± {s['acc_std']:.4f}{gens_str}")
