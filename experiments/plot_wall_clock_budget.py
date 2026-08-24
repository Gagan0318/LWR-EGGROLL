"""Wall-clock budget vs accuracy — combines tight_budget (60s, 120s) + wall_budget (300s)."""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs("figures", exist_ok=True)

# ── collect data: (method, budget) → [best_test_acc, ...] ──
data = defaultdict(list)

# tight_budget: results/tight_budget/budget{N}s/{method}/seed{n}.json
for budget_dir in glob.glob("results/tight_budget/budget*s"):
    budget = int(os.path.basename(budget_dir).replace("budget","").replace("s",""))
    for method_dir in sorted(glob.glob(os.path.join(budget_dir, "*"))):
        method = os.path.basename(method_dir)
        for f in glob.glob(os.path.join(method_dir, "*.json")):
            with open(f) as fh:
                r = json.load(fh)
            acc = r.get("best_test_acc", None)
            if acc is not None:
                data[(method, budget)].append(float(acc))

# wall_budget (300s): results/wall_budget/{method}/seed{n}.json
for method_dir in sorted(glob.glob("results/wall_budget/*")):
    if not os.path.isdir(method_dir): continue
    method = os.path.basename(method_dir)
    for f in glob.glob(os.path.join(method_dir, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        acc = r.get("best_test_acc", None)
        if acc is not None:
            data[(method, 300)].append(float(acc))

if not data:
    print("[!] No data found in results/tight_budget/ or results/wall_budget/")
    exit(1)

# ── organise ──
budgets = sorted(set(b for _, b in data.keys()))
all_methods = sorted(set(m for m, _ in data.keys()))

# pick interesting methods (LWR + best vanillas)
lwr_methods = [m for m in all_methods if m.startswith("lwr_")]
vanilla_methods = [m for m in all_methods if m.startswith("eggroll_")]
methods = lwr_methods + vanilla_methods
# filter to those with data at multiple budgets
methods = [m for m in methods if sum(1 for b in budgets if data.get((m,b))) >= 2]
if not methods:
    methods = all_methods[:6]

print(f"Budgets: {budgets}")
print(f"Methods: {methods}")
for m in methods:
    for b in budgets:
        vals = data.get((m, b), [])
        if vals: print(f"  {m} @ {b}s: {np.mean(vals)*100:.1f}% (n={len(vals)})")

# ── plot ──
fig, ax = plt.subplots(figsize=(10, 5.5))
x = np.arange(len(budgets))
n = len(methods)
w = 0.7 / n

colors_lwr = ['#2a78d6', '#1baf7a', '#4a3aa7']
colors_van = ['#eb6834', '#eda100', '#e87ba4', '#898781', '#e24b4a']
ci, cv = 0, 0

for i, m in enumerate(methods):
    if m.startswith("lwr_"):
        c = colors_lwr[ci % len(colors_lwr)]; ci += 1
    else:
        c = colors_van[cv % len(colors_van)]; cv += 1
    means = [np.mean(data[(m, b)])*100 if data.get((m, b)) else 0 for b in budgets]
    errs  = [np.std(data[(m, b)])*100 if len(data.get((m, b), [])) > 1 else 0 for b in budgets]
    offset = (i - n/2 + 0.5) * w
    ax.bar(x + offset, means, w, yerr=errs, capsize=3, label=m, color=c,
           edgecolor='white', linewidth=0.5)

ax.set_xlabel("Wall-clock budget (seconds)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Wall-clock budget vs accuracy (MNIST, σ=0.05)")
ax.set_xticks(x)
ax.set_xticklabels([f"{b}s" for b in budgets])
ax.legend(fontsize=8, framealpha=0.9, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig("figures/wall_clock_budget.png", dpi=200, bbox_inches='tight')
print("Saved → figures/wall_clock_budget.png")
plt.close()
