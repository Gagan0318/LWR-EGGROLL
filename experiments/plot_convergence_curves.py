"""Convergence curves — accuracy vs generation for key methods.
Uses cross_dataset or lwr_allocation_controls for MNIST data.
History format: [[wall_s, gen_step, accuracy], ...]"""
import json, glob, os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs("figures", exist_ok=True)

# ── methods of interest with display names and colors ──
METHODS = [
    ("lwr_8_4_0", "LWR (8,4,0)", '#2a78d6'),
    ("lwr_8_2_0", "LWR (8,2,0)", '#1baf7a'),
    ("lwr_0_2_8", "Reversed (0,2,8)", '#e24b4a'),
    ("eggroll_r4", "Vanilla r=4", '#eb6834'),
]

curves = defaultdict(list)

# scan lwr_allocation_controls
for method_dir in glob.glob("results/lwr_allocation_controls/*"):
    method = os.path.basename(method_dir)
    for f in glob.glob(os.path.join(method_dir, "*.json")):
        with open(f) as fh:
            r = json.load(fh)
        hist = r.get("history", [])
        if hist:
            for key, _, _ in METHODS:
                if key == method:
                    curves[key].append(hist)
                    break

# also scan cross_dataset for MNIST-equivalent (fashion as alternative)
for ds in ["fashion", "kmnist", "emnist"]:
    for subtype in ["lwr", "vanilla"]:
        for method_dir in glob.glob(f"results/cross_dataset/{ds}/{subtype}/*"):
            method = os.path.basename(method_dir)
            # only add if we don't have enough MNIST data
            if method in [k for k, _, _ in METHODS] and len(curves.get(method, [])) < 3:
                for f in glob.glob(os.path.join(method_dir, "*.json")):
                    with open(f) as fh:
                        r = json.load(fh)
                    hist = r.get("history", [])
                    if hist and len(curves.get(method, [])) < 5:
                        curves[method].append(hist)

# also try wall_budget which has convergence histories
for method_dir in glob.glob("results/wall_budget/*"):
    if not os.path.isdir(method_dir): continue
    method = os.path.basename(method_dir)
    if method in [k for k, _, _ in METHODS] and len(curves.get(method, [])) < 3:
        for f in glob.glob(os.path.join(method_dir, "*.json")):
            with open(f) as fh:
                r = json.load(fh)
            hist = r.get("history", [])
            if hist:
                curves[method].append(hist)

if not curves:
    print("[!] No convergence histories found")
    exit(1)

print("Methods with histories:")
for k in curves:
    print(f"  {k}: {len(curves[k])} seeds, {len(curves[k][0])} points in first")

# ── plot ──
fig, ax = plt.subplots(figsize=(10, 5))

for key, display, color in METHODS:
    if key not in curves: continue
    all_gens = set()
    for hist in curves[key]:
        for h in hist:
            all_gens.add(int(h[1]))  # h = [wall_s, gen_step, accuracy]
    gen_grid = sorted(all_gens)

    seed_accs = []
    for hist in curves[key]:
        gens = [int(h[1]) for h in hist]
        accs = [float(h[2]) for h in hist]
        interp = np.interp(gen_grid, gens, accs)
        seed_accs.append(interp)

    seed_accs = np.array(seed_accs)
    mean_acc = np.mean(seed_accs, axis=0) * 100
    ax.plot(gen_grid, mean_acc, label=display, color=color, linewidth=2)
    if seed_accs.shape[0] > 1:
        std_acc = np.std(seed_accs, axis=0) * 100
        ax.fill_between(gen_grid, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.15, color=color)

ax.set_xlabel("Generation (eval step)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Convergence curves — accuracy vs generation")
ax.legend(fontsize=9, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/convergence_curves.png", dpi=200, bbox_inches='tight')
print("Saved → figures/convergence_curves.png")
plt.close()
