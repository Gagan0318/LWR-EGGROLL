"""LunarLander symmetric tuned — horizontal bar with error bars.
Reads results/lunarlander_symmetric_tuned/{method}_seed{n}.json
Fields: method, seed, best_fitness, history"""
import json, glob, os, re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

os.makedirs("figures", exist_ok=True)

data = defaultdict(list)

for f in sorted(glob.glob("results/lunarlander_symmetric_tuned/*.json")):
    if "summary" in f: continue
    with open(f) as fh:
        r = json.load(fh)
    method = r.get("method", os.path.basename(f).rsplit("_seed", 1)[0])
    reward = r.get("best_fitness", None)
    if reward is not None:
        data[method].append(float(reward))

if not data:
    print("[!] No data in results/lunarlander_symmetric_tuned/")
    exit(1)

# sort by mean reward
methods = sorted(data.keys(), key=lambda m: np.mean(data[m]))
means = [np.mean(data[m]) for m in methods]
stds  = [np.std(data[m]) if len(data[m]) > 1 else 0 for m in methods]

# rank budget from method name
def get_budget(m):
    nums = re.findall(r'\d+', m.split("seed")[0])
    if m.startswith("eggroll_r"):
        r = int(nums[0]) if nums else 0
        return r * 3  # 3 layers
    elif m.startswith("lwr_") or m.startswith("openai") or m.startswith("reinforce"):
        if m.startswith("lwr_") and len(nums) >= 3:
            return sum(int(n) for n in nums[:3])
    return None

def method_color(m):
    if 'capped_4_2_1' in m: return '#2a78d6'
    if 'lwr_' in m: return '#1baf7a'
    if 'eggroll' in m: return '#eb6834'
    if 'openai' in m: return '#eda100'
    if 'reinforce' in m: return '#e87ba4'
    return '#898781'

colors = [method_color(m) for m in methods]

fig, ax = plt.subplots(figsize=(11, max(4, len(methods) * 0.55 + 1)))
y = np.arange(len(methods))

ax.barh(y, means, xerr=stds, capsize=4, color=colors,
        edgecolor='white', linewidth=0.5, height=0.5)

for i, m in enumerate(methods):
    budget = get_budget(m)
    budget_str = f"  (budget={budget})" if budget else ""
    ax.text(max(0, means[i] + stds[i] + 3), i, f"{means[i]:.1f} ± {stds[i]:.1f}{budget_str}",
            va='center', fontsize=8, color='#52514e')

ax.set_yticks(y)
ax.set_yticklabels(methods, fontsize=9)
ax.set_xlabel("Best reward")
ax.set_title("LunarLander symmetric tuned — all methods")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig("figures/lunarlander_comparison.png", dpi=200, bbox_inches='tight')
print("Saved → figures/lunarlander_comparison.png")
plt.close()
