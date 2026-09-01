#!/usr/bin/env python3
import json, sys, os, glob
from collections import defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
files = sorted(glob.glob(os.path.join(ROOT, "**", "*.json"), recursive=True))
if not files:
    print(f"No .json files under {ROOT!r}"); sys.exit(0)

def num(x):
    try: return float(x)
    except: return None

def bsf_of(h):
    # tolerate dict entries, [gen,val] pairs, or bare numbers
    if isinstance(h, dict): return num(h.get("best_so_far"))
    if isinstance(h, (list, tuple)) and h: return num(h[-1])
    return num(h)

experiments, others = defaultdict(list), []
for path in files:
    try: d = json.load(open(path))
    except Exception as e: print(f"[skip] {path}: {e}"); continue
    rel = os.path.relpath(path, ROOT)
    if isinstance(d, dict) and isinstance(d.get("history"), list):
        bsf = [b for b in (bsf_of(h) for h in d["history"]) if b is not None]
        experiments[d.get("method","?")].append(dict(
            path=rel, seed=d.get("seed"), top_best=num(d.get("best_fitness")),
            final_bsf=(bsf[-1] if bsf else num(d.get("best_fitness"))),
            max_bsf=(max(bsf) if bsf else None),
            mean=num(d.get("final_mean_fitness")), gens=d.get("generations"),
            wall_min=(num(d.get("wall_seconds"))/60 if num(d.get("wall_seconds")) else None),
            n_hist=len(d["history"])))
    else:
        others.append((rel, d))

print("="*70); print("EXPERIMENT RESULTS (per seed)"); print("="*70)
for method in sorted(experiments):
    print(f"\n### {method}")
    vals = []
    for r in sorted(experiments[method], key=lambda r:(r['seed'] is None, r['seed'])):
        v = r['final_bsf']; vals.append(v); warn = ""
        if r['top_best'] is not None and v is not None and abs(r['top_best']-v) > 1e-3:
            warn += f"  [!top_best={r['top_best']:.2f}]"
        if r['n_hist'] and r['gens'] and r['n_hist'] < r['gens']:
            warn += f"  [!TRUNCATED {r['n_hist']}/{r['gens']}]"
        vs = f"{v:.2f}" if v is not None else "NA"
        ms = f"{r['mean']:.2f}" if r['mean'] is not None else "NA"
        wm = f"{r['wall_min']:.0f}min" if r['wall_min'] is not None else "NA"
        print(f"  seed {r['seed']}: best_so_far={vs}  mean={ms}  wall={wm}  gens={r['gens']}{warn}   ({r['path']})")
    good = [v for v in vals if v is not None]
    if good: print(f"  MEAN best_so_far = {sum(good)/len(good):.2f}  (n={len(good)})")

print("\n"+"="*70); print("OTHER JSONs (pilots / sweeps / configs) - full dump"); print("="*70)
def show(o, ind=1, cap=15):
    p = "  "*ind
    if isinstance(o, dict):
        for k,v in o.items():
            if isinstance(v,(dict,list)): print(f"{p}{k}:"); show(v, ind+1, cap)
            else: print(f"{p}{k}: {v}")
    elif isinstance(o, list):
        if len(o) > cap and all(not isinstance(x,(dict,list)) for x in o):
            print(f"{p}[{len(o)} vals] {o[:cap]} ...")
        else:
            for x in o[:cap]:
                show(x, ind+1, cap) if isinstance(x,(dict,list)) else print(f"{p}- {x}")
            if len(o) > cap: print(f"{p}... (+{len(o)-cap} more)")
    else: print(f"{p}{o}")
for rel, d in others:
    print(f"\n### {rel}"); show(d)

print("\n"+"="*70); print("HYPERPARAMETERS FOUND across all files"); print("="*70)
TOKENS = ("sigma","noise_std","learning_rate","alpha","population","popsize",
          "population_size","generation","rank","seed","std","npop")
seen = defaultdict(set)
def hunt(o):
    if isinstance(o, dict):
        for k,v in o.items():
            if not isinstance(v,(dict,list)) and any(t in k.lower() for t in TOKENS):
                seen[k].add(str(v))
            hunt(v)
    elif isinstance(o, list):
        for x in o: hunt(x)
for path in files:
    try: hunt(json.load(open(path)))
    except: pass
for k in sorted(seen): print(f"  {k}: {sorted(seen[k])}")
if not seen: print("  (none in JSONs - likely in the notebook/config, not result files)")
