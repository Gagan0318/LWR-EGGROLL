"""Population x rank interaction.
N in {256,512,1024,4096} x r in {1,4,16} x 3 seeds = 36 runs. ~1-2 hours.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import time
import numpy as np
from experiments.compare_4_methods_mnist import CFG, load_mnist, train_eggroll

SEEDS = (0, 1, 2)
SIGMA = 0.05
WALL_BUDGET = 300.0
POP_SIZES = (256, 512, 1024, 4096)
RANKS = (1, 4, 16)
RESULTS_DIR = Path("results/pop_rank_interaction")

def main():
    X_train, y_train, X_test, y_test = load_mnist()
    total = len(POP_SIZES) * len(RANKS) * len(SEEDS)
    print(f"Total runs: {total}")
    for pop in POP_SIZES:
        for rank in RANKS:
            for seed in SEEDS:
                tag = f"pop{pop}_r{rank}"
                outdir = RESULTS_DIR / tag
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"seed{seed}.json"
                if fname.exists():
                    print(f"[skip] {fname}"); continue
                CFG.eggroll_sigma_init = SIGMA
                CFG.eggroll_pop = pop
                CFG.max_wall_seconds = WALL_BUDGET
                print(f"\n[N={pop} r={rank} seed={seed}]", flush=True)
                t0 = time.time()
                result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank)
                wall = time.time() - t0
                out = dict(result) if isinstance(result, dict) else {"result": str(result)}
                out.update({"pop_size": pop, "rank": rank, "seed": seed, "sigma": SIGMA, "wall_seconds": wall})
                safe = {k: v for k, v in out.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
                fname.write_text(json.dumps(safe, indent=2, default=str))
    summary = {}
    for pop in POP_SIZES:
        for rank in RANKS:
            tag = f"pop{pop}_r{rank}"
            accs = []
            for seed in SEEDS:
                fname = RESULTS_DIR / tag / f"seed{seed}.json"
                if fname.exists():
                    with open(fname) as f:
                        d = json.load(f)
                    if d.get("best_test_acc") is not None:
                        accs.append(d["best_test_acc"])
            summary[tag] = {"pop": pop, "rank": rank, "acc_mean": float(np.mean(accs)) if accs else None, "acc_std": float(np.std(accs)) if accs else None, "n_seeds": len(accs)}
    out_path = RESULTS_DIR / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSummary: {out_path}")
    print(f"\n{'N':>6} {'r':>4} {'acc_mean':>10} {'acc_std':>10}")
    print("-" * 35)
    for pop in POP_SIZES:
        for rank in RANKS:
            v = summary[f"pop{pop}_r{rank}"]
            if v["acc_mean"] is not None:
                print(f"{pop:>6} {rank:>4} {v['acc_mean']:>10.4f} {v['acc_std']:>10.4f}")
    print("\nDone.")

if __name__ == "__main__":
    main()
