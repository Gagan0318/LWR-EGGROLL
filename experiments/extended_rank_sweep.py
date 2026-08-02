"""Extended rank granularity sweep — r ∈ {12, 24, 32}.

Adds three high-rank cells to existing sigma×rank sweep at n=5 seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import json
import time
from pathlib import Path

from experiments.compare_4_methods_mnist import (
    CFG,
    load_mnist,
    train_eggroll,
)


SIGMAS = (0.01, 0.03, 0.05, 0.1, 0.3)
NEW_RANKS = (12, 24, 32)
SEEDS = (0, 1, 2, 3, 4)
POP = 2048
RESULTS_ROOT = Path("results/variance_rank")


def run_single(sigma, rank, seed, X_train, y_train, X_test, y_test):
    CFG.eggroll_sigma_init = sigma
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = 1800.0

    t0 = time.time()
    result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank)
    wall = time.time() - t0

    out = dict(result) if isinstance(result, dict) else {"result": str(result)}
    out.update({
        "sigma": sigma,
        "rank": rank,
        "seed": seed,
        "wall_seconds": wall,
    })
    return out


def main():
    print("[main] Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    for sigma in SIGMAS:
        for rank in NEW_RANKS:
            cell_dir = RESULTS_ROOT / f"sig{sigma}_r{rank}"
            cell_dir.mkdir(parents=True, exist_ok=True)
            for seed in SEEDS:
                fname = cell_dir / f"vsweep_sig{sigma}_r{rank}_seed{seed}.json"
                if fname.exists():
                    print(f"[skip] {fname.name}")
                    continue
                print(f"\n[σ={sigma} r={rank} seed={seed}]", flush=True)
                result = run_single(sigma, rank, seed, X_train, y_train, X_test, y_test)
                safe = {k: v for k, v in result.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
                fname.write_text(json.dumps(safe, indent=2, default=str))

    print("\nDone. Run merge_variance_rank.py next.")


if __name__ == "__main__":
    main()
