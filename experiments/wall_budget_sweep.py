"""Wall-clock budget experiment — fixed 300s per run.

Grid: 6 vanilla ranks + 3 LWR allocations × 3 seeds = 27 runs.
Fixed: sigma=0.05, pop=2048, arch=[256, 256, 256].
"""
from __future__ import annotations

import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent.parent))

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from experiments.compare_4_methods_mnist import (
    CFG,
    load_mnist,
    train_eggroll,
    train_lwr_eggroll,
)


VANILLA_RANKS = (1, 2, 4, 8, 16, 32)
LWR_ALLOCATIONS = (
    (2, 1, 0),
    (4, 2, 1),
    (8, 2, 0),
)
SEEDS = (0, 1, 2)
WALL_BUDGET = 300.0
SIGMA = 0.05
POP = 2048
RESULTS_ROOT = Path("results/wall_budget")


def build_configs():
    configs = []
    for r in VANILLA_RANKS:
        configs.append({
            "name": f"eggroll_r{r}",
            "method": "eggroll",
            "rank_spec": r,
            "budget_per_gen": r,
        })
    for alloc in LWR_ALLOCATIONS:
        alloc_name = "_".join(str(x) for x in alloc)
        # Map (input, hidden, output) triple to per-shape dict
        # for arch [784 -> 256 -> 256 -> 256 -> 10]
        r_in, r_hidden, r_out = alloc
        rank_spec = {
            (256, 784): r_in,
            (256, 256): r_hidden,
            (10, 256): r_out,
        }
        configs.append({
            "name": f"lwr_{alloc_name}",
            "method": "lwr",
            "rank_spec": rank_spec,
            "alloc_str": str(alloc),
            "budget_per_gen": sum(alloc),
        })
    return configs


def run_single(config, seed, X_train, y_train, X_test, y_test):
    # Mutate CFG globals — the training code reads these directly.
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL_BUDGET

    t0 = time.time()
    if config["method"] == "eggroll":
        result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=config["rank_spec"])
    else:
        result = train_lwr_eggroll(
            seed, X_train, y_train, X_test, y_test,
            rank_spec=config["rank_spec"],
            label=config["name"],
        )
    wall = time.time() - t0

    # Normalise into a plain dict (result from _train_hyperscalees_common is a dict)
    out = dict(result) if isinstance(result, dict) else {"result": str(result)}
    out.update({
        "config_name": config["name"],
        "method": config["method"],
        "rank_spec": str(config["rank_spec"]),
        "budget_per_gen": config["budget_per_gen"],
        "seed": seed,
        "wall_seconds_measured": wall,
        "wall_budget": WALL_BUDGET,
        "hit_wall_cap": out.get("stop_reason") == "wall_cap" if isinstance(result, dict) else None,
        "converged_before_cap": out.get("stop_reason") == "patience" if isinstance(result, dict) else None,
    })
    return out


def main():
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    print("[main] Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    configs = build_configs()
    print(f"Total runs: {len(configs) * len(SEEDS)}")
    print(f"Wall budget per run: {WALL_BUDGET}s")

    all_results = {c["name"]: [] for c in configs}

    for config in configs:
        outdir = RESULTS_ROOT / config["name"]
        outdir.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname.name}")
                with fname.open() as f:
                    all_results[config["name"]].append(json.load(f))
                continue
            print(f"\n[{config['name']} seed={seed}]", flush=True)
            result = run_single(config, seed, X_train, y_train, X_test, y_test)
            # Strip non-JSON-serialisable fields defensively
            safe = {k: v for k, v in result.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
            fname.write_text(json.dumps(safe, indent=2, default=str))
            all_results[config["name"]].append(safe)

    # Aggregate summary
    summary = {}
    for name, runs in all_results.items():
        accs = [r.get("best_test_acc") for r in runs if r.get("best_test_acc") is not None]
        summary[name] = {
            "method": runs[0]["method"],
            "budget_per_gen": runs[0]["budget_per_gen"],
            "acc_mean": float(np.mean(accs)) if accs else float("nan"),
            "acc_std": float(np.std(accs)) if accs else float("nan"),
            "n_seeds": len(runs),
        }

    out = {
        "summary": summary,
        "wall_budget_seconds": WALL_BUDGET,
        "vanilla_ranks": list(VANILLA_RANKS),
        "lwr_allocations": [list(a) for a in LWR_ALLOCATIONS],
        "seeds": list(SEEDS),
    }
    (RESULTS_ROOT / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nDone. Summary at {RESULTS_ROOT / 'summary.json'}")


if __name__ == "__main__":
    main()
