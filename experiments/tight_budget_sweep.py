"""Tight wall-budget experiment — 60s and 120s budgets.
Tests whether LWR's advantage grows under genuine time pressure.
6 configs × 2 budgets × 3 seeds = 36 runs. ~1 hour.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import time
import numpy as np
from experiments.compare_4_methods_mnist import (
    CFG,
    load_mnist,
    train_eggroll,
    train_lwr_eggroll,
)

SEEDS = (0, 1, 2)
SIGMA = 0.05
POP = 2048
BUDGETS = (60.0, 120.0)

CONFIGS = [
    {"name": "eggroll_r1",  "method": "eggroll", "rank_spec": 1},
    {"name": "eggroll_r4",  "method": "eggroll", "rank_spec": 4},
    {"name": "eggroll_r8",  "method": "eggroll", "rank_spec": 8},
    {"name": "lwr_8_2_0",   "method": "lwr",     "rank_spec": {(256,784): 8, (256,256): 2, (10,256): 0}},
    {"name": "lwr_4_1_0",   "method": "lwr",     "rank_spec": {(256,784): 4, (256,256): 1, (10,256): 0}},
    {"name": "lwr_2_1_0",   "method": "lwr",     "rank_spec": {(256,784): 2, (256,256): 1, (10,256): 0}},
]

RESULTS_DIR = Path("results/tight_budget")


def main():
    X_train, y_train, X_test, y_test = load_mnist()
    total = len(BUDGETS) * len(CONFIGS) * len(SEEDS)
    print(f"Total runs: {total}")

    for budget in BUDGETS:
        print(f"\n{'='*60}")
        print(f"  WALL BUDGET: {budget:.0f}s")
        print(f"{'='*60}")

        for config in CONFIGS:
            for seed in SEEDS:
                tag = f"budget{int(budget)}s/{config['name']}"
                outdir = RESULTS_DIR / f"budget{int(budget)}s" / config["name"]
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"seed{seed}.json"
                if fname.exists():
                    print(f"[skip] {fname}")
                    continue

                CFG.eggroll_sigma_init = SIGMA
                CFG.eggroll_pop = POP
                CFG.max_wall_seconds = budget

                print(f"\n[{tag} seed={seed}]", flush=True)
                t0 = time.time()
                if config["method"] == "eggroll":
                    result = train_eggroll(seed, X_train, y_train, X_test, y_test,
                                           rank=config["rank_spec"])
                else:
                    result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                               rank_spec=config["rank_spec"],
                                               label=config["name"])
                wall = time.time() - t0

                out = dict(result) if isinstance(result, dict) else {"result": str(result)}
                out.update({
                    "config_name": config["name"],
                    "method": config["method"],
                    "rank_spec": str(config["rank_spec"]),
                    "seed": seed,
                    "wall_budget": budget,
                    "wall_seconds_measured": wall,
                    "hit_wall_cap": out.get("stop_reason") == "wall_cap",
                    "converged_before_cap": out.get("stop_reason") == "patience",
                })
                safe = {k: v for k, v in out.items()
                        if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
                fname.write_text(json.dumps(safe, indent=2, default=str))

    # Summary per budget
    for budget in BUDGETS:
        budget_dir = RESULTS_DIR / f"budget{int(budget)}s"
        print(f"\n--- Budget {int(budget)}s ---")
        print(f"{'Config':>15} {'acc_mean':>10} {'acc_std':>10} {'wall_capped':>12}")
        print("-" * 50)
        for config in CONFIGS:
            config_dir = budget_dir / config["name"]
            accs = []
            capped = 0
            for seed in SEEDS:
                fname = config_dir / f"seed{seed}.json"
                if fname.exists():
                    with open(fname) as f:
                        d = json.load(f)
                    if d.get("best_test_acc") is not None:
                        accs.append(d["best_test_acc"])
                    if d.get("hit_wall_cap"):
                        capped += 1
            if accs:
                print(f"{config['name']:>15} {np.mean(accs):>10.4f} {np.std(accs):>10.4f} {capped:>12}/{len(accs)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
