"""
Validation of the LWR-EGGROLL allocation derived from the sensitivity pilot.

Compares:
  A. uniform_r4       — matched-budget baseline (total rank = 12)
  B. lwr_derived_min1 — pilot allocation with output floored at rank 1
                        (input=8, hidden=2, output=1, total=11)
  C. lwr_derived_r0   — pilot allocation with output at rank 0
                        (input=8, hidden=2, output=0, total=10)

n=5 seeds each. Outputs per-run JSON to results/validation/ and a summary
to results/validation/summary.json.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

import json
import time
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

sys.path.insert(0, str(Path(__file__).parent))
from compare_4_methods_mnist import (
    CFG, load_mnist, _train_hyperscalees_common,
)

import hyperscalees as hs
from hyperscalees.noiser.lwr_eggroll import LWREggRoll


PROJECT_ROOT = Path(__file__).parent.parent
VALIDATION_DIR = PROJECT_ROOT / "results" / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


INPUT_SHAPE = (256, 784)
HIDDEN_SHAPE = (256, 256)
OUTPUT_SHAPE = (10, 256)

VALIDATION_SEEDS = (0, 1, 2, 3, 4)

CONFIGS = [
    {
        "label": "uniform_r4",
        "rank_spec": {
            INPUT_SHAPE: 4,
            HIDDEN_SHAPE: 4,
            OUTPUT_SHAPE: 4,
        },
        "note": "Uniform r=4 baseline. Total rank budget = 12.",
    },
    {
        "label": "lwr_derived_min1",
        "rank_spec": {
            INPUT_SHAPE: 8,
            HIDDEN_SHAPE: 2,
            OUTPUT_SHAPE: 1,
        },
        "note": (
            "Pilot-derived allocation with output floored at rank 1. "
            "Total rank budget = 11."
        ),
    },
    {
        "label": "lwr_derived_r0",
        "rank_spec": {
            INPUT_SHAPE: 8,
            HIDDEN_SHAPE: 2,
            OUTPUT_SHAPE: 0,
        },
        "note": (
            "Pilot-derived allocation with output at rank 0 "
            "(no perturbation). Total rank budget = 10."
        ),
    },
]


def save_result(result, subdir):
    (VALIDATION_DIR / subdir).mkdir(exist_ok=True)
    fn = VALIDATION_DIR / subdir / f"{result['method']}_seed{result['seed']}.json"
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {fn}")


def run_validation():
    print("\n" + "=" * 70)
    print("LWR-EGGROLL VALIDATION")
    print("=" * 70)

    X_train, y_train, X_test, y_test = load_mnist()

    all_results = []

    for cfg in CONFIGS:
        print("\n" + "-" * 70)
        print(f"CONFIG: {cfg['label']}")
        print(f"  rank_spec: {cfg['rank_spec']}")
        print(f"  note: {cfg['note']}")
        print("-" * 70)

        for seed in VALIDATION_SEEDS:
            print(f"\n>>> {cfg['label']}  seed={seed}")

            result = _train_hyperscalees_common(
                seed=seed,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                noiser_class=LWREggRoll,
                rank_spec=cfg["rank_spec"],
                label=cfg["label"],
            )
            result["validation_config"] = cfg["label"]
            result["config_note"] = cfg["note"]

            save_result(result, cfg["label"])
            all_results.append(result)

    return all_results


def summarize(results):
    grouped = defaultdict(list)
    for r in results:
        grouped[r["method"]].append(r)

    summary = {}
    for method, runs in grouped.items():
        accs = [r["best_test_acc"] for r in runs]
        walls = [
            r.get("converged_at_wall_s") or r.get("total_wall_s") or 0
            for r in runs
        ]
        gens = [
            r.get("converged_at_step") or 0
            for r in runs
        ]
        summary[method] = {
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "wall_mean": float(np.mean(walls)),
            "wall_std": float(np.std(walls)),
            "gens_mean": float(np.mean(gens)),
            "n_seeds": len(runs),
        }

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    header = (
        f"{'method':<20} {'acc mean':<12} {'acc std':<12} "
        f"{'wall s':<12} {'gens':<10} {'n':<4}"
    )
    print(header)
    print("-" * 80)
    for method, v in summary.items():
        print(
            f"{method:<20} {v['acc_mean']:<12.4f} {v['acc_std']:<12.4f} "
            f"{v['wall_mean']:<12.1f} {v['gens_mean']:<10.0f} {v['n_seeds']:<4}"
        )
    return summary


if __name__ == "__main__":
    t_start = time.perf_counter()

    all_results = run_validation()
    summary = summarize(all_results)

    total_wall_min = (time.perf_counter() - t_start) / 60

    combined = {
        "summary": summary,
        "configs": [
            {
                "label": cfg["label"],
                "rank_spec": {str(k): v for k, v in cfg["rank_spec"].items()},
                "note": cfg["note"],
            }
            for cfg in CONFIGS
        ],
        "seeds": list(VALIDATION_SEEDS),
        "total_wall_minutes": total_wall_min,
    }
    with open(VALIDATION_DIR / "summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 70)
    print(f"VALIDATION COMPLETE — {total_wall_min:.1f} minutes")
    print("=" * 70)
    print(f"Summary written to: {VALIDATION_DIR / 'summary.json'}")
