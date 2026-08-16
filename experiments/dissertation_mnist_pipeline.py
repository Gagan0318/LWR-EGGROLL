"""
dissertation_mnist_pipeline.py

End-to-end pipeline for the MNIST dissertation experiments:
  1. Run the three-phase sensitivity pilot → produces allocation
  2. Run LWR-EGGROLL with the pilot-derived allocation
  3. Run vanilla EGGROLL r=4 as baseline
  4. Run reversed allocation as control
  5. Save all results with the pilot allocation in metadata

This script ensures the experiment uses the pilot's output directly,
not a hardcoded allocation. The sensitivity pilot is the mechanism
that produces the allocation; the experiments consume it.

Usage:
    cd ~/dissertation/eggroll-diss
    python experiments/dissertation_mnist_pipeline.py
"""

import sys
import json
import time
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_4_methods_mnist import (
    CFG, load_mnist, train_eggroll, train_lwr_eggroll, save_result,
)
from adaptive_sensitivity_pilot import AdaptiveSensitivityPilot

# ── Config ──────────────────────────────────────────────────────
DATASETS = ["mnist"]  # extend to ["mnist", "fashion_mnist", "kmnist", "emnist_digits"]
SEEDS = [0, 1, 2]
N_PILOT_SEEDS = 5
OUTPUT_DIR = Path(REPO_ROOT) / "results" / "dissertation_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Architecture shapes in HyperscaleES convention
INPUT_SHAPE  = (256, 784)
HIDDEN_SHAPE = (256, 256)
OUTPUT_SHAPE = (10, 256)

LAYER_SHAPES = [INPUT_SHAPE, HIDDEN_SHAPE, OUTPUT_SHAPE]
LAYER_NAMES  = ["input", "hidden", "output"]


def run_pipeline(dataset_name="mnist"):
    print("=" * 70)
    print(f"DISSERTATION PIPELINE — {dataset_name.upper()}")
    print("=" * 70)

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n[1/5] Loading data...")
    X_train, y_train, X_test, y_test = load_mnist()
    # TODO: add dataset switching for Fashion-MNIST, KMNIST, EMNIST-Digits

    # ── Step 2: Run sensitivity pilot ──────────────────────────
    print("\n[2/5] Running three-phase sensitivity pilot...")
    pilot = AdaptiveSensitivityPilot(
        layer_shapes=LAYER_SHAPES,
        layer_names=LAYER_NAMES,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        max_rank=8,
        baseline_rank=4,
        output_dir=str(OUTPUT_DIR / dataset_name / "pilot"),
    )
    pilot_result = pilot.run(n_seeds=N_PILOT_SEEDS)

    # Extract the pilot-derived allocation (shape-keyed dict)
    allocation = pilot_result.allocation  # e.g. {(256,784): 8, (256,256): 4, (10,256): 0}
    allocation_named = pilot_result.allocation_named  # e.g. {"input": 8, "hidden": 4, "output": 0}
    ordering = pilot_result.sensitivity_ordering

    print(f"\n  Pilot-derived allocation: {allocation_named}")
    print(f"  Sensitivity ordering:    {' > '.join(ordering)}")
    print(f"  Total rank budget:       {sum(allocation.values())}")

    # Build reversed allocation (swap most and least sensitive ranks)
    rank_values = list(allocation_named.values())
    reversed_values = list(reversed(rank_values))
    reversed_allocation = dict(zip(
        [LAYER_SHAPES[LAYER_NAMES.index(n)] for n in ordering],
        reversed_values
    ))
    reversed_named = dict(zip(ordering, reversed_values))

    print(f"  Reversed allocation:     {reversed_named}")

    # ── Step 3: Run LWR-EGGROLL with pilot allocation ──────────
    print(f"\n[3/5] Running LWR-EGGROLL with pilot-derived allocation...")
    alloc_label = "_".join(str(allocation_named[n]) for n in LAYER_NAMES)
    lwr_results = []
    for seed in SEEDS:
        print(f"  LWR ({alloc_label}) seed={seed}...", end="", flush=True)
        r = train_lwr_eggroll(
            seed, X_train, y_train, X_test, y_test,
            rank_spec=allocation,
            label=f"lwr_pilot_{alloc_label}",
        )
        r["pilot_allocation"] = {str(k): v for k, v in allocation.items()}
        r["pilot_ordering"] = ordering
        r["allocation_source"] = "sensitivity_pilot"
        save_result(r)
        lwr_results.append(r)
        print(f" acc={r['best_test_acc']:.4f}")

    # ── Step 4: Run vanilla EGGROLL r=4 baseline ───────────────
    print(f"\n[4/5] Running vanilla EGGROLL r=4 baseline...")
    vanilla_results = []
    for seed in SEEDS:
        print(f"  Vanilla r=4 seed={seed}...", end="", flush=True)
        r = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=4)
        save_result(r)
        vanilla_results.append(r)
        print(f" acc={r['best_test_acc']:.4f}")

    # ── Step 5: Run reversed allocation control ────────────────
    print(f"\n[5/5] Running reversed allocation control...")
    rev_label = "_".join(str(reversed_named[n]) for n in LAYER_NAMES)
    reversed_results = []
    for seed in SEEDS:
        print(f"  Reversed ({rev_label}) seed={seed}...", end="", flush=True)
        r = train_lwr_eggroll(
            seed, X_train, y_train, X_test, y_test,
            rank_spec=reversed_allocation,
            label=f"lwr_reversed_{rev_label}",
        )
        r["allocation_source"] = "reversed_control"
        save_result(r)
        reversed_results.append(r)
        print(f" acc={r['best_test_acc']:.4f}")

    # ── Summary ────────────────────────────────────────────────
    import numpy as np
    lwr_accs = [r["best_test_acc"] for r in lwr_results]
    van_accs = [r["best_test_acc"] for r in vanilla_results]
    rev_accs = [r["best_test_acc"] for r in reversed_results]

    print(f"\n{'=' * 70}")
    print(f"RESULTS — {dataset_name.upper()}")
    print(f"{'=' * 70}")
    print(f"  LWR pilot ({alloc_label}):   {np.mean(lwr_accs):.4f} ± {np.std(lwr_accs):.4f}")
    print(f"  Vanilla r=4:               {np.mean(van_accs):.4f} ± {np.std(van_accs):.4f}")
    print(f"  Reversed ({rev_label}):     {np.mean(rev_accs):.4f} ± {np.std(rev_accs):.4f}")
    print(f"  LWR advantage over vanilla: +{(np.mean(lwr_accs) - np.mean(van_accs)) * 100:.2f}pp")
    print(f"  Aligned-reversed gap:       {(np.mean(lwr_accs) - np.mean(rev_accs)) * 100:.2f}pp")
    print(f"{'=' * 70}")

    # Save pipeline summary
    summary = {
        "dataset": dataset_name,
        "pilot_allocation": allocation_named,
        "pilot_ordering": ordering,
        "reversed_allocation": reversed_named,
        "lwr_mean": float(np.mean(lwr_accs)),
        "lwr_std": float(np.std(lwr_accs)),
        "vanilla_mean": float(np.mean(van_accs)),
        "vanilla_std": float(np.std(van_accs)),
        "reversed_mean": float(np.mean(rev_accs)),
        "reversed_std": float(np.std(rev_accs)),
        "advantage_pp": float((np.mean(lwr_accs) - np.mean(van_accs)) * 100),
        "aligned_reversed_gap_pp": float((np.mean(lwr_accs) - np.mean(rev_accs)) * 100),
    }
    summary_path = OUTPUT_DIR / dataset_name / "pipeline_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")


if __name__ == "__main__":
    for dataset in DATASETS:
        run_pipeline(dataset)
