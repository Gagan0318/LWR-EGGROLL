"""
Multi-Seed Phase 3 Validation — MNIST
======================================
Tests whether the Phase 3 binary inclusion recommendation (rank 0 vs rank 1
for the least sensitive layer) is stable across random initialisations.

Hypothesis: When the frozen parameter fraction is low (~1% for output layer
on MNIST), the recommendation should be consistent. The Brax Ant data
(93% frozen, best_so_far range 5.1–50.0) suggests instability at high
frozen fractions — this experiment provides the low-fraction control.

Runs: 5 seeds × 2 conditions (rank 0 vs rank 1 on output) = 10 training runs
Architecture: [784, 256, 256, 256, 10]
Non-target layers: rank 4 (baseline)
Target layer: output (10, 256) — least sensitive per pilot

Expected runtime: ~30–60 min on CPU (WSL, no GPU needed)
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.30"

import sys
import json
import time
from pathlib import Path

# --- Path setup ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "experiments" else SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from compare_4_methods_mnist import (
    CFG, load_mnist, _train_hyperscalees_common,
)
from hyperscalees.noiser.lwr_eggroll import LWREggRoll

# --- Configuration ---
SEEDS = [0, 1, 2, 3, 4]
PHASE3_GENS = 50          # enough to see learning signal, not full convergence
POP = 256
SIGMA = 0.03
LR = 0.01

# Layer shapes (transposed: out_dim, in_dim)
INPUT_SHAPE  = (256, 784)
HIDDEN_SHAPE = (256, 256)
OUTPUT_SHAPE = (10, 256)

# Phase 3 conditions: rank 0 vs rank 1 on output, others at rank 4
SPEC_RANK0 = {INPUT_SHAPE: 4, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 0}
SPEC_RANK1 = {INPUT_SHAPE: 4, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 1}

# Results directory
RESULTS_DIR = PROJECT_ROOT / "results" / "phase3_multiseed"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Frozen parameter fractions
TOTAL_PARAMS = 784*256 + 256 + 256*256 + 256 + 256*256 + 256 + 256*10 + 10  # ~270K
OUTPUT_PARAMS = 256*10 + 10  # 2,570
FROZEN_FRACTION = OUTPUT_PARAMS / TOTAL_PARAMS


def run_condition(seed, rank_spec, label, X_train, y_train, X_test, y_test):
    """Run a single Phase 3 condition and return result."""
    result_file = RESULTS_DIR / f"{label}_seed{seed}.json"
    
    # Skip if already done
    if result_file.exists():
        with open(result_file) as f:
            r = json.load(f)
        print(f"  [SKIP] {label} seed={seed} — "
              f"best_acc={max(h[2] for h in r['history']):.4f}")
        return r
    
    print(f"  [RUN]  {label} seed={seed}...", end="", flush=True)
    t0 = time.time()
    
    result = _train_hyperscalees_common(
        seed, X_train, y_train, X_test, y_test,
        LWREggRoll, rank_spec, label
    )
    
    elapsed = time.time() - t0
    best_acc = max(h[2] for h in result["history"])
    
    # Save to disk immediately
    with open(result_file, "w") as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f" done in {elapsed:.0f}s, best_acc={best_acc:.4f}")
    return result


def main():
    print("=" * 60)
    print("MULTI-SEED PHASE 3 VALIDATION — MNIST")
    print(f"Architecture: [784, 256, 256, 256, 10]")
    print(f"Target layer: output {OUTPUT_SHAPE}")
    print(f"Frozen fraction (rank 0 on output): {FROZEN_FRACTION:.3%}")
    print(f"Seeds: {SEEDS}, Gens: {PHASE3_GENS}, POP: {POP}")
    print(f"Conditions: rank_0_output vs rank_1_output")
    print("=" * 60)
    
    # Configure training parameters
    CFG.eggroll_pop = POP
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_lr = LR
    CFG.eggroll_max_gens = PHASE3_GENS
    CFG.hidden_dims = [256, 256, 256]
    CFG.n_classes = 10
    # Set a generous wall-clock limit so gens is the binding constraint
    CFG.max_wall_seconds = 600
    
    # Load data once
    print("\nLoading MNIST...", flush=True)
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Run all conditions
    results_r0 = {}
    results_r1 = {}
    
    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        
        r0 = run_condition(seed, SPEC_RANK0, "rank_0_output",
                          X_train, y_train, X_test, y_test)
        results_r0[seed] = r0
        
        r1 = run_condition(seed, SPEC_RANK1, "rank_1_output",
                          X_train, y_train, X_test, y_test)
        results_r1[seed] = r1
    
    # --- Analysis ---
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    accs_r0 = []
    accs_r1 = []
    recommendations = []
    
    print(f"\n{'Seed':<6} {'Rank 0':<12} {'Rank 1':<12} {'Diff':<10} {'Recommends'}")
    print("-" * 52)
    
    for seed in SEEDS:
        best_r0 = max(h[2] for h in results_r0[seed]["history"])
        best_r1 = max(h[2] for h in results_r1[seed]["history"])
        diff = best_r0 - best_r1
        rec = "rank 0" if best_r0 >= best_r1 else "rank 1"
        
        accs_r0.append(best_r0)
        accs_r1.append(best_r1)
        recommendations.append(rec)
        
        print(f"{seed:<6} {best_r0:<12.4f} {best_r1:<12.4f} {diff:+.4f}    {rec}")
    
    import numpy as np
    mean_r0 = np.mean(accs_r0)
    std_r0 = np.std(accs_r0)
    mean_r1 = np.mean(accs_r1)
    std_r1 = np.std(accs_r1)
    
    rank0_count = recommendations.count("rank 0")
    consistency = rank0_count / len(SEEDS)
    
    print(f"\n{'Mean':<6} {mean_r0:<12.4f} {mean_r1:<12.4f} {mean_r0 - mean_r1:+.4f}")
    print(f"{'Std':<6} {std_r0:<12.4f} {std_r1:<12.4f}")
    
    print(f"\nRecommendation: rank 0 in {rank0_count}/{len(SEEDS)} seeds "
          f"({consistency:.0%} consistency)")
    print(f"Frozen parameter fraction: {FROZEN_FRACTION:.3%}")
    
    if consistency == 1.0:
        verdict = ("STABLE — rank 0 recommendation holds across all seeds. "
                   "At {:.1%} frozen fraction, the initialisation lottery "
                   "does not affect the Phase 3 outcome.".format(FROZEN_FRACTION))
    elif consistency >= 0.6:
        verdict = ("MOSTLY STABLE — rank 0 wins in {}/{} seeds. "
                   "Some seed sensitivity, but the recommendation "
                   "is directionally consistent.".format(rank0_count, len(SEEDS)))
    else:
        verdict = ("UNSTABLE — recommendation varies across seeds. "
                   "Multi-seed Phase 3 is required at this frozen fraction.")
    
    print(f"\nVerdict: {verdict}")
    
    # Save summary
    summary = {
        "experiment": "phase3_multiseed_validation",
        "architecture": [784, 256, 256, 256, 10],
        "target_layer": "output",
        "target_shape": list(OUTPUT_SHAPE),
        "frozen_fraction": FROZEN_FRACTION,
        "seeds": SEEDS,
        "gens": PHASE3_GENS,
        "pop": POP,
        "sigma": SIGMA,
        "per_seed": {
            str(s): {
                "rank_0_acc": accs_r0[i],
                "rank_1_acc": accs_r1[i],
                "diff": accs_r0[i] - accs_r1[i],
                "recommends": recommendations[i]
            }
            for i, s in enumerate(SEEDS)
        },
        "summary": {
            "rank_0_mean": mean_r0, "rank_0_std": std_r0,
            "rank_1_mean": mean_r1, "rank_1_std": std_r1,
            "consistency": consistency,
            "verdict": verdict
        }
    }
    
    with open(RESULTS_DIR / "phase3_multiseed_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {RESULTS_DIR}/")
    
    # --- Comparison with Brax Ant ---
    print("\n" + "=" * 60)
    print("COMPARISON: MNIST vs BRAX ANT")
    print("=" * 60)
    print(f"{'':20} {'MNIST (output)':<20} {'Brax Ant (hidden)'}")
    print(f"{'Frozen fraction':<20} {FROZEN_FRACTION:<20.1%} {'93.6%'}")
    print(f"{'Phase 3 consistency':<20} {consistency:<20.0%} {'N/A (inferred from'}")
    print(f"{'':20} {'':20} {'cross-seed variance)'}")
    print(f"{'Cross-seed range':<20} {max(accs_r0)-min(accs_r0):<20.4f} {'5.1 — 50.0 (best_so_far)'}")
    print(f"\nConclusion: Phase 3 is {'stable' if consistency >= 0.8 else 'unstable'} "
          f"at {FROZEN_FRACTION:.1%} frozen fraction (MNIST output layer).")
    print("Brax Ant data (93.6% frozen) shows extreme seed dependency,")
    print("motivating multi-seed Phase 3 when frozen fraction exceeds a threshold.")


if __name__ == "__main__":
    main()
