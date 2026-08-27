#!/usr/bin/env python3
"""
correctness_test.py

Definitive seed-matched comparison between vanilla EGGROLL at r=4 and
LWR-EGGROLL with uniform rank_spec=4 (integer). Both use the shared
_train_hyperscalees_common loop, differing only in noiser_class.

If the two are bit-identical, every seed will produce the exact same
final accuracy. If there's a JAX trace divergence, accuracies will
differ by a small amount (< 1pp) due to accumulated floating-point
rounding differences across thousands of generations.

Usage (from ~/dissertation/eggroll-diss):
    python experiments/correctness_test.py

Results saved to results/correctness_test/ as JSON.
Runtime: ~10-15 minutes on RTX 5060 (3 seeds × 2 methods × ~2 min each).
"""

import sys
import json
import time
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_4_methods_mnist import (
    CFG, load_mnist, _train_hyperscalees_common
)
import hyperscalees as hs
from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.noiser.lwr_eggroll import LWREggRoll

# ── Config ──────────────────────────────────────────────────────────────
SEEDS = [0, 1, 2]
OUTPUT_DIR = Path(REPO_ROOT) / "results" / "correctness_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Standard architecture shapes in HyperscaleES convention (out, in)
INPUT_SHAPE  = (256, 784)
HIDDEN_SHAPE = (256, 256)
OUTPUT_SHAPE = (10, 256)

# Uniform rank 4 as a dict — this is what LWR receives when testing
# per-shape dispatch with uniform values
UNIFORM_RANK_DICT = {
    INPUT_SHAPE:  4,
    HIDDEN_SHAPE: 4,
    OUTPUT_SHAPE: 4,
}


def run_vanilla(seed, X_train, y_train, X_test, y_test):
    """Run vanilla EGGROLL at scalar rank=4."""
    return _train_hyperscalees_common(
        seed=seed,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        noiser_class=EggRoll,
        rank_spec=4,
        label=f"vanilla_r4_seed{seed}",
    )


def run_lwr_int(seed, X_train, y_train, X_test, y_test):
    """Run LWR-EGGROLL with rank_spec=4 (integer, backward-compatible)."""
    return _train_hyperscalees_common(
        seed=seed,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        noiser_class=LWREggRoll,
        rank_spec=4,
        label=f"lwr_int4_seed{seed}",
    )


def run_lwr_dict(seed, X_train, y_train, X_test, y_test):
    """Run LWR-EGGROLL with rank_spec as a dict {shape: 4} for all shapes."""
    return _train_hyperscalees_common(
        seed=seed,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        noiser_class=LWREggRoll,
        rank_spec=UNIFORM_RANK_DICT,
        label=f"lwr_dict4_seed{seed}",
    )


def main():
    print("=" * 70)
    print("CORRECTNESS TEST: Vanilla EGGROLL r=4 vs LWR-EGGROLL uniform r=4")
    print("=" * 70)
    print(f"Seeds: {SEEDS}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    X_train, y_train, X_test, y_test = load_mnist()
    
    results = []

    for seed in SEEDS:
        print(f"\n{'─' * 50}")
        print(f"  Seed {seed}")
        print(f"{'─' * 50}")

        # 1. Vanilla EGGROLL
        print(f"\n  [1/3] Vanilla EGGROLL r=4, seed={seed}")
        t0 = time.time()
        r_vanilla = run_vanilla(seed, X_train, y_train, X_test, y_test)
        t_vanilla = time.time() - t0
        acc_vanilla = r_vanilla["best_test_acc"]
        print(f"        acc = {acc_vanilla:.6f}  ({t_vanilla:.1f}s)")

        # 2. LWR with int rank_spec=4
        print(f"\n  [2/3] LWR-EGGROLL rank_spec=4 (int), seed={seed}")
        t0 = time.time()
        r_lwr_int = run_lwr_int(seed, X_train, y_train, X_test, y_test)
        t_lwr_int = time.time() - t0
        acc_lwr_int = r_lwr_int["best_test_acc"]
        print(f"        acc = {acc_lwr_int:.6f}  ({t_lwr_int:.1f}s)")

        # 3. LWR with dict rank_spec={all shapes: 4}
        print(f"\n  [3/3] LWR-EGGROLL rank_spec=dict(all=4), seed={seed}")
        t0 = time.time()
        r_lwr_dict = run_lwr_dict(seed, X_train, y_train, X_test, y_test)
        t_lwr_dict = time.time() - t0
        acc_lwr_dict = r_lwr_dict["best_test_acc"]
        print(f"        acc = {acc_lwr_dict:.6f}  ({t_lwr_dict:.1f}s)")

        # Compute deltas
        delta_int  = abs(acc_vanilla - acc_lwr_int)
        delta_dict = abs(acc_vanilla - acc_lwr_dict)
        delta_int_dict = abs(acc_lwr_int - acc_lwr_dict)

        seed_result = {
            "seed": seed,
            "vanilla_r4": acc_vanilla,
            "lwr_int4": acc_lwr_int,
            "lwr_dict4": acc_lwr_dict,
            "delta_vanilla_vs_lwr_int": delta_int,
            "delta_vanilla_vs_lwr_dict": delta_dict,
            "delta_lwr_int_vs_lwr_dict": delta_int_dict,
            "bit_identical_int": acc_vanilla == acc_lwr_int,
            "bit_identical_dict": acc_vanilla == acc_lwr_dict,
            "wall_s_vanilla": t_vanilla,
            "wall_s_lwr_int": t_lwr_int,
            "wall_s_lwr_dict": t_lwr_dict,
        }
        results.append(seed_result)

        print(f"\n  Δ(vanilla vs lwr_int):  {delta_int:.6f}  "
              f"{'✓ BIT-IDENTICAL' if seed_result['bit_identical_int'] else '✗ differs'}")
        print(f"  Δ(vanilla vs lwr_dict): {delta_dict:.6f}  "
              f"{'✓ BIT-IDENTICAL' if seed_result['bit_identical_dict'] else '✗ differs'}")
        print(f"  Δ(lwr_int vs lwr_dict): {delta_int_dict:.6f}  "
              f"{'✓ BIT-IDENTICAL' if seed_result['bit_identical_dict'] else '✗ differs'}")

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_bit_identical_int = all(r["bit_identical_int"] for r in results)
    all_bit_identical_dict = all(r["bit_identical_dict"] for r in results)
    max_delta_int = max(r["delta_vanilla_vs_lwr_int"] for r in results)
    max_delta_dict = max(r["delta_vanilla_vs_lwr_dict"] for r in results)

    print(f"\n  Vanilla vs LWR(int):   "
          f"{'ALL BIT-IDENTICAL' if all_bit_identical_int else f'max delta = {max_delta_int:.6f}'}")
    print(f"  Vanilla vs LWR(dict):  "
          f"{'ALL BIT-IDENTICAL' if all_bit_identical_dict else f'max delta = {max_delta_dict:.6f}'}")

    print("\n  Per-seed breakdown:")
    print(f"  {'Seed':>4}  {'Vanilla':>10}  {'LWR(int)':>10}  {'LWR(dict)':>10}  "
          f"{'Δ int':>8}  {'Δ dict':>8}  {'Match?':>10}")
    for r in results:
        match = "✓ yes" if r["bit_identical_int"] and r["bit_identical_dict"] else "✗ no"
        print(f"  {r['seed']:>4}  {r['vanilla_r4']:>10.6f}  {r['lwr_int4']:>10.6f}  "
              f"{r['lwr_dict4']:>10.6f}  {r['delta_vanilla_vs_lwr_int']:>8.6f}  "
              f"{r['delta_vanilla_vs_lwr_dict']:>8.6f}  {match:>10}")

    # ── Dissertation claim recommendation ───────────────────────────────
    print("\n" + "─" * 70)
    if all_bit_identical_int and all_bit_identical_dict:
        print("  CLAIM: 'LWR-EGGROLL with uniform rank specification produces")
        print("  bit-identical results to vanilla EGGROLL at matched seed.'")
        print("  → Safe to write 'zero implementation artefact' in the dissertation.")
    elif max_delta_int < 0.005 and max_delta_dict < 0.005:
        print("  CLAIM: 'LWR-EGGROLL with uniform rank specification produces")
        print("  results statistically indistinguishable from vanilla EGGROLL")
        print(f"  (max accuracy delta {max(max_delta_int, max_delta_dict):.4f}, within")
        print("  cross-seed variance), consistent with JAX compilation-path")
        print("  rounding differences rather than algorithmic divergence.'")
        print("  → Write 'near-identical' not 'bit-identical' in the dissertation.")
    else:
        print(f"  WARNING: max delta = {max(max_delta_int, max_delta_dict):.4f}")
        print("  This exceeds expected JAX trace noise. Investigate before claiming")
        print("  equivalence in the dissertation.")
    print("─" * 70)

    # ── Save ────────────────────────────────────────────────────────────
    summary = {
        "test": "correctness_comparison",
        "description": "Seed-matched vanilla EGGROLL r=4 vs LWR-EGGROLL uniform r=4",
        "architecture": [784, 256, 256, 256, 10],
        "seeds": SEEDS,
        "all_bit_identical_int": all_bit_identical_int,
        "all_bit_identical_dict": all_bit_identical_dict,
        "max_delta_int": max_delta_int,
        "max_delta_dict": max_delta_dict,
        "per_seed": results,
    }

    out_path = OUTPUT_DIR / "correctness_test.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
