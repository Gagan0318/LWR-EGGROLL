"""Phase 1-Informed Rank Allocation Experiment
Derives per-dataset LWR allocations using Phase 1 × Phase 2 interaction:
  - Phase 2 strongly positive → rank 8 (input)
  - Phase 2 negative → rank 0 (output)
  - Phase 2 moderate + Phase 1 high variance → rank 4 (elevated)
  - Phase 2 moderate + Phase 1 low variance → rank 2 (conservative)

Compares against Phase 2-only allocation (8, 4, 0) on all four datasets.
Seeds: 3. Architecture: [784, 256, 256, 256, 10].
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

import sys
import time
import json
from pathlib import Path
import jax.numpy as jnp
import numpy as np
from torchvision import datasets as tv_datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_4_methods_mnist import train_lwr_eggroll, CFG

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results/phase1_informed")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 3
SEEDS = list(range(N_SEEDS))

LAYER_SHAPES = {
    "input": (256, 784),
    "hidden": (256, 256),
    "output": (10, 256),
}

# Phase 2 results (consistent across all datasets from main pilot):
# input: strongly positive degradation → always rank 8
# hidden: moderately positive degradation → rank depends on Phase 1
# output: negative degradation → always rank 0
PHASE2_TIERS = {
    "input": "strongly_positive",
    "hidden": "moderate",
    "output": "negative",
}

# Phase 2-only allocation (baseline comparison, cap at rank 4)
PHASE2_ONLY = {(256, 784): 4, (256, 256): 2, (10, 256): 0}
PHASE2_ONLY_LABEL = "lwr_4_2_0_phase2only"

DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST", "EMNIST-Digits"]

def load_dataset(name):
    print(f"\n  Loading {name}...")
    if name == "MNIST":
        tr = tv_datasets.MNIST(DATA_DIR, train=True, download=True)
        te = tv_datasets.MNIST(DATA_DIR, train=False, download=True)
    elif name == "Fashion-MNIST":
        tr = tv_datasets.FashionMNIST(DATA_DIR, train=True, download=True)
        te = tv_datasets.FashionMNIST(DATA_DIR, train=False, download=True)
    elif name == "KMNIST":
        tr = tv_datasets.KMNIST(DATA_DIR, train=True, download=True)
        te = tv_datasets.KMNIST(DATA_DIR, train=False, download=True)
    elif name == "EMNIST-Digits":
        tr = tv_datasets.EMNIST(DATA_DIR, split="digits", train=True, download=True)
        te = tv_datasets.EMNIST(DATA_DIR, split="digits", train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    X_train = jnp.array(tr.data.numpy().reshape(-1, 784).astype("float32") / 255.0)
    y_train = jnp.array(tr.targets.numpy().astype("int32"))
    X_test = jnp.array(te.data.numpy().reshape(-1, 784).astype("float32") / 255.0)
    y_test = jnp.array(te.targets.numpy().astype("int32"))
    return X_train, y_train, X_test, y_test


def derive_phase1_informed_allocation(phase1_layers):
    """Derive rank allocation using Phase 1 × Phase 2 interaction.

    For layers in the 'moderate' Phase 2 tier (hidden), Phase 1 variance
    determines whether to assign rank 4 (high variance → useful spread)
    or rank 2 (low variance → modest spread, less rank needed).

    Args:
        phase1_layers: list of dicts with 'name' and 'mean' (variance)

    Returns:
        rank_spec dict keyed by shape tuple, and a human-readable label
    """
    # Get Phase 1 ranking (by variance, descending)
    sorted_by_var = sorted(phase1_layers, key=lambda x: x["mean"], reverse=True)
    p1_rank = {l["name"]: i for i, l in enumerate(sorted_by_var)}
    # 0 = highest variance, 1 = middle, 2 = lowest

    alloc = {}
    for layer_name, shape in LAYER_SHAPES.items():
        p2_tier = PHASE2_TIERS[layer_name]

        if p2_tier == "strongly_positive":
            alloc[shape] = 4
        elif p2_tier == "negative":
            alloc[shape] = 0
        elif p2_tier == "moderate":
            # Phase 1 breaks the tie using three tiers
            if p1_rank[layer_name] == 0:
                # Highest Phase 1 variance → large useful spread → rank 4
                alloc[shape] = 4
            elif p1_rank[layer_name] == 1:
                # Middle Phase 1 variance → moderate spread → rank 2
                alloc[shape] = 2
            else:
                # Lowest Phase 1 variance → minimal spread → rank 1
                alloc[shape] = 1

    label = f"lwr_{alloc[LAYER_SHAPES['input']]}_{alloc[LAYER_SHAPES['hidden']]}_{alloc[LAYER_SHAPES['output']]}"
    return alloc, label


def run_condition(X_train, y_train, X_test, y_test, rank_spec, label, seeds):
    """Run training across seeds, return list of result dicts."""
    results = []
    for seed in seeds:
        print(f"    seed={seed} ...", end="", flush=True)
        t0 = time.time()
        r = train_lwr_eggroll(
            seed, X_train, y_train, X_test, y_test,
            rank_spec=rank_spec, label=f"{label}_s{seed}",
        )
        elapsed = time.time() - t0
        print(f" acc={r['best_test_acc']:.4f}  ({elapsed:.1f}s)")
        results.append(r)
    accs = [r["best_test_acc"] for r in results]
    return results, float(np.mean(accs)), float(np.std(accs))


def main():
    # Load Phase 1 results
    phase1_path = Path("results/phase1_mean_fitness/phase1_mean_fitness_results.json")
    if not phase1_path.exists():
        print(f"ERROR: Phase 1 results not found at {phase1_path}")
        print("Run rerun_phase1_mean_fitness.py first.")
        sys.exit(1)

    with open(phase1_path) as f:
        phase1_data = json.load(f)

    print("=" * 70)
    print("PHASE 1-INFORMED RANK ALLOCATION EXPERIMENT")
    print("Comparing Phase 2-only vs Phase 1×Phase 2 informed allocations")
    print("=" * 70)

    # Derive per-dataset allocations
    print("\n--- Derived Allocations ---")
    dataset_allocations = {}
    for ds in DATASETS:
        if ds not in phase1_data:
            print(f"  {ds}: no Phase 1 data, skipping")
            continue
        layers = phase1_data[ds]["layers"]
        alloc, label = derive_phase1_informed_allocation(layers)
        dataset_allocations[ds] = (alloc, label)

        # Show derivation
        sorted_by_var = sorted(layers, key=lambda x: x["mean"], reverse=True)
        p1_ordering = " > ".join(l["name"] for l in sorted_by_var)
        print(f"  {ds:20s}  P1: {p1_ordering:30s}  → {label}")

    print(f"\n  Phase 2-only baseline: {PHASE2_ONLY_LABEL} (4, 2, 0) for all datasets")

    # Run experiments
    all_results = {}
    t_total = time.time()

    for ds in DATASETS:
        if ds not in dataset_allocations:
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds}")
        print(f"{'='*60}", flush=True)

        X_train, y_train, X_test, y_test = load_dataset(ds)
        informed_alloc, informed_label = dataset_allocations[ds]

        ds_results = {}

        # Condition 1: Phase 2-only allocation (4, 2, 0)
        print(f"\n  --- {PHASE2_ONLY_LABEL} (budget={sum(PHASE2_ONLY.values())}) ---")
        _, p2_mean, p2_std = run_condition(
            X_train, y_train, X_test, y_test,
            PHASE2_ONLY, PHASE2_ONLY_LABEL, SEEDS,
        )
        ds_results[PHASE2_ONLY_LABEL] = {
            "allocation": {k: f"{v}" for k, v in zip(["input","hidden","output"], [4, 2, 0])},
            "rank_budget": sum(PHASE2_ONLY.values()),
            "mean_acc": p2_mean,
            "std_acc": p2_std,
        }

        # Condition 2: Phase 1-informed allocation
        informed_budget = sum(informed_alloc.values())
        print(f"\n  --- {informed_label} (budget={informed_budget}) ---")
        _, p1_mean, p1_std = run_condition(
            X_train, y_train, X_test, y_test,
            informed_alloc, informed_label, SEEDS,
        )
        ds_results[informed_label] = {
            "allocation": {name: str(informed_alloc[shape]) for name, shape in LAYER_SHAPES.items()},
            "rank_budget": informed_budget,
            "mean_acc": p1_mean,
            "std_acc": p1_std,
        }

        # Compute difference
        diff = p1_mean - p2_mean
        winner = informed_label if diff > 0 else PHASE2_ONLY_LABEL
        ds_results["comparison"] = {
            "phase1_informed_advantage": diff,
            "winner": winner,
        }

        print(f"\n  {ds} RESULT:")
        print(f"    Phase 2-only (8,4,0):     {p2_mean:.4f} ± {p2_std:.4f}  (budget={sum(PHASE2_ONLY.values())})")
        print(f"    Phase 1-informed ({informed_label}): {p1_mean:.4f} ± {p1_std:.4f}  (budget={informed_budget})")
        print(f"    Difference: {diff:+.4f}  →  {winner}")

        all_results[ds] = ds_results

    total_time = time.time() - t_total

    # Summary
    print(f"\n{'='*70}")
    print("PHASE 1-INFORMED ALLOCATION SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<20} {'P2-only (4,2,0)':>18} {'P1-informed':>18} {'Diff':>10} {'Winner':>20}")
    print("-" * 88)

    wins_p1 = 0
    wins_p2 = 0
    for ds, data in all_results.items():
        p2_acc = data[PHASE2_ONLY_LABEL]["mean_acc"]
        informed_key = [k for k in data if k.startswith("lwr_") and k != PHASE2_ONLY_LABEL][0]
        p1_acc = data[informed_key]["mean_acc"]
        diff = data["comparison"]["phase1_informed_advantage"]
        winner = data["comparison"]["winner"]
        if diff > 0:
            wins_p1 += 1
        else:
            wins_p2 += 1
        print(f"{ds:<20} {p2_acc:>17.4f} {p1_acc:>17.4f} {diff:>+10.4f} {winner:>20}")

    print(f"\nPhase 1-informed wins: {wins_p1}/{len(all_results)}")
    print(f"Phase 2-only wins:    {wins_p2}/{len(all_results)}")
    print(f"Total wall clock: {total_time:.0f}s ({total_time/60:.1f} min)")

    # Determine overall conclusion
    print(f"\n--- CONCLUSION ---")
    if wins_p1 > wins_p2:
        print("Phase 1 variance provides actionable signal for moderate-tier layers.")
        print("Using Phase 1 to break ties in the moderate Phase 2 tier improves allocation.")
    elif wins_p1 == wins_p2:
        print("Mixed results — Phase 1 tie-breaking helps on some datasets but not others.")
        print("The practical value of Phase 1 for allocation is dataset-dependent.")
    else:
        print("Phase 2-only allocation is sufficient.")
        print("Phase 1 variance does not improve allocation for moderate-tier layers,")
        print("but retains value as a mechanistic diagnostic (magnitude vs direction insight).")

    # Save
    with open(RESULTS_DIR / "phase1_informed_results.json", "w") as f:
        json.dump({
            "description": "Phase 1-informed vs Phase 2-only rank allocation comparison",
            "phase2_only_allocation": {"input": 8, "hidden": 4, "output": 0},
            "results": all_results,
            "total_wall_seconds": total_time,
        }, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
