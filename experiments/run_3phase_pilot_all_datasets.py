"""Run the three-phase adaptive sensitivity pilot on all four datasets.

Datasets: MNIST, Fashion-MNIST, KMNIST, EMNIST-Digits
Architecture: standard [784, 256, 256, 256, 10]
Seeds: 3 per condition
Mode: allocation (rank set {0, 1, 2, 4, 8})

Expected runtime: ~2-3 hours total.
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

import sys
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from torchvision import datasets as tv_datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adaptive_sensitivity_pilot import run_pilot_with_hyperscalees

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

N_SEEDS = 3
LAYER_SHAPES = {
    (256, 784): "input",
    (256, 256): "hidden",
    (10, 256): "output",
}


def load_dataset(name):
    """Load and return (X_train, y_train, X_test, y_test) as JAX arrays."""
    print(f"\nLoading {name}...")

    if name == "MNIST":
        train_ds = tv_datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
        test_ds = tv_datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    elif name == "Fashion-MNIST":
        train_ds = tv_datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True)
        test_ds = tv_datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True)
    elif name == "KMNIST":
        train_ds = tv_datasets.KMNIST(root=str(DATA_DIR), train=True, download=True)
        test_ds = tv_datasets.KMNIST(root=str(DATA_DIR), train=False, download=True)
    elif name == "EMNIST-Digits":
        train_ds = tv_datasets.EMNIST(root=str(DATA_DIR), split="digits", train=True, download=True)
        test_ds = tv_datasets.EMNIST(root=str(DATA_DIR), split="digits", train=False, download=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")

    X_train = jnp.asarray(np.array(train_ds.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train_ds.targets, dtype=np.int32))
    X_test = jnp.asarray(np.array(test_ds.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test = jnp.asarray(np.array(test_ds.targets, dtype=np.int32))

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


def main():
    datasets = ["MNIST", "Fashion-MNIST", "KMNIST", "EMNIST-Digits"]

    print("=" * 60)
    print("THREE-PHASE ADAPTIVE SENSITIVITY PILOT — ALL DATASETS")
    print(f"Seeds: {N_SEEDS}")
    print(f"Datasets: {', '.join(datasets)}")
    print("=" * 60)

    all_results = {}
    t_total = time.time()

    for ds_name in datasets:
        print(f"\n{'#' * 60}")
        print(f"# DATASET: {ds_name}")
        print(f"{'#' * 60}")

        X_train, y_train, X_test, y_test = load_dataset(ds_name)

        output_dir = f"results/pilot_3phase/{ds_name.lower().replace('-', '_')}"

        t0 = time.time()
        result = run_pilot_with_hyperscalees(
            layer_shapes=LAYER_SHAPES,
            dataset=(X_train, y_train, X_test, y_test),
            max_rank=8,
            baseline_rank=4,
            n_seeds=N_SEEDS,
            output_dir=output_dir,
        )
        elapsed = time.time() - t0

        all_results[ds_name] = {
            "allocation": result.rank_allocation_named,
            "ordering": result.sensitivity_ordering,
            "phase3": result.phase3_results,
            "wall_seconds": elapsed,
        }

        print(f"\n{ds_name} complete in {elapsed:.0f}s")
        print(f"  Allocation: {result.rank_allocation_named}")
        print(f"  Ordering: {' >> '.join(result.sensitivity_ordering)}")
        if result.phase3_results:
            p3 = result.phase3_results
            print(f"  Phase 3: {p3['layer_name']} → rank {p3['assigned_rank']} "
                  f"(r0={p3['rank0_mean']:.4f} vs r1={p3['rank1_mean']:.4f})")

    # ── Cross-dataset summary ──
    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print("CROSS-DATASET SUMMARY")
    print(f"{'=' * 60}")
    print(f"\n{'Dataset':<18} {'Ordering':<30} {'Allocation':<25} {'P3 Decision'}")
    print("-" * 90)

    orderings_match = True
    allocations_match = True
    first_ordering = None
    first_allocation = None

    for ds_name, r in all_results.items():
        ordering_str = " > ".join(r["ordering"])
        alloc_str = str(r["allocation"])
        p3_str = f"rank {r['phase3']['assigned_rank']}" if r["phase3"] else "N/A"
        print(f"{ds_name:<18} {ordering_str:<30} {alloc_str:<25} {p3_str}")

        if first_ordering is None:
            first_ordering = r["ordering"]
            first_allocation = r["allocation"]
        else:
            if r["ordering"] != first_ordering:
                orderings_match = False
            if r["allocation"] != first_allocation:
                allocations_match = False

    print()
    if orderings_match:
        print("✓ Sensitivity ordering is CONSISTENT across all datasets.")
    else:
        print("✗ Sensitivity ordering DIFFERS across datasets. Investigate.")

    if allocations_match:
        print("✓ Rank allocation is IDENTICAL across all datasets.")
    else:
        print("✗ Rank allocation DIFFERS across datasets. Investigate.")

    print(f"\nTotal wall clock: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
