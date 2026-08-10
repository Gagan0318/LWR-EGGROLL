"""Rerun Phase 1 ONLY on all four MNIST datasets.
Purpose: capture fitness_mean_history alongside fitness_variance_history.
Uses the shared-checkpoint approach (100 gen pretrain → single-gen eval).
Runtime: ~2-3 minutes total (Phase 1 is the cheapest phase).
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
from adaptive_sensitivity_pilot import AdaptiveSensitivityPilot
from compare_4_methods_mnist import train_lwr_eggroll

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path("results/phase1_mean_fitness")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 3
LAYER_SHAPES = {
    (256, 784): "input",
    (256, 256): "hidden",
    (10, 256): "output",
}

def load_dataset(name):
    print(f"\nLoading {name}...")
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
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def make_train_fn(X_train, y_train, X_test, y_test):
    def adapted_train_fn(seed, rank_spec, label, X_train=None, y_train=None, X_test=None, y_test=None,
                         max_gens=None, return_checkpoint=False,
                         initial_params=None):
        return train_lwr_eggroll(
            seed, X_train, y_train, X_test, y_test,
            rank_spec=rank_spec, label=label,
            max_gens=max_gens, return_checkpoint=return_checkpoint,
            initial_params=initial_params,
        )
    return adapted_train_fn

DATASETS = ["MNIST", "Fashion-MNIST", "KMNIST", "EMNIST-Digits"]

def main():
    print("=" * 60)
    print("PHASE 1 RERUN — Mean Fitness Capture")
    print(f"Datasets: {len(DATASETS)}")
    print(f"Seeds: {N_SEEDS}")
    print("=" * 60, flush=True)

    all_results = {}
    t_total = time.time()

    for ds_name in DATASETS:
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}", flush=True)

        X_train, y_train, X_test, y_test = load_dataset(ds_name)
        train_fn = make_train_fn(X_train, y_train, X_test, y_test)

        pilot = AdaptiveSensitivityPilot(
            train_fn=train_fn,
            layer_shapes=LAYER_SHAPES,
            dataset=(X_train, y_train, X_test, y_test),
            max_rank=8,
            baseline_rank=4,
            output_dir=str(RESULTS_DIR),
        )





    

        seeds = list(range(N_SEEDS))
        t0 = time.time()
        phase1_results = pilot._run_phase1(seeds)
        elapsed = time.time() - t0

        ds_data = {"dataset": ds_name, "wall_seconds": elapsed, "layers": []}
        print(f"\n  Phase 1 results for {ds_name} ({elapsed:.1f}s):")
        for r in phase1_results:
            print(f"    {r.layer_name:8s} {r.shape}  "
                  f"variance = {r.mean:.6f} ± {r.std:.6f}")
            ds_data["layers"].append({
                "name": r.layer_name,
                "shape": list(r.shape),
                "metric": r.metric_name,
                "values": r.values_per_seed,
                "mean": r.mean,
                "std": r.std,
            })

        # Ordering from Phase 1
        sorted_layers = sorted(phase1_results, key=lambda x: x.mean, reverse=True)
        ordering = " > ".join(r.layer_name for r in sorted_layers)
        ds_data["ordering"] = ordering
        print(f"  Phase 1 ordering: {ordering}")

        all_results[ds_name] = ds_data

    total_time = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"PHASE 1 RERUN COMPLETE — {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}")

    # Summary
    for ds, data in all_results.items():
        print(f"  {ds:20s} {data['ordering']}")

    with open(RESULTS_DIR / "phase1_mean_fitness_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)

if __name__ == "__main__":
    main()
