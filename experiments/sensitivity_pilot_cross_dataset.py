"""Sensitivity pilot on Fashion-MNIST and KMNIST.
For each dataset, isolate one layer at a time (rank 4) while freezing
all others (rank 0). Measure fitness variance across the population.
High variance = layer is sensitive to perturbation = deserves more rank.

3 layers × 5 seeds × 2 datasets = 30 runs. ~30-45 minutes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import time
import numpy as np
import jax.numpy as jnp
from torchvision import datasets as tv_datasets
from experiments.compare_4_methods_mnist import (
    CFG,
    train_lwr_eggroll,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SEEDS = (0, 1, 2, 3, 4)
SIGMA = 0.05
POP = 2048
WALL_BUDGET = 300.0
ACTIVE_RANK = 4

INPUT_SHAPE = (256, 784)
HIDDEN_SHAPE = (256, 256)
OUTPUT_SHAPE = (10, 256)

# Each config isolates one layer
LAYER_CONFIGS = {
    "input_only": {
        INPUT_SHAPE: ACTIVE_RANK,
        HIDDEN_SHAPE: 0,
        OUTPUT_SHAPE: 0,
    },
    "hidden_only": {
        INPUT_SHAPE: 0,
        HIDDEN_SHAPE: ACTIVE_RANK,
        OUTPUT_SHAPE: 0,
    },
    "output_only": {
        INPUT_SHAPE: 0,
        HIDDEN_SHAPE: 0,
        OUTPUT_SHAPE: ACTIVE_RANK,
    },
}


def load_dataset(name):
    ds_class = {"fashion": tv_datasets.FashionMNIST, "kmnist": tv_datasets.KMNIST}[name]
    train = ds_class(root=str(DATA_DIR), train=True, download=True)
    test  = ds_class(root=str(DATA_DIR), train=False, download=True)
    X_train = np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0
    y_train = np.array(train.targets, dtype=np.int32)
    X_test  = np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0
    y_test  = np.array(test.targets, dtype=np.int32)
    X_train, y_train = jnp.asarray(X_train), jnp.asarray(y_train)
    X_test, y_test = jnp.asarray(X_test), jnp.asarray(y_test)
    print(f"[data] {name}: X_train {X_train.shape}, X_test {X_test.shape}")
    return X_train, y_train, X_test, y_test


def main():
    total = len(LAYER_CONFIGS) * len(SEEDS) * 2
    print(f"Total runs: {total}")

    for dataset_name in ("fashion", "kmnist"):
        print(f"\n{'='*60}")
        print(f"  SENSITIVITY PILOT: {dataset_name.upper()}")
        print(f"{'='*60}")

        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        results_dir = Path(f"results/sensitivity_pilot/{dataset_name}")

        CFG.eggroll_sigma_init = SIGMA
        CFG.eggroll_pop = POP
        CFG.max_wall_seconds = WALL_BUDGET

        for layer_name, rank_spec in LAYER_CONFIGS.items():
            for seed in SEEDS:
                outdir = results_dir / layer_name
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"seed{seed}.json"
                if fname.exists():
                    print(f"[skip] {fname}")
                    continue

                print(f"\n[{dataset_name}/{layer_name} seed={seed}]", flush=True)
                t0 = time.time()
                result = train_lwr_eggroll(
                    seed, X_train, y_train, X_test, y_test,
                    rank_spec=rank_spec, label=layer_name,
                )
                wall = time.time() - t0

                out = dict(result) if isinstance(result, dict) else {"result": str(result)}
                out.update({
                    "dataset": dataset_name,
                    "layer": layer_name,
                    "rank_spec": str(rank_spec),
                    "seed": seed,
                    "sigma": SIGMA,
                    "wall_seconds": wall,
                })

                # Extract final fitness variance
                fvh = out.get("fitness_variance_history", [])
                out["final_fitness_variance"] = fvh[-1] if fvh else None
                out["mean_fitness_variance"] = float(np.mean(fvh)) if fvh else None

                safe = {k: v for k, v in out.items()
                        if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
                fname.write_text(json.dumps(safe, indent=2, default=str))

        # Summary
        print(f"\n--- Sensitivity results: {dataset_name} ---")
        print(f"{'Layer':>15} {'acc_mean':>10} {'acc_std':>10} {'var_mean':>15} {'var_std':>15}")
        print("-" * 70)
        for layer_name in LAYER_CONFIGS:
            accs = []
            variances = []
            layer_dir = results_dir / layer_name
            for seed in SEEDS:
                fname = layer_dir / f"seed{seed}.json"
                if fname.exists():
                    with open(fname) as f:
                        d = json.load(f)
                    if d.get("best_test_acc") is not None:
                        accs.append(d["best_test_acc"])
                    if d.get("final_fitness_variance") is not None:
                        variances.append(d["final_fitness_variance"])
            if accs:
                print(f"{layer_name:>15} {np.mean(accs):>10.4f} {np.std(accs):>10.4f} "
                      f"{np.mean(variances):>15.2f} {np.std(variances):>15.2f}")

        # Save summary
        summary = {}
        for layer_name in LAYER_CONFIGS:
            accs = []
            variances = []
            layer_dir = results_dir / layer_name
            for seed in SEEDS:
                fname = layer_dir / f"seed{seed}.json"
                if fname.exists():
                    with open(fname) as f:
                        d = json.load(f)
                    if d.get("best_test_acc") is not None:
                        accs.append(d["best_test_acc"])
                    if d.get("final_fitness_variance") is not None:
                        variances.append(d["final_fitness_variance"])
            summary[layer_name] = {
                "acc_mean": float(np.mean(accs)) if accs else None,
                "acc_std": float(np.std(accs)) if accs else None,
                "variance_mean": float(np.mean(variances)) if variances else None,
                "variance_std": float(np.std(variances)) if variances else None,
                "n_seeds": len(accs),
            }
        summary_path = results_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Summary saved to {summary_path}")

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
