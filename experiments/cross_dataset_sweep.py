"""Cross-dataset generalisation: vanilla rank sweep + LWR on Fashion-MNIST and KMNIST.
66 runs, ~2-3 hours.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import time
import numpy as np
import jax.numpy as jnp
from torchvision import datasets
from experiments.compare_4_methods_mnist import (
    CFG,
    train_eggroll,
    train_lwr_eggroll,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SEEDS = (0, 1, 2)
SIGMA = 0.05
POP = 2048
WALL_BUDGET = 300.0
VANILLA_RANKS = (1, 2, 4, 8, 16, 32)
LWR_CONFIGS = {
    "lwr_8_2_0":  {(256,784): 8, (256,256): 2, (10,256): 0},
    "lwr_4_1_0":  {(256,784): 4, (256,256): 1, (10,256): 0},
    "lwr_8_4_0":  {(256,784): 8, (256,256): 4, (10,256): 0},
    "lwr_0_2_8":  {(256,784): 0, (256,256): 2, (10,256): 8},
    "lwr_4_4_4":  {(256,784): 4, (256,256): 4, (10,256): 4},
}

def load_dataset(name):
    ds_class = {"fashion": datasets.FashionMNIST, "kmnist": datasets.KMNIST}[name]
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

def run_vanilla(dataset_name, X_train, y_train, X_test, y_test):
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL_BUDGET
    for rank in VANILLA_RANKS:
        for seed in SEEDS:
            outdir = Path(f"results/cross_dataset/{dataset_name}/vanilla/eggroll_r{rank}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue
            print(f"\n[{dataset_name}/vanilla r={rank} seed={seed}]", flush=True)
            t0 = time.time()
            result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank)
            wall = time.time() - t0
            out = dict(result) if isinstance(result, dict) else {"result": str(result)}
            out.update({"dataset": dataset_name, "method": "eggroll", "rank": rank, "seed": seed, "sigma": SIGMA, "wall_seconds": wall})
            safe = {k: v for k, v in out.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
            fname.write_text(json.dumps(safe, indent=2, default=str))

def run_lwr(dataset_name, X_train, y_train, X_test, y_test):
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL_BUDGET
    for name, rspec in LWR_CONFIGS.items():
        for seed in SEEDS:
            outdir = Path(f"results/cross_dataset/{dataset_name}/lwr/{name}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue
            print(f"\n[{dataset_name}/lwr {name} seed={seed}]", flush=True)
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test, rank_spec=rspec, label=name)
            wall = time.time() - t0
            out = dict(result) if isinstance(result, dict) else {"result": str(result)}
            out.update({"dataset": dataset_name, "method": "lwr", "config": name, "seed": seed, "sigma": SIGMA, "wall_seconds": wall})
            safe = {k: v for k, v in out.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
            fname.write_text(json.dumps(safe, indent=2, default=str))

def summarise(dataset_name):
    results_dir = Path(f"results/cross_dataset/{dataset_name}")
    summary = {}
    for json_file in sorted(results_dir.rglob("seed*.json")):
        with open(json_file) as f:
            d = json.load(f)
        group = "vanilla" if json_file.parent.parent.name == "vanilla" else "lwr"
        full_key = f"{group}/{json_file.parent.name}"
        if full_key not in summary:
            summary[full_key] = []
        summary[full_key].append(d.get("best_test_acc"))
    agg = {}
    for k, accs in summary.items():
        valid = [a for a in accs if a is not None]
        agg[k] = {"acc_mean": float(np.mean(valid)) if valid else None, "acc_std": float(np.std(valid)) if valid else None, "n_seeds": len(valid)}
    out_path = results_dir / "summary.json"
    out_path.write_text(json.dumps(agg, indent=2))
    print(f"\n[summary] {dataset_name}: {out_path}")
    for k, v in sorted(agg.items()):
        if v["acc_mean"] is not None:
            print(f"  {k}: {v['acc_mean']:.4f} +/- {v['acc_std']:.4f}")

def main():
    for dataset_name in ("fashion", "kmnist"):
        print(f"\n{'='*60}")
        print(f"  DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        print("\n--- Vanilla EGGROLL rank sweep ---")
        run_vanilla(dataset_name, X_train, y_train, X_test, y_test)
        print("\n--- LWR-EGGROLL allocations ---")
        run_lwr(dataset_name, X_train, y_train, X_test, y_test)
        summarise(dataset_name)
    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()
