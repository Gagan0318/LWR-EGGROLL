"""Four-method comparison on Fashion-MNIST and KMNIST.
backprop, OpenAI-ES, Sep-CMA-ES, EGGROLL(r=4) x 2 datasets x 3 seeds = 24 runs. ~1-1.5 hours.
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
    CFG, train_backprop, train_openai_es, train_sep_cma_es, train_eggroll,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
SEEDS = (0, 1, 2)
SIGMA = 0.05
POP = 2048
WALL_BUDGET = 300.0
EGGROLL_RANK = 4

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

METHODS = {
    "backprop": lambda seed, Xtr, ytr, Xte, yte: train_backprop(seed, Xtr, ytr, Xte, yte),
    "openai_es": lambda seed, Xtr, ytr, Xte, yte: train_openai_es(seed, Xtr, ytr, Xte, yte),
    "sep_cma_es": lambda seed, Xtr, ytr, Xte, yte: train_sep_cma_es(seed, Xtr, ytr, Xte, yte),
    "eggroll_r4": lambda seed, Xtr, ytr, Xte, yte: train_eggroll(seed, Xtr, ytr, Xte, yte, rank=EGGROLL_RANK),
}

def main():
    total = 2 * len(METHODS) * len(SEEDS)
    print(f"Total runs: {total}")
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL_BUDGET
    for dataset_name in ("fashion", "kmnist"):
        print(f"\n{'='*60}")
        print(f"  DATASET: {dataset_name.upper()}")
        print(f"{'='*60}")
        X_train, y_train, X_test, y_test = load_dataset(dataset_name)
        results_dir = Path(f"results/cross_dataset/{dataset_name}/four_method")
        for method_name, train_fn in METHODS.items():
            for seed in SEEDS:
                outdir = results_dir / method_name
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"seed{seed}.json"
                if fname.exists():
                    print(f"[skip] {fname}"); continue
                print(f"\n[{dataset_name}/{method_name} seed={seed}]", flush=True)
                t0 = time.time()
                result = train_fn(seed, X_train, y_train, X_test, y_test)
                wall = time.time() - t0
                out = dict(result) if isinstance(result, dict) else {"result": str(result)}
                out.update({"dataset": dataset_name, "method": method_name, "seed": seed, "wall_seconds": wall})
                safe = {k: v for k, v in out.items() if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
                fname.write_text(json.dumps(safe, indent=2, default=str))
        summary = {}
        for method_name in METHODS:
            accs = []
            for seed in SEEDS:
                fname = results_dir / method_name / f"seed{seed}.json"
                if fname.exists():
                    with open(fname) as f:
                        d = json.load(f)
                    if d.get("best_test_acc") is not None:
                        accs.append(d["best_test_acc"])
            summary[method_name] = {"acc_mean": float(np.mean(accs)) if accs else None, "acc_std": float(np.std(accs)) if accs else None, "n_seeds": len(accs)}
        out_path = results_dir / "summary.json"
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"\n[summary] {dataset_name} four-method:")
        for k, v in summary.items():
            if v["acc_mean"] is not None:
                print(f"  {k:>15}: {v['acc_mean']:.4f} +/- {v['acc_std']:.4f}")
    print("\n=== ALL DONE ===")

if __name__ == "__main__":
    main()

