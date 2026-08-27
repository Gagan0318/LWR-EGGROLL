"""EMNIST-Digits full experiment suite.
Runs everything on EMNIST-Digits that was already run on Fashion-MNIST and KMNIST:
  1. Vanilla rank sweep (r ∈ {1,2,4,8,16,32} × 3 seeds = 18 runs)
  2. LWR allocations (5 configs × 3 seeds = 15 runs)
  3. Four-method comparison (4 methods × 3 seeds = 12 runs)
  4. Sensitivity pilot (3 layers × 5 seeds = 15 runs)
Total: 60 runs, ~2-3 hours.
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
    load_mnist,
    train_backprop,
    train_openai_es,
    train_sep_cma_es,
    train_eggroll,
    train_lwr_eggroll,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEEDS_3 = (0, 1, 2)
SEEDS_5 = (0, 1, 2, 3, 4)
SIGMA = 0.05
POP = 2048
WALL_BUDGET = 300.0

INPUT_SHAPE = (256, 784)
HIDDEN_SHAPE = (256, 256)
OUTPUT_SHAPE = (10, 256)

VANILLA_RANKS = (1, 2, 4, 8, 16, 32)

LWR_CONFIGS = {
    "lwr_8_2_0": {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 0},
    "lwr_4_1_0": {INPUT_SHAPE: 4, HIDDEN_SHAPE: 1, OUTPUT_SHAPE: 0},
    "lwr_8_4_0": {INPUT_SHAPE: 8, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 0},
    "lwr_0_2_8": {INPUT_SHAPE: 0, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 8},
    "lwr_4_4_4": {INPUT_SHAPE: 4, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 4},
}

SENSITIVITY_CONFIGS = {
    "input_only":  {INPUT_SHAPE: 4, HIDDEN_SHAPE: 0, OUTPUT_SHAPE: 0},
    "hidden_only": {INPUT_SHAPE: 0, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 0},
    "output_only": {INPUT_SHAPE: 0, HIDDEN_SHAPE: 0, OUTPUT_SHAPE: 4},
}

METHODS = {
    "backprop":    lambda seed, Xtr, ytr, Xte, yte: train_backprop(seed, Xtr, ytr, Xte, yte),
    "openai_es":   lambda seed, Xtr, ytr, Xte, yte: train_openai_es(seed, Xtr, ytr, Xte, yte),
    "sep_cma_es":  lambda seed, Xtr, ytr, Xte, yte: train_sep_cma_es(seed, Xtr, ytr, Xte, yte),
    "eggroll_r4":  lambda seed, Xtr, ytr, Xte, yte: train_eggroll(seed, Xtr, ytr, Xte, yte, rank=4),
}

DATASET = "emnist"


def load_emnist():
    train = tv_datasets.EMNIST(root=str(DATA_DIR), split='digits', train=True, download=True)
    test  = tv_datasets.EMNIST(root=str(DATA_DIR), split='digits', train=False, download=True)
    X_train = np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0
    y_train = np.array(train.targets, dtype=np.int32)
    X_test  = np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0
    y_test  = np.array(test.targets, dtype=np.int32)
    X_train, y_train = jnp.asarray(X_train), jnp.asarray(y_train)
    X_test, y_test = jnp.asarray(X_test), jnp.asarray(y_test)
    print(f"[data] EMNIST-Digits: X_train {X_train.shape}, X_test {X_test.shape}")
    print(f"[data] classes: {len(np.unique(np.array(test.targets)))}")
    return X_train, y_train, X_test, y_test


def save_result(result, extra, fname):
    out = dict(result) if isinstance(result, dict) else {"result": str(result)}
    out.update(extra)
    safe = {k: v for k, v in out.items()
            if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
    fname.write_text(json.dumps(safe, indent=2, default=str))


def run_vanilla(X_train, y_train, X_test, y_test):
    print("\n--- 1/4: Vanilla EGGROLL rank sweep ---")
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL_BUDGET
    for rank in VANILLA_RANKS:
        for seed in SEEDS_3:
            outdir = Path(f"results/cross_dataset/{DATASET}/vanilla/eggroll_r{rank}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue
            print(f"\n[vanilla r={rank} seed={seed}]", flush=True)
            t0 = time.time()
            result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank)
            save_result(result, {"dataset": DATASET, "method": "eggroll", "rank": rank,
                                 "seed": seed, "sigma": SIGMA, "wall_seconds": time.time() - t0}, fname)


def run_lwr(X_train, y_train, X_test, y_test):
    print("\n--- 2/4: LWR-EGGROLL allocations ---")
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL_BUDGET
    for name, rspec in LWR_CONFIGS.items():
        for seed in SEEDS_3:
            outdir = Path(f"results/cross_dataset/{DATASET}/lwr/{name}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue
            print(f"\n[lwr {name} seed={seed}]", flush=True)
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rspec, label=name)
            save_result(result, {"dataset": DATASET, "method": "lwr", "config": name,
                                 "seed": seed, "sigma": SIGMA, "wall_seconds": time.time() - t0}, fname)


def run_four_method(X_train, y_train, X_test, y_test):
    print("\n--- 3/4: Four-method comparison ---")
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL_BUDGET
    for method_name, train_fn in METHODS.items():
        for seed in SEEDS_3:
            outdir = Path(f"results/cross_dataset/{DATASET}/four_method/{method_name}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue
            print(f"\n[four_method {method_name} seed={seed}]", flush=True)
            t0 = time.time()
            result = train_fn(seed, X_train, y_train, X_test, y_test)
            save_result(result, {"dataset": DATASET, "method": method_name,
                                 "seed": seed, "wall_seconds": time.time() - t0}, fname)


def run_sensitivity(X_train, y_train, X_test, y_test):
    print("\n--- 4/4: Sensitivity pilot ---")
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL_BUDGET
    for layer_name, rank_spec in SENSITIVITY_CONFIGS.items():
        for seed in SEEDS_5:
            outdir = Path(f"results/sensitivity_pilot/{DATASET}/{layer_name}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue
            print(f"\n[sensitivity {layer_name} seed={seed}]", flush=True)
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rank_spec, label=layer_name)
            wall = time.time() - t0
            out_extra = {"dataset": DATASET, "layer": layer_name,
                         "rank_spec": str(rank_spec), "seed": seed, "sigma": SIGMA,
                         "wall_seconds": wall}
            out = dict(result) if isinstance(result, dict) else {"result": str(result)}
            fvh = out.get("fitness_variance_history", [])
            out_extra["final_fitness_variance"] = fvh[-1] if fvh else None
            out_extra["mean_fitness_variance"] = float(np.mean(fvh)) if fvh else None
            save_result(result, out_extra, fname)


def summarise():
    base = Path(f"results/cross_dataset/{DATASET}")

    # Vanilla + LWR summary
    summary = {}
    for json_file in sorted(base.rglob("seed*.json")):
        if "four_method" in str(json_file):
            continue
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
        agg[k] = {"acc_mean": float(np.mean(valid)) if valid else None,
                   "acc_std": float(np.std(valid)) if valid else None, "n_seeds": len(valid)}
    (base / "summary.json").write_text(json.dumps(agg, indent=2))
    print(f"\n[summary] Vanilla + LWR:")
    for k, v in sorted(agg.items()):
        if v["acc_mean"] is not None:
            print(f"  {k}: {v['acc_mean']:.4f} +/- {v['acc_std']:.4f}")

    # Four-method summary
    fm_dir = base / "four_method"
    fm_summary = {}
    for method_name in METHODS:
        accs = []
        for seed in SEEDS_3:
            fname = fm_dir / method_name / f"seed{seed}.json"
            if fname.exists():
                with open(fname) as f:
                    d = json.load(f)
                if d.get("best_test_acc") is not None:
                    accs.append(d["best_test_acc"])
        fm_summary[method_name] = {"acc_mean": float(np.mean(accs)) if accs else None,
                                    "acc_std": float(np.std(accs)) if accs else None, "n_seeds": len(accs)}
    (fm_dir / "summary.json").write_text(json.dumps(fm_summary, indent=2))
    print(f"\n[summary] Four-method:")
    for k, v in fm_summary.items():
        if v["acc_mean"] is not None:
            print(f"  {k:>15}: {v['acc_mean']:.4f} +/- {v['acc_std']:.4f}")

    # Sensitivity summary
    sens_dir = Path(f"results/sensitivity_pilot/{DATASET}")
    sens_summary = {}
    for layer_name in SENSITIVITY_CONFIGS:
        accs, variances = [], []
        layer_dir = sens_dir / layer_name
        for seed in SEEDS_5:
            fname = layer_dir / f"seed{seed}.json"
            if fname.exists():
                with open(fname) as f:
                    d = json.load(f)
                if d.get("best_test_acc") is not None:
                    accs.append(d["best_test_acc"])
                if d.get("final_fitness_variance") is not None:
                    variances.append(d["final_fitness_variance"])
        sens_summary[layer_name] = {
            "acc_mean": float(np.mean(accs)) if accs else None,
            "acc_std": float(np.std(accs)) if accs else None,
            "variance_mean": float(np.mean(variances)) if variances else None,
            "variance_std": float(np.std(variances)) if variances else None,
            "n_seeds": len(accs),
        }
    (sens_dir / "summary.json").write_text(json.dumps(sens_summary, indent=2))
    print(f"\n[summary] Sensitivity pilot:")
    for k, v in sens_summary.items():
        if v["acc_mean"] is not None:
            print(f"  {k:>15}: acc={v['acc_mean']:.4f}  variance={v.get('variance_mean', 'N/A')}")


def main():
    print("="*60)
    print("  EMNIST-DIGITS FULL EXPERIMENT SUITE")
    print("="*60)

    X_train, y_train, X_test, y_test = load_emnist()

    run_vanilla(X_train, y_train, X_test, y_test)
    run_lwr(X_train, y_train, X_test, y_test)
    run_four_method(X_train, y_train, X_test, y_test)
    run_sensitivity(X_train, y_train, X_test, y_test)
    summarise()

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
