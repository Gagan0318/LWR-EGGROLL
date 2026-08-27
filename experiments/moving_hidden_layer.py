"""Moving Hidden Layer — Tapered Architecture Experiment.
Tests whether a tapered MLP [784, 512, 256, 128, 10] lets LWR assign
differentiated ranks across structurally distinct hidden layers.

Standard architecture has hidden shapes (256, 256), making hidden layers
indistinguishable by shape-based rank lookup. Tapered architecture gives
four unique shapes: (512, 784), (256, 512), (128, 256), (10, 128).

Phases:
  1. Sensitivity pilot: perturb one layer at a time (r=4, others r=0)
  2. Vanilla rank sweep: uniform r in {1, 2, 4, 8}
  3. LWR comparison: pilot-derived allocation vs uniform baselines

Seeds: 5 (pilot), 3 (main runs). Architecture: [784, 512, 256, 128, 10].
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

import sys, json, time
from pathlib import Path
import numpy as np
import jax

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_4_methods_mnist import (
    CFG,
    load_mnist,
    train_eggroll,
    train_lwr_eggroll,
)

# Tapered architecture: [784, 512, 256, 128, 10]
TAPERED = {
    "hidden_dims": (512, 256, 128),
    "shapes": {
        "input":   (512, 784),   # Dense_0
        "hidden1": (256, 512),   # Dense_1
        "hidden2": (128, 256),   # Dense_2
        "output":  (10, 128),    # Dense_3
    },
}

SEEDS_5 = (0, 1, 2, 3, 4)
SEEDS_3 = (0, 1, 2)
OUT_DIR = Path("results/tapered")


def load_mnist_local():
    """Load MNIST via the shared loader."""
    X_train, y_train, X_test, y_test = load_mnist()
    print(f"[data] MNIST: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


def save_result(result, extra, fname):
    out = dict(result) if isinstance(result, dict) else {"result": str(result)}
    out.update(extra)
    safe = {k: v for k, v in out.items()
            if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.write_text(json.dumps(safe, indent=2, default=str))


def setup_cfg(hidden_dims, sigma=0.05, pop=2048, wall=300.0):
    CFG.hidden_dims = hidden_dims
    CFG.n_classes = 10
    CFG.eggroll_sigma_init = sigma
    CFG.eggroll_pop = pop
    CFG.max_wall_seconds = wall


def main():
    print(f"[env] JAX {jax.__version__} on {jax.default_backend()}")
    print(f"[env] Devices: {jax.devices()}")
    print("=" * 60)
    print("  MOVING HIDDEN LAYER — TAPERED ARCHITECTURE")
    print("  Architecture: [784, 512, 256, 128, 10]")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist_local()

    hd = TAPERED["hidden_dims"]
    shapes = TAPERED["shapes"]
    inp  = shapes["input"]    # (512, 784)
    hid1 = shapes["hidden1"]  # (256, 512)
    hid2 = shapes["hidden2"]  # (128, 256)
    out  = shapes["output"]   # (10, 128)

    setup_cfg(hd)

    # ═══════════════════════════════════════════════════════════
    #  PHASE 1: SENSITIVITY PILOT (n=5)
    #  Perturb one layer at a time, others frozen at r=0
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 1: SENSITIVITY PILOT")
    print("=" * 60)

    pilot_groups = {
        "input_only":   {inp: 4, hid1: 0, hid2: 0, out: 0},
        "hidden1_only": {inp: 0, hid1: 4, hid2: 0, out: 0},
        "hidden2_only": {inp: 0, hid1: 0, hid2: 4, out: 0},
        "output_only":  {inp: 0, hid1: 0, hid2: 0, out: 4},
    }

    pilot_results = {}
    for group_name, rspec in pilot_groups.items():
        print(f"\n--- Sensitivity: {group_name} ---")
        print(f"    rank_spec: {rspec}")
        accs = []
        for seed in SEEDS_5:
            fname = OUT_DIR / "sensitivity" / group_name / f"seed{seed}.json"
            if fname.exists():
                print(f"    seed {seed}: already exists, skipping")
                with open(fname) as f:
                    d = json.load(f)
                accs.append(d.get("best_test_acc", 0))
                continue
            print(f"    seed {seed} ...", end=" ", flush=True)
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rspec, label=group_name)
            wall = time.time() - t0
            save_result(result, {
                "arch": "tapered_512_256_128",
                "experiment": "sensitivity",
                "group": group_name,
                "seed": seed,
                "hidden_dims": list(hd),
                "rank_spec": str(rspec),
                "wall_s": wall,
            }, fname)
            acc = result.get("best_test_acc", 0)
            accs.append(acc)
            print(f"acc={acc:.4f}  ({wall:.1f}s)")
        pilot_results[group_name] = {
            "mean": float(np.mean(accs)),
            "std": float(np.std(accs)),
            "per_seed": accs,
        }

    # Print sensitivity ordering
    print("\n--- Sensitivity Ordering ---")
    ordering = sorted(pilot_results.items(), key=lambda x: x[1]["mean"], reverse=True)
    for name, stats in ordering:
        print(f"  {name:<15} {stats['mean']:.4f} ± {stats['std']:.4f}")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2: VANILLA RANK SWEEP
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2: VANILLA EGGROLL RANK SWEEP")
    print("=" * 60)

    for rank in [1, 2, 4, 8]:
        print(f"\n--- Vanilla r={rank} ---")
        for seed in SEEDS_3:
            fname = OUT_DIR / "vanilla" / f"eggroll_r{rank}" / f"seed{seed}.json"
            if fname.exists():
                print(f"    seed {seed}: already exists, skipping")
                continue
            print(f"    seed {seed} ...", end=" ", flush=True)
            t0 = time.time()
            result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank)
            wall = time.time() - t0
            save_result(result, {
                "arch": "tapered_512_256_128",
                "experiment": "vanilla",
                "rank": rank,
                "seed": seed,
                "hidden_dims": list(hd),
                "wall_s": wall,
            }, fname)
            acc = result.get("best_test_acc", 0)
            print(f"acc={acc:.4f}  ({wall:.1f}s)")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3: LWR-EGGROLL COMPARISON
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 3: LWR-EGGROLL")
    print("=" * 60)

    # Pilot-derived allocation: input > hidden1 > hidden2 > output
    lwr_configs = {
        "lwr_8_4_2_0": {inp: 8, hid1: 4, hid2: 2, out: 0},
        "lwr_4_2_1_0": {inp: 4, hid1: 2, hid2: 1, out: 0},
    }

    for config_name, rspec in lwr_configs.items():
        print(f"\n--- {config_name} ---")
        print(f"    rank_spec: {rspec}")
        for seed in SEEDS_3:
            fname = OUT_DIR / "lwr" / config_name / f"seed{seed}.json"
            if fname.exists():
                print(f"    seed {seed}: already exists, skipping")
                continue
            print(f"    seed {seed} ...", end=" ", flush=True)
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rspec, label=config_name)
            wall = time.time() - t0
            save_result(result, {
                "arch": "tapered_512_256_128",
                "experiment": "lwr",
                "config": config_name,
                "seed": seed,
                "hidden_dims": list(hd),
                "rank_spec": str(rspec),
                "wall_s": wall,
            }, fname)
            acc = result.get("best_test_acc", 0)
            print(f"acc={acc:.4f}  ({wall:.1f}s)")

    # ═══════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    # Collect all results
    summary = {"sensitivity": {}, "vanilla": {}, "lwr": {}}

    # Sensitivity
    for group_name in pilot_groups:
        accs = []
        for seed in SEEDS_5:
            fname = OUT_DIR / "sensitivity" / group_name / f"seed{seed}.json"
            if fname.exists():
                with open(fname) as f:
                    d = json.load(f)
                if d.get("best_test_acc") is not None:
                    accs.append(d["best_test_acc"])
        if accs:
            summary["sensitivity"][group_name] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "n": len(accs),
            }

    # Vanilla
    for rank in [1, 2, 4, 8]:
        accs = []
        for seed in SEEDS_3:
            fname = OUT_DIR / "vanilla" / f"eggroll_r{rank}" / f"seed{seed}.json"
            if fname.exists():
                with open(fname) as f:
                    d = json.load(f)
                if d.get("best_test_acc") is not None:
                    accs.append(d["best_test_acc"])
        if accs:
            summary["vanilla"][f"eggroll_r{rank}"] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "n": len(accs),
            }

    # LWR
    for config_name in lwr_configs:
        accs = []
        for seed in SEEDS_3:
            fname = OUT_DIR / "lwr" / config_name / f"seed{seed}.json"
            if fname.exists():
                with open(fname) as f:
                    d = json.load(f)
                if d.get("best_test_acc") is not None:
                    accs.append(d["best_test_acc"])
        if accs:
            summary["lwr"][config_name] = {
                "mean": float(np.mean(accs)),
                "std": float(np.std(accs)),
                "n": len(accs),
            }

    # Print
    print("\nSensitivity (5 seeds):")
    for name, stats in sorted(summary["sensitivity"].items(),
                               key=lambda x: x[1]["mean"], reverse=True):
        print(f"  {name:<15} {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\nVanilla EGGROLL (3 seeds):")
    for name, stats in summary["vanilla"].items():
        print(f"  {name:<15} {stats['mean']:.4f} ± {stats['std']:.4f}")

    print("\nLWR-EGGROLL (3 seeds):")
    for name, stats in summary["lwr"].items():
        print(f"  {name:<15} {stats['mean']:.4f} ± {stats['std']:.4f}")

    # Save summary
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
