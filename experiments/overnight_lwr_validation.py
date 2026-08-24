#!/usr/bin/env python3
"""
overnight_lwr_validation.py

Overnight experiment runner:
  Part 1: LWR (8,4,0) on MNIST — 3 seeds  (~15 min)
  Part 2: LWR (4,2,0) on all 4 datasets — 3 seeds each  (~60 min)
  Part 3: Vanilla r=4 baselines if missing  (~60 min)

Total estimated: ~2-2.5 hours on RTX 5060

Usage:
    cd ~/dissertation/eggroll-diss
    nohup python experiments/overnight_lwr_validation.py > logs/overnight.log 2>&1 &
    tail -f logs/overnight.log
"""

import sys
import json
import time
import os
import numpy as np
from pathlib import Path
from datetime import datetime

REPO_ROOT = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_4_methods_mnist import (
    load_mnist, train_eggroll, train_lwr_eggroll, save_result,
)

# ── Shapes (HyperscaleES convention: output_dim, input_dim) ──
INPUT_SHAPE  = (256, 784)
HIDDEN_SHAPE = (256, 256)
OUTPUT_SHAPE = (10, 256)

# ── Rank specifications ──
RANK_8_4_0 = {INPUT_SHAPE: 8, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 0}  # pilot-derived, budget 12
RANK_4_2_0 = {INPUT_SHAPE: 4, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 0}  # half-budget,   budget 6

SEEDS = [0, 1, 2]

# ── Output directory ──
OUTPUT_DIR = Path(REPO_ROOT) / "results" / "overnight_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dataset loaders ──
def load_fashion_mnist():
    """Load Fashion-MNIST using the same preprocessing as load_mnist()."""
    import jax.numpy as jnp
    try:
        from torchvision import datasets
        ds_train = datasets.FashionMNIST(root='/tmp/data', train=True, download=True)
        ds_test  = datasets.FashionMNIST(root='/tmp/data', train=False, download=True)
        X_train = jnp.array(ds_train.data.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_train = jnp.array(ds_train.targets.numpy(), dtype=jnp.int32)
        X_test  = jnp.array(ds_test.data.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_test  = jnp.array(ds_test.targets.numpy(), dtype=jnp.int32)
    except ImportError:
        from tensorflow.keras.datasets import fashion_mnist
        (xt, yt), (xv, yv) = fashion_mnist.load_data()
        X_train = jnp.array(xt.reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_train = jnp.array(yt, dtype=jnp.int32)
        X_test  = jnp.array(xv.reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_test  = jnp.array(yv, dtype=jnp.int32)
    return X_train, y_train, X_test, y_test

def load_kmnist():
    """Load KMNIST."""
    import jax.numpy as jnp
    try:
        from torchvision import datasets
        ds_train = datasets.KMNIST(root='/tmp/data', train=True, download=True)
        ds_test  = datasets.KMNIST(root='/tmp/data', train=False, download=True)
        X_train = jnp.array(ds_train.data.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_train = jnp.array(ds_train.targets.numpy(), dtype=jnp.int32)
        X_test  = jnp.array(ds_test.data.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_test  = jnp.array(ds_test.targets.numpy(), dtype=jnp.int32)
    except Exception:
        import tensorflow_datasets as tfds
        ds = tfds.load('kmnist', split=['train', 'test'], as_supervised=True, batch_size=-1)
        (xt, yt), (xv, yv) = ds
        X_train = jnp.array(xt.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_train = jnp.array(yt.numpy(), dtype=jnp.int32)
        X_test  = jnp.array(xv.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_test  = jnp.array(yv.numpy(), dtype=jnp.int32)
    return X_train, y_train, X_test, y_test

def load_emnist_digits():
    """Load EMNIST-Digits."""
    import jax.numpy as jnp
    try:
        from torchvision import datasets
        ds_train = datasets.EMNIST(root='/tmp/data', split='digits', train=True, download=True)
        ds_test  = datasets.EMNIST(root='/tmp/data', split='digits', train=False, download=True)
        X_train = jnp.array(ds_train.data.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_train = jnp.array(ds_train.targets.numpy(), dtype=jnp.int32)
        X_test  = jnp.array(ds_test.data.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_test  = jnp.array(ds_test.targets.numpy(), dtype=jnp.int32)
    except Exception:
        import tensorflow_datasets as tfds
        ds = tfds.load('emnist/digits', split=['train', 'test'], as_supervised=True, batch_size=-1)
        (xt, yt), (xv, yv) = ds
        X_train = jnp.array(xt.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_train = jnp.array(yt.numpy(), dtype=jnp.int32)
        X_test  = jnp.array(xv.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
        y_test  = jnp.array(yv.numpy(), dtype=jnp.int32)
    return X_train, y_train, X_test, y_test

DATASET_LOADERS = {
    "mnist":         load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "kmnist":        load_kmnist,
    "emnist_digits": load_emnist_digits,
}


def run_single(dataset_name, config_label, rank_spec, seed, X_train, y_train, X_test, y_test):
    """Run a single LWR experiment and save result JSON."""
    out_path = OUTPUT_DIR / dataset_name / config_label / f"seed{seed}.json"
    
    # Skip if already completed
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        print(f"  SKIP {dataset_name}/{config_label}/seed{seed} "
              f"(already done: {existing['best_test_acc']:.4f})")
        return existing['best_test_acc']
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  RUN  {dataset_name}/{config_label}/seed{seed} ...", end=" ", flush=True)
    t0 = time.time()
    
    result = train_lwr_eggroll(
        seed=seed,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        rank_spec=rank_spec,
        label=config_label,
    )
    
    elapsed = time.time() - t0
    
    # Enrich result with metadata
    result['dataset'] = dataset_name
    result['config'] = config_label
    result['seed'] = seed
    result['rank_spec_str'] = str(rank_spec)
    result['wall_time_s'] = elapsed
    result['timestamp'] = datetime.now().isoformat()
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    acc = result.get('best_test_acc', 0)
    print(f"acc={acc:.4f}  ({elapsed:.0f}s)")
    return acc


def run_vanilla(dataset_name, seed, X_train, y_train, X_test, y_test, rank=4):
    """Run vanilla EGGROLL r=4 baseline."""
    config_label = f"vanilla_r{rank}"
    out_path = OUTPUT_DIR / dataset_name / config_label / f"seed{seed}.json"
    
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        print(f"  SKIP {dataset_name}/{config_label}/seed{seed} "
              f"(already done: {existing['best_test_acc']:.4f})")
        return existing['best_test_acc']
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"  RUN  {dataset_name}/{config_label}/seed{seed} ...", end=" ", flush=True)
    t0 = time.time()
    
    result = train_eggroll(
        seed=seed,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        rank=rank,
    )
    
    elapsed = time.time() - t0
    result['dataset'] = dataset_name
    result['config'] = config_label
    result['seed'] = seed
    result['wall_time_s'] = elapsed
    result['timestamp'] = datetime.now().isoformat()
    
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    acc = result.get('best_test_acc', 0)
    print(f"acc={acc:.4f}  ({elapsed:.0f}s)")
    return acc


def summarise(dataset_name, configs):
    """Print summary table for a dataset."""
    print(f"\n{'─' * 60}")
    print(f"  SUMMARY: {dataset_name.upper()}")
    print(f"{'─' * 60}")
    for label, accs in configs.items():
        if accs:
            mean = np.mean(accs) * 100
            std = np.std(accs) * 100
            print(f"  {label:20s}  {mean:.2f} ± {std:.2f}%")
    
    # Compute advantages
    if 'vanilla_r4' in configs and configs['vanilla_r4']:
        van_mean = np.mean(configs['vanilla_r4']) * 100
        for label, accs in configs.items():
            if label != 'vanilla_r4' and accs:
                lwr_mean = np.mean(accs) * 100
                print(f"    → {label} vs vanilla: {lwr_mean - van_mean:+.2f}pp")
    print(f"{'─' * 60}\n")


# ════════════════════════════════════════════════════════════════
#  MAIN OVERNIGHT RUNNER
# ════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    total_start = time.time()
    print("=" * 60)
    print("  OVERNIGHT LWR VALIDATION")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # ── PART 1: LWR (8,4,0) on MNIST only ─────────────────────
    print("\n\n╔══════════════════════════════════════════════╗")
    print("║  PART 1: LWR (8,4,0) on MNIST — 3 seeds     ║")
    print("║  Budget 12 (same as vanilla r=4)             ║")
    print("╚══════════════════════════════════════════════╝")
    
    X_tr, y_tr, X_te, y_te = load_mnist()
    
    mnist_840 = []
    for seed in SEEDS:
        acc = run_single("mnist", "lwr_8_4_0", RANK_8_4_0, seed, X_tr, y_tr, X_te, y_te)
        mnist_840.append(acc)
    
    # Also run vanilla r=4 on MNIST if not cached
    mnist_van = []
    for seed in SEEDS:
        acc = run_vanilla("mnist", seed, X_tr, y_tr, X_te, y_te)
        mnist_van.append(acc)
    
    summarise("mnist", {"lwr_8_4_0": mnist_840, "vanilla_r4": mnist_van})
    
    # ── PART 2: LWR (4,2,0) on ALL 4 datasets ─────────────────
    print("\n\n╔══════════════════════════════════════════════╗")
    print("║  PART 2: LWR (4,2,0) on all datasets        ║")
    print("║  Budget 6 (half of vanilla r=4)              ║")
    print("╚══════════════════════════════════════════════╝")
    
    for ds_name, loader in DATASET_LOADERS.items():
        print(f"\n── {ds_name.upper()} ──")
        X_tr, y_tr, X_te, y_te = loader()
        
        ds_420 = []
        for seed in SEEDS:
            acc = run_single(ds_name, "lwr_4_2_0", RANK_4_2_0, seed, X_tr, y_tr, X_te, y_te)
            ds_420.append(acc)
        
        # Vanilla baseline for this dataset
        ds_van = []
        for seed in SEEDS:
            acc = run_vanilla(ds_name, seed, X_tr, y_tr, X_te, y_te)
            ds_van.append(acc)
        
        summarise(ds_name, {"lwr_4_2_0": ds_420, "vanilla_r4": ds_van})
    
    # ── FINAL SUMMARY ──────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total wall time: {total_elapsed/60:.1f} minutes")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Results in: {OUTPUT_DIR}")
    print("=" * 60)
