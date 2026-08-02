"""LWR robustness suite — four generalisation experiments.
1. EMNIST-Letters (26 classes): does "freeze output" hold with larger output?
2. Transfer test: MNIST-derived allocation applied to Fashion/KMNIST without re-piloting
3. LWR at different population sizes: N=512, N=4096
4. Hidden rank granularity: input=8, output=0, hidden in {0,1,2,4,8}
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
    train_eggroll,
    train_lwr_eggroll,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEEDS_3 = (0, 1, 2)
SEEDS_5 = (0, 1, 2, 3, 4)

# Standard shapes for [784, 256, 256, 256, 10] — transposed as HyperscaleES stores them
INPUT_SHAPE  = (256, 784)
HIDDEN_SHAPE = (256, 256)
OUTPUT_SHAPE = (10, 256)


def save_result(result, extra, fname):
    out = dict(result) if isinstance(result, dict) else {"result": str(result)}
    out.update(extra)
    safe = {k: v for k, v in out.items()
            if isinstance(v, (int, float, str, list, dict, bool, type(None)))}
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.write_text(json.dumps(safe, indent=2, default=str))


def load_mnist():
    train = tv_datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test  = tv_datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32))
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32))
    print(f"[data] MNIST: train={X_train.shape[0]}, test={X_test.shape[0]}, classes=10")
    return X_train, y_train, X_test, y_test


def load_fashion_mnist():
    train = tv_datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True)
    test  = tv_datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32))
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32))
    print(f"[data] Fashion-MNIST: train={X_train.shape[0]}, test={X_test.shape[0]}, classes=10")
    return X_train, y_train, X_test, y_test


def load_kmnist():
    train = tv_datasets.KMNIST(root=str(DATA_DIR), train=True, download=True)
    test  = tv_datasets.KMNIST(root=str(DATA_DIR), train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32))
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32))
    print(f"[data] KMNIST: train={X_train.shape[0]}, test={X_test.shape[0]}, classes=10")
    return X_train, y_train, X_test, y_test


def setup_cfg(sigma=0.05, pop=2048, wall=300.0):
    CFG.eggroll_sigma_init = sigma
    CFG.eggroll_pop = pop
    CFG.max_wall_seconds = wall


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 1: HIDDEN RANK GRANULARITY SWEEP
#  Fix input=8, output=0, sweep hidden in {0,1,2,4,8}
# ═══════════════════════════════════════════════════════════════

def experiment_hidden_sweep():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: HIDDEN RANK GRANULARITY SWEEP")
    print("  Fix input=8, output=0, sweep hidden in {0,1,2,4,8}")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    setup_cfg()

    for h_rank in [0, 1, 2, 4, 8]:
        rank_dict = {INPUT_SHAPE: 8, HIDDEN_SHAPE: h_rank, OUTPUT_SHAPE: 0}
        label = f"lwr_8_{h_rank}_0"
        print(f"\n--- {label} ---")

        for seed in SEEDS_3:
            outdir = Path(f"results/generalisation/hidden_sweep/{label}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue

            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rank_dict, label=label)
            save_result(result, {"experiment": "hidden_sweep", "hidden_rank": h_rank,
                                 "seed": seed, "wall_s": time.time() - t0}, fname)


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 2: LWR AT DIFFERENT POPULATION SIZES
#  Does LWR advantage hold at N=512 and N=4096?
# ═══════════════════════════════════════════════════════════════

def experiment_pop_size():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: LWR AT DIFFERENT POPULATION SIZES")
    print("  Does LWR advantage hold at N=512 and N=4096?")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    lwr_820 = {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 0}

    for N in [512, 4096]:
        print(f"\n--- Population N={N} ---")
        setup_cfg(pop=N)

        # Vanilla r=4
        for seed in SEEDS_3:
            outdir = Path(f"results/generalisation/pop_size/N{N}/vanilla_r4")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[vanilla_r4 N={N} seed={seed}]")
            t0 = time.time()
            result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=4)
            save_result(result, {"experiment": "pop_size", "pop": N, "method": "vanilla_r4",
                                 "seed": seed, "wall_s": time.time() - t0}, fname)

        # LWR (8,2,0)
        for seed in SEEDS_3:
            outdir = Path(f"results/generalisation/pop_size/N{N}/lwr_8_2_0")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[lwr_8_2_0 N={N} seed={seed}]")
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=lwr_820, label=f"lwr_8_2_0_N{N}")
            save_result(result, {"experiment": "pop_size", "pop": N, "method": "lwr_8_2_0",
                                 "seed": seed, "wall_s": time.time() - t0}, fname)

    # Reset pop to default
    setup_cfg(pop=2048)


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 3: TRANSFER TEST
#  MNIST-derived (8,2,0) applied to Fashion/KMNIST
#  WITHOUT re-running sensitivity pilot
# ═══════════════════════════════════════════════════════════════

def experiment_transfer():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: TRANSFER TEST")
    print("  MNIST-derived (8,2,0) applied to Fashion/KMNIST")
    print("  WITHOUT re-running sensitivity pilot")
    print("=" * 60)

    setup_cfg()
    lwr_820     = {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 0}
    lwr_028_rev = {INPUT_SHAPE: 0, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 8}

    for dataset_name, loader in [("fashion_mnist", load_fashion_mnist),
                                  ("kmnist", load_kmnist)]:
        print(f"\n--- Transfer: {dataset_name} ---")
        X_train, y_train, X_test, y_test = loader()

        configs = {
            "vanilla_r4":           (4, None),
            "lwr_8_2_0_transferred": (None, lwr_820),
            "lwr_0_2_8_reversed":    (None, lwr_028_rev),
        }

        for name, (rank_int, rank_dict) in configs.items():
            for seed in SEEDS_3:
                outdir = Path(f"results/generalisation/transfer/{dataset_name}/{name}")
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"seed{seed}.json"
                if fname.exists():
                    print(f"[skip] {fname}"); continue

                print(f"\n[{name} {dataset_name} seed={seed}]")
                t0 = time.time()
                if rank_dict is not None:
                    result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                               rank_spec=rank_dict, label=name)
                else:
                    result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank_int)

                save_result(result, {"experiment": "transfer", "dataset": dataset_name,
                                     "config": name, "seed": seed,
                                     "wall_s": time.time() - t0}, fname)


# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT 4: EMNIST-LETTERS (26 classes)
#  Does "freeze output" hold with larger output layer?
# ═══════════════════════════════════════════════════════════════

def load_emnist_letters():
    train = tv_datasets.EMNIST(root=str(DATA_DIR), split="letters", train=True, download=True)
    test  = tv_datasets.EMNIST(root=str(DATA_DIR), split="letters", train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32) - 1)  # 1-indexed to 0-indexed
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32) - 1)
    n_classes = int(y_train.max()) + 1
    print(f"[data] EMNIST-Letters: train={X_train.shape[0]}, test={X_test.shape[0]}, classes={n_classes}")
    return X_train, y_train, X_test, y_test, n_classes


def experiment_emnist_letters():
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: EMNIST-LETTERS (26 classes)")
    print("  Does 'freeze output' hold with larger output layer?")
    print("=" * 60)

    X_train, y_train, X_test, y_test, n_classes = load_emnist_letters()

    # Update CFG for 26 classes
    CFG.n_classes = n_classes
    setup_cfg()

    # Shapes with 26-class output: output becomes (26, 256)
    OUTPUT_26 = (26, 256)

    # Sensitivity pilot
    print("\n--- Sensitivity pilot (26 classes) ---")
    sensitivity_configs = {
        "input_only":  {INPUT_SHAPE: 4, HIDDEN_SHAPE: 0, OUTPUT_26: 0},
        "hidden_only": {INPUT_SHAPE: 0, HIDDEN_SHAPE: 4, OUTPUT_26: 0},
        "output_only": {INPUT_SHAPE: 0, HIDDEN_SHAPE: 0, OUTPUT_26: 4},
    }
    for name, rspec in sensitivity_configs.items():
        for seed in SEEDS_5:
            outdir = Path(f"results/generalisation/emnist_letters/sensitivity/{name}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[sensitivity {name} seed={seed}]")
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rspec, label=name)
            save_result(result, {"experiment": "emnist_letters_sensitivity", "group": name,
                                 "seed": seed, "n_classes": n_classes,
                                 "wall_s": time.time() - t0}, fname)

    # Main configs
    print("\n--- Main configs (26 classes) ---")
    main_configs = {
        "vanilla_r4":  (4, None),
        "lwr_8_2_0":   (None, {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_26: 0}),
        "lwr_8_2_2":   (None, {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_26: 2}),
        "lwr_8_4_0":   (None, {INPUT_SHAPE: 8, HIDDEN_SHAPE: 4, OUTPUT_26: 0}),
        "lwr_0_2_8":   (None, {INPUT_SHAPE: 0, HIDDEN_SHAPE: 2, OUTPUT_26: 8}),
    }
    for name, (rank_int, rank_dict) in main_configs.items():
        for seed in SEEDS_3:
            outdir = Path(f"results/generalisation/emnist_letters/{name}")
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[{name} seed={seed}]")
            t0 = time.time()
            if rank_dict is not None:
                result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                           rank_spec=rank_dict, label=name)
            else:
                result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank_int)

            save_result(result, {"experiment": "emnist_letters", "config": name,
                                 "seed": seed, "n_classes": n_classes,
                                 "wall_s": time.time() - t0}, fname)

    # Reset n_classes
    CFG.n_classes = 10


# ═══════════════════════════════════════════════════════════════
#  SUMMARY
# ═══════════════════════════════════════════════════════════════

def print_summary():
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    def summarise(pattern, label):
        accs = []
        for f in sorted(Path(".").glob(pattern)):
            try:
                d = json.loads(f.read_text())
                if "best_acc" in d:
                    accs.append(d["best_acc"])
                elif "test_acc" in d:
                    accs.append(d["test_acc"])
            except Exception:
                pass
        if accs:
            print(f"  {label}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (n={len(accs)})")

    # Hidden sweep
    print("\n--- Hidden rank sweep (input=8, output=0) ---")
    for h in [0, 1, 2, 4, 8]:
        summarise(f"results/generalisation/hidden_sweep/lwr_8_{h}_0/seed*.json", f"lwr_8_{h}_0")

    # Pop size
    for N in [512, 4096]:
        print(f"\n--- Population N={N} ---")
        summarise(f"results/generalisation/pop_size/N{N}/vanilla_r4/seed*.json", "vanilla_r4")
        summarise(f"results/generalisation/pop_size/N{N}/lwr_8_2_0/seed*.json", "lwr_8_2_0")

    # Transfer
    for ds in ["fashion_mnist", "kmnist"]:
        print(f"\n--- Transfer: {ds} ---")
        for name in ["vanilla_r4", "lwr_8_2_0_transferred", "lwr_0_2_8_reversed"]:
            summarise(f"results/generalisation/transfer/{ds}/{name}/seed*.json", name)

    # EMNIST-Letters
    print("\n--- EMNIST-Letters (26 classes) ---")
    for group in ["input_only", "hidden_only", "output_only"]:
        summarise(f"results/generalisation/emnist_letters/sensitivity/{group}/seed*.json", f"sensitivity/{group}")
    for name in ["vanilla_r4", "lwr_8_2_0", "lwr_8_2_2", "lwr_8_4_0", "lwr_0_2_8"]:
        summarise(f"results/generalisation/emnist_letters/{name}/seed*.json", name)


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    import jax
    print(f"[env] JAX {jax.__version__} on {jax.default_backend()}")
    print(f"[env] Devices: {jax.devices()}")
    print("=" * 60)
    print("  LWR ROBUSTNESS SUITE")
    print("=" * 60)

    experiment_hidden_sweep()      # ~30 min
    experiment_pop_size()          # ~45 min
    experiment_transfer()          # ~60 min
    experiment_emnist_letters()    # ~2.5 hrs

    print_summary()
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()