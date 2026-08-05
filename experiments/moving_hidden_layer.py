"""Moving hidden layer — tapered architecture on MNIST.
Tests whether a tapered MLP [784, 512, 256, 128, 10] lets LWR assign
different ranks to different hidden layers (since shapes are now unique).

Phases:
  1. Sensitivity pilot: input_only, hidden1_only, hidden2_only, output_only (n=5)
  2. Vanilla r=4 baseline (n=3)
  3. LWR aligned from pilot (n=3)
  4. LWR reversed control (n=3)
  5. Comparison with uniform-width standard_3h (n=3, reuses existing data)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json
import time
import numpy as np
import jax
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
OUT_DIR = Path("results/moving_hidden_layer")

# Tapered architecture: [784, 512, 256, 128, 10]
# All hidden shapes are unique → LWR can assign per-hidden-layer rank
TAPERED = {
    "hidden_dims": (512, 256, 128),
    "shapes": {
        "input":   (512, 784),   # Dense_0: 784 → 512
        "hidden1": (256, 512),   # Dense_1: 512 → 256
        "hidden2": (128, 256),   # Dense_2: 256 → 128
        "output":  (10, 128),    # Dense_3: 128 → 10
    },
}


def load_mnist():
    train = tv_datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test  = tv_datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32))
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32))
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

    X_train, y_train, X_test, y_test = load_mnist()

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

    for group_name, rspec in pilot_groups.items():
        print(f"\n--- Sensitivity: {group_name} ---")
        print(f"    rank_spec: {rspec}")
        for seed in SEEDS_5:
            fname = OUT_DIR / "sensitivity" / group_name / f"seed{seed}.json"
            if fname.exists():
                print(f"    seed {seed}: already exists, skipping")
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
            acc = result.get("best_test_acc", "?")
            print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 2: VANILLA EGGROLL r=4 (n=3)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2: VANILLA r=4")
    print("=" * 60)

    for seed in SEEDS_3:
        fname = OUT_DIR / "vanilla_r4" / f"seed{seed}.json"
        if fname.exists():
            print(f"    seed {seed}: already exists, skipping")
            continue
        print(f"    seed {seed} ...", end=" ", flush=True)
        t0 = time.time()
        result = train_eggroll(seed, X_train, y_train, X_test, y_test,
                               rank=4)
        wall = time.time() - t0
        save_result(result, {
            "arch": "tapered_512_256_128",
            "experiment": "vanilla_r4",
            "seed": seed,
            "hidden_dims": list(hd),
            "wall_s": wall,
        }, fname)
        acc = result.get("best_test_acc", "?")
        print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 3: LWR ALIGNED (n=3)
    #  Rank allocation based on pilot results:
    #    Expected: input >> hidden1 > hidden2 > output
    #    Allocation: input=8, hidden1=4, hidden2=2, output=0
    #  (If pilot shows different ordering, adjust before running)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 3: LWR ALIGNED")
    print("=" * 60)

    # Default allocation — adjust after seeing pilot results
    lwr_aligned = {inp: 8, hid1: 4, hid2: 2, out: 0}
    print(f"    rank_spec: {lwr_aligned}")
    print(f"    total rank budget: {8+4+2+0} = 14")

    for seed in SEEDS_3:
        fname = OUT_DIR / "lwr_aligned" / f"seed{seed}.json"
        if fname.exists():
            print(f"    seed {seed}: already exists, skipping")
            continue
        print(f"    seed {seed} ...", end=" ", flush=True)
        t0 = time.time()
        result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                   rank_spec=lwr_aligned, label="lwr_aligned")
        wall = time.time() - t0
        save_result(result, {
            "arch": "tapered_512_256_128",
            "experiment": "lwr_aligned",
            "seed": seed,
            "hidden_dims": list(hd),
            "rank_spec": str(lwr_aligned),
            "wall_s": wall,
        }, fname)
        acc = result.get("best_test_acc", "?")
        print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ═══════════════════════════════════════════════════════════
    #  PHASE 4: LWR REVERSED (n=3)
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 4: LWR REVERSED")
    print("=" * 60)

    lwr_reversed = {inp: 0, hid1: 2, hid2: 4, out: 8}
    print(f"    rank_spec: {lwr_reversed}")

    for seed in SEEDS_3:
        fname = OUT_DIR / "lwr_reversed" / f"seed{seed}.json"
        if fname.exists():
            print(f"    seed {seed}: already exists, skipping")
            continue
        print(f"    seed {seed} ...", end=" ", flush=True)
        t0 = time.time()
        result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                   rank_spec=lwr_reversed, label="lwr_reversed")
        wall = time.time() - t0
        save_result(result, {
            "arch": "tapered_512_256_128",
            "experiment": "lwr_reversed",
            "seed": seed,
            "hidden_dims": list(hd),
            "rank_spec": str(lwr_reversed),
            "wall_s": wall,
        }, fname)
        acc = result.get("best_test_acc", "?")
        print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ═══════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for phase in ["sensitivity", "vanilla_r4", "lwr_aligned", "lwr_reversed"]:
        phase_dir = OUT_DIR / phase
        if not phase_dir.exists():
            continue
        # Collect all results from subdirectories or direct files
        json_files = list(phase_dir.rglob("*.json"))
        if not json_files:
            continue
        accs = []
        for f in json_files:
            d = json.load(open(f))
            accs.append(d.get("best_test_acc", 0))
        if accs:
            print(f"  {phase:20s}: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%  (n={len(accs)})")


if __name__ == "__main__":
    main()
