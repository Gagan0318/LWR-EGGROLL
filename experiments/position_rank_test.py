"""Position-based rank test via unique hidden widths.
Architecture: [784, 257, 256, 255, 10] — all shapes unique.
Tests whether per-position rank allocation on a near-uniform architecture
outperforms shape-constrained uniform hidden rank.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import json, time
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
OUT_DIR = Path("results/position_rank_test")

# Near-uniform architecture with unique shapes
HIDDEN_DIMS = (257, 256, 255)

# HyperscaleES stores transposed: (out_dim, in_dim)
SHAPES = {
    "input":   (257, 784),   # Dense_0
    "hidden1": (256, 257),   # Dense_1
    "hidden2": (255, 256),   # Dense_2
    "output":  (10, 255),    # Dense_3
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
    print("  POSITION-BASED RANK TEST")
    print("  Architecture: [784, 257, 256, 255, 10]")
    print("  All shapes unique → per-position rank allocation")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    setup_cfg(HIDDEN_DIMS)

    inp  = SHAPES["input"]
    hid1 = SHAPES["hidden1"]
    hid2 = SHAPES["hidden2"]
    out  = SHAPES["output"]

    # ══════════════════════════════════════════════════════════
    #  PHASE 1: SENSITIVITY PILOT (n=5)
    # ══════════════════════════════════════════════════════════
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
        for seed in SEEDS_5:
            fname = OUT_DIR / "sensitivity" / group_name / f"seed{seed}.json"
            if fname.exists():
                print(f"    seed {seed}: exists, skipping")
                continue
            print(f"    seed {seed} ...", end=" ", flush=True)
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rspec, label=group_name)
            wall = time.time() - t0
            save_result(result, {
                "arch": "position_257_256_255",
                "experiment": "sensitivity",
                "group": group_name, "seed": seed,
                "hidden_dims": list(HIDDEN_DIMS),
                "rank_spec": str(rspec), "wall_s": wall,
            }, fname)
            acc = result.get("best_test_acc", "?")
            print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  PHASE 2: VANILLA r=4 BASELINE (n=3)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 2: VANILLA r=4")
    print("=" * 60)

    for seed in SEEDS_3:
        fname = OUT_DIR / "vanilla_r4" / f"seed{seed}.json"
        if fname.exists():
            print(f"    seed {seed}: exists, skipping")
            continue
        print(f"    seed {seed} ...", end=" ", flush=True)
        t0 = time.time()
        result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=4)
        wall = time.time() - t0
        save_result(result, {
            "arch": "position_257_256_255",
            "experiment": "vanilla_r4",
            "seed": seed, "hidden_dims": list(HIDDEN_DIMS),
            "wall_s": wall,
        }, fname)
        acc = result.get("best_test_acc", "?")
        print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  PHASE 3: LWR SHAPE-BASED (n=3)
    #  Same allocation for hidden1 and hidden2 (r=2 each)
    #  This is what current LWR would do on uniform [256,256,256]
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 3: LWR SHAPE-BASED (hidden1=hidden2=2)")
    print("=" * 60)

    lwr_shape = {inp: 8, hid1: 2, hid2: 2, out: 0}
    print(f"    rank_spec: {lwr_shape}")

    for seed in SEEDS_3:
        fname = OUT_DIR / "lwr_shape_based" / f"seed{seed}.json"
        if fname.exists():
            print(f"    seed {seed}: exists, skipping")
            continue
        print(f"    seed {seed} ...", end=" ", flush=True)
        t0 = time.time()
        result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                   rank_spec=lwr_shape, label="lwr_shape_based")
        wall = time.time() - t0
        save_result(result, {
            "arch": "position_257_256_255",
            "experiment": "lwr_shape_based",
            "seed": seed, "hidden_dims": list(HIDDEN_DIMS),
            "rank_spec": str(lwr_shape), "wall_s": wall,
        }, fname)
        acc = result.get("best_test_acc", "?")
        print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  PHASE 4: LWR POSITION-BASED (n=3)
    #  Different rank for hidden1 vs hidden2 based on position
    #  hidden1 (closer to input) gets more rank than hidden2
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 4: LWR POSITION-BASED (hidden1=4, hidden2=2)")
    print("=" * 60)

    lwr_position = {inp: 8, hid1: 4, hid2: 2, out: 0}
    print(f"    rank_spec: {lwr_position}")

    for seed in SEEDS_3:
        fname = OUT_DIR / "lwr_position_based" / f"seed{seed}.json"
        if fname.exists():
            print(f"    seed {seed}: exists, skipping")
            continue
        print(f"    seed {seed} ...", end=" ", flush=True)
        t0 = time.time()
        result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                   rank_spec=lwr_position, label="lwr_position_based")
        wall = time.time() - t0
        save_result(result, {
            "arch": "position_257_256_255",
            "experiment": "lwr_position_based",
            "seed": seed, "hidden_dims": list(HIDDEN_DIMS),
            "rank_spec": str(lwr_position), "wall_s": wall,
        }, fname)
        acc = result.get("best_test_acc", "?")
        print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  PHASE 5: LWR REVERSED (n=3)
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  PHASE 5: LWR REVERSED")
    print("=" * 60)

    lwr_reversed = {inp: 0, hid1: 2, hid2: 4, out: 8}
    print(f"    rank_spec: {lwr_reversed}")

    for seed in SEEDS_3:
        fname = OUT_DIR / "lwr_reversed" / f"seed{seed}.json"
        if fname.exists():
            print(f"    seed {seed}: exists, skipping")
            continue
        print(f"    seed {seed} ...", end=" ", flush=True)
        t0 = time.time()
        result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                   rank_spec=lwr_reversed, label="lwr_reversed")
        wall = time.time() - t0
        save_result(result, {
            "arch": "position_257_256_255",
            "experiment": "lwr_reversed",
            "seed": seed, "hidden_dims": list(HIDDEN_DIMS),
            "rank_spec": str(lwr_reversed), "wall_s": wall,
        }, fname)
        acc = result.get("best_test_acc", "?")
        print(f"acc={acc:.4f}  wall={wall:.1f}s")

    # ══════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for phase in ["sensitivity", "vanilla_r4", "lwr_shape_based",
                   "lwr_position_based", "lwr_reversed"]:
        phase_dir = OUT_DIR / phase
        if not phase_dir.exists():
            continue
        json_files = list(phase_dir.rglob("*.json"))
        if not json_files:
            continue
        accs = [json.load(open(f)).get("best_test_acc", 0) for f in json_files]
        if accs:
            print(f"  {phase:22s}: {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%  (n={len(accs)})")


if __name__ == "__main__":
    main()
