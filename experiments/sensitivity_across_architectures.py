"""Sensitivity across architectures — MNIST.
Tests whether input >> hidden > output holds across different depths/widths.

Three architectures:
  narrow_2h:    [784, 128, 128, 10]          ~118K params
  standard_3h:  [784, 256, 256, 256, 10]     ~335K params (baseline)
  deep_4h:      [784, 256, 256, 256, 256, 10] ~401K params
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


# ═══════════════════════════════════════════════════════════════
#  ARCHITECTURE DEFINITIONS
#  Shapes are TRANSPOSED as HyperscaleES stores them: (out, in)
# ═══════════════════════════════════════════════════════════════

ARCHS = {
    "narrow_2h": {
        "hidden_dims": (128, 128),
        "input_shape":  (128, 784),   # Dense_0: 784 -> 128
        "hidden_shape": (128, 128),   # Dense_1: 128 -> 128
        "output_shape": (10, 128),    # Dense_2: 128 -> 10
    },
    "standard_3h": {
        "hidden_dims": (256, 256, 256),
        "input_shape":  (256, 784),   # Dense_0: 784 -> 256
        "hidden_shape": (256, 256),   # Dense_1,2: 256 -> 256
        "output_shape": (10, 256),    # Dense_3: 256 -> 10
    },
    "deep_4h": {
        "hidden_dims": (256, 256, 256, 256),
        "input_shape":  (256, 784),   # Dense_0: 784 -> 256
        "hidden_shape": (256, 256),   # Dense_1,2,3: 256 -> 256
        "output_shape": (10, 256),    # Dense_4: 256 -> 10
    },
}


def main():
    print(f"[env] JAX {jax.__version__} on {jax.default_backend()}")
    print(f"[env] Devices: {jax.devices()}")
    print("=" * 60)
    print("  SENSITIVITY ACROSS ARCHITECTURES")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()

    for arch_name, arch in ARCHS.items():
        hd = arch["hidden_dims"]
        inp = arch["input_shape"]
        hid = arch["hidden_shape"]
        out = arch["output_shape"]

        print(f"\n{'=' * 60}")
        print(f"  ARCHITECTURE: {arch_name}  dims={[784]+list(hd)+[10]}")
        print(f"  Shapes: input={inp}, hidden={hid}, output={out}")
        print(f"{'=' * 60}")

        setup_cfg(hidden_dims=hd)

        base_dir = Path(f"results/arch_variation/{arch_name}")

        # ── 1. Sensitivity pilot ─────────────────────────────
        print(f"\n--- Sensitivity pilot: {arch_name} ---")
        sensitivity_configs = {
            "input_only":  {inp: 4, hid: 0, out: 0},
            "hidden_only": {inp: 0, hid: 4, out: 0},
            "output_only": {inp: 0, hid: 0, out: 4},
        }
        for group_name, rspec in sensitivity_configs.items():
            for seed in SEEDS_5:
                outdir = base_dir / f"sensitivity/{group_name}"
                outdir.mkdir(parents=True, exist_ok=True)
                fname = outdir / f"seed{seed}.json"
                if fname.exists():
                    print(f"[skip] {fname}"); continue

                print(f"\n[sensitivity {group_name} seed={seed}]")
                t0 = time.time()
                result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                           rank_spec=rspec, label=group_name)
                save_result(result, {"arch": arch_name, "experiment": "sensitivity",
                                     "group": group_name, "seed": seed,
                                     "hidden_dims": list(hd),
                                     "wall_s": time.time() - t0}, fname)

        # ── 2. Vanilla EGGROLL r=4 ──────────────────────────
        print(f"\n--- Vanilla r=4: {arch_name} ---")
        for seed in SEEDS_3:
            outdir = base_dir / "vanilla_r4"
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[vanilla_r4 {arch_name} seed={seed}]")
            t0 = time.time()
            result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=4)
            save_result(result, {"arch": arch_name, "experiment": "vanilla",
                                 "seed": seed, "hidden_dims": list(hd),
                                 "wall_s": time.time() - t0}, fname)

        # ── 3. LWR aligned (input=8, hidden=2, output=0) ────
        print(f"\n--- LWR aligned: {arch_name} ---")
        lwr_aligned = {inp: 8, hid: 2, out: 0}
        for seed in SEEDS_3:
            outdir = base_dir / "lwr_aligned"
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[lwr_aligned {arch_name} seed={seed}]")
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=lwr_aligned, label="lwr_aligned")
            save_result(result, {"arch": arch_name, "experiment": "lwr_aligned",
                                 "seed": seed, "hidden_dims": list(hd),
                                 "rank_dict": str(lwr_aligned),
                                 "wall_s": time.time() - t0}, fname)

        # ── 4. LWR reversed (input=0, hidden=2, output=8) ───
        print(f"\n--- LWR reversed: {arch_name} ---")
        lwr_reversed = {inp: 0, hid: 2, out: 8}
        for seed in SEEDS_3:
            outdir = base_dir / "lwr_reversed"
            outdir.mkdir(parents=True, exist_ok=True)
            fname = outdir / f"seed{seed}.json"
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[lwr_reversed {arch_name} seed={seed}]")
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=lwr_reversed, label="lwr_reversed")
            save_result(result, {"arch": arch_name, "experiment": "lwr_reversed",
                                 "seed": seed, "hidden_dims": list(hd),
                                 "rank_dict": str(lwr_reversed),
                                 "wall_s": time.time() - t0}, fname)

    # ═══════════════════════════════════════════════════════════
    #  SUMMARY
    # ═══════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for arch_name in ARCHS:
        print(f"\n--- {arch_name} ---")
        base = Path(f"results/arch_variation/{arch_name}")

        for group in ["input_only", "hidden_only", "output_only"]:
            accs = []
            for f in sorted(base.glob(f"sensitivity/{group}/seed*.json")):
                try:
                    d = json.loads(f.read_text())
                    accs.append(d.get("best_acc", d.get("test_acc", 0)))
                except Exception:
                    pass
            if accs:
                print(f"  sensitivity/{group}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

        for config in ["vanilla_r4", "lwr_aligned", "lwr_reversed"]:
            accs = []
            for f in sorted(base.glob(f"{config}/seed*.json")):
                try:
                    d = json.loads(f.read_text())
                    accs.append(d.get("best_acc", d.get("test_acc", 0)))
                except Exception:
                    pass
            if accs:
                print(f"  {config}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")

    # Reset CFG to default
    setup_cfg(hidden_dims=(256, 256, 256))

    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()