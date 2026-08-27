"""Extended overnight suite — 7 hours of experiments.

Exp 1: Four-method baseline on Fashion-MNIST & KMNIST (~2 hrs)
Exp 2: Sigma x LWR interaction on MNIST (~1.5 hrs)
Exp 3: Wall-clock budget sweep on EMNIST-Digits (~2 hrs)
Exp 4: Input rank sweep — input in {1,2,4,8}, hidden=2, output=0 (~30 min)
Exp 5: Output rank sweep — input=8, hidden=2, output in {0,1,2,4,8} (~30 min)
Exp 6: Budget-matched comparison — total=12, different allocations (~30 min)
Exp 7: Per-generation wall-clock cost vs rank (~15 min)
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
    train_backprop,
    train_openai_es,
    train_sep_cma_es,
)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEEDS_3 = (0, 1, 2)

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
    print(f"[saved] {fname}")


def setup_cfg(sigma=0.05, pop=2048, wall=300.0):
    CFG.eggroll_sigma_init = sigma
    CFG.eggroll_pop = pop
    CFG.max_wall_seconds = wall
    CFG.hidden_dims = (256, 256, 256)
    CFG.n_classes = 10


def load_fashion_mnist():
    train = tv_datasets.FashionMNIST(root=str(DATA_DIR), train=True, download=True)
    test  = tv_datasets.FashionMNIST(root=str(DATA_DIR), train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32))
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32))
    print(f"[data] Fashion-MNIST loaded: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


def load_kmnist():
    train = tv_datasets.KMNIST(root=str(DATA_DIR), train=True, download=True)
    test  = tv_datasets.KMNIST(root=str(DATA_DIR), train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32))
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32))
    print(f"[data] KMNIST loaded: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


def load_mnist():
    train = tv_datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test  = tv_datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32))
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32))
    print(f"[data] MNIST loaded: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


def load_emnist_digits():
    train = tv_datasets.EMNIST(root=str(DATA_DIR), split="digits", train=True, download=True)
    test  = tv_datasets.EMNIST(root=str(DATA_DIR), split="digits", train=False, download=True)
    X_train = jnp.asarray(np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np.array(train.targets, dtype=np.int32))
    X_test  = jnp.asarray(np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0)
    y_test  = jnp.asarray(np.array(test.targets, dtype=np.int32))
    print(f"[data] EMNIST-Digits loaded: train={X_train.shape[0]}, test={X_test.shape[0]}")
    return X_train, y_train, X_test, y_test


# ═══════════════════════════════════════════════════════════════
#  EXP 1: FOUR-METHOD BASELINE ON FASHION-MNIST & KMNIST
# ═══════════════════════════════════════════════════════════════

def exp1_four_method():
    print("\n" + "=" * 60)
    print("  EXP 1: FOUR-METHOD BASELINE (Fashion-MNIST & KMNIST)")
    print("=" * 60)

    setup_cfg()

    for ds_name, loader in [("fashion_mnist", load_fashion_mnist),
                             ("kmnist", load_kmnist)]:
        print(f"\n--- {ds_name} ---")
        X_train, y_train, X_test, y_test = loader()
        base = Path(f"results/four_method/{ds_name}")

        methods = {
            "backprop":   ("bp", train_backprop),
            "openai_es":  ("es", train_openai_es),
            "sep_cma_es": ("cma", train_sep_cma_es),
            "eggroll_r4": ("egg", None),
        }

        for method_name, (tag, train_fn) in methods.items():
            for seed in SEEDS_3:
                fname = base / f"{method_name}/seed{seed}.json"
                if fname.exists():
                    print(f"[skip] {fname}"); continue

                print(f"\n[{method_name} {ds_name} seed={seed}]")
                t0 = time.time()

                if method_name == "eggroll_r4":
                    result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=4)
                else:
                    result = train_fn(seed, X_train, y_train, X_test, y_test)

                save_result(result, {"experiment": "four_method", "dataset": ds_name,
                                     "method": method_name, "seed": seed,
                                     "wall_s": time.time() - t0}, fname)


# ═══════════════════════════════════════════════════════════════
#  EXP 2: SIGMA x LWR INTERACTION
# ═══════════════════════════════════════════════════════════════

def exp2_sigma_lwr():
    print("\n" + "=" * 60)
    print("  EXP 2: SIGMA x LWR INTERACTION (MNIST)")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    lwr_820 = {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 0}

    for sigma in [0.01, 0.03, 0.05, 0.1]:
        print(f"\n--- sigma={sigma} ---")
        setup_cfg(sigma=sigma)

        # Vanilla r=4
        for seed in SEEDS_3:
            fname = Path(f"results/sigma_lwr/sigma{sigma}/vanilla_r4/seed{seed}.json")
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[vanilla_r4 sigma={sigma} seed={seed}]")
            t0 = time.time()
            result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=4)
            save_result(result, {"experiment": "sigma_lwr", "sigma": sigma,
                                 "config": "vanilla_r4", "seed": seed,
                                 "wall_s": time.time() - t0}, fname)

        # LWR (8,2,0)
        for seed in SEEDS_3:
            fname = Path(f"results/sigma_lwr/sigma{sigma}/lwr_8_2_0/seed{seed}.json")
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[lwr_8_2_0 sigma={sigma} seed={seed}]")
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=lwr_820, label=f"lwr_8_2_0_s{sigma}")
            save_result(result, {"experiment": "sigma_lwr", "sigma": sigma,
                                 "config": "lwr_8_2_0", "seed": seed,
                                 "wall_s": time.time() - t0}, fname)

    setup_cfg()  # reset


# ═══════════════════════════════════════════════════════════════
#  EXP 3: WALL-CLOCK BUDGET SWEEP ON EMNIST-DIGITS
# ═══════════════════════════════════════════════════════════════

def exp3_wall_budget_emnist():
    print("\n" + "=" * 60)
    print("  EXP 3: WALL-CLOCK BUDGET SWEEP (EMNIST-Digits)")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_emnist_digits()

    lwr_820 = {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 0}
    lwr_840 = {INPUT_SHAPE: 8, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 0}

    configs = {
        "vanilla_r4":  (4, None),
        "lwr_8_2_0":   (None, lwr_820),
        "lwr_8_4_0":   (None, lwr_840),
    }

    for budget in [60.0, 120.0, 300.0]:
        print(f"\n--- budget={budget}s ---")

        for name, (rank_int, rank_dict) in configs.items():
            setup_cfg(wall=budget)

            for seed in SEEDS_3:
                fname = Path(f"results/wall_budget_emnist/budget{int(budget)}s/{name}/seed{seed}.json")
                if fname.exists():
                    print(f"[skip] {fname}"); continue

                print(f"\n[{name} budget={budget}s seed={seed}]")
                t0 = time.time()
                if rank_dict is not None:
                    result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                               rank_spec=rank_dict, label=name)
                else:
                    result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank_int)

                save_result(result, {"experiment": "wall_budget_emnist", "budget_s": budget,
                                     "config": name, "seed": seed,
                                     "wall_s": time.time() - t0}, fname)

    setup_cfg()  # reset


# ═══════════════════════════════════════════════════════════════
#  EXP 4: INPUT RANK SWEEP
#  Fix hidden=2, output=0, sweep input in {1,2,4,8}
# ═══════════════════════════════════════════════════════════════

def exp4_input_rank_sweep():
    print("\n" + "=" * 60)
    print("  EXP 4: INPUT RANK SWEEP (MNIST)")
    print("  Fix hidden=2, output=0, sweep input in {1,2,4,8}")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    setup_cfg()

    for input_r in [1, 2, 4, 8]:
        rank_dict = {INPUT_SHAPE: input_r, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 0}
        label = f"lwr_{input_r}_2_0"
        print(f"\n--- {label} ---")

        for seed in SEEDS_3:
            fname = Path(f"results/rank_study/input_sweep/{label}/seed{seed}.json")
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[{label} seed={seed}]")
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rank_dict, label=label)
            save_result(result, {"experiment": "input_rank_sweep", "input_rank": input_r,
                                 "seed": seed, "wall_s": time.time() - t0}, fname)


# ═══════════════════════════════════════════════════════════════
#  EXP 5: OUTPUT RANK SWEEP
#  Fix input=8, hidden=2, sweep output in {0,1,2,4,8}
# ═══════════════════════════════════════════════════════════════

def exp5_output_rank_sweep():
    print("\n" + "=" * 60)
    print("  EXP 5: OUTPUT RANK SWEEP (MNIST)")
    print("  Fix input=8, hidden=2, sweep output in {0,1,2,4,8}")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    setup_cfg()

    for output_r in [0, 1, 2, 4, 8]:
        rank_dict = {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: output_r}
        label = f"lwr_8_2_{output_r}"
        print(f"\n--- {label} ---")

        for seed in SEEDS_3:
            fname = Path(f"results/rank_study/output_sweep/{label}/seed{seed}.json")
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[{label} seed={seed}]")
            t0 = time.time()
            result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                       rank_spec=rank_dict, label=label)
            save_result(result, {"experiment": "output_rank_sweep", "output_rank": output_r,
                                 "seed": seed, "wall_s": time.time() - t0}, fname)


# ═══════════════════════════════════════════════════════════════
#  EXP 6: BUDGET-MATCHED COMPARISON
#  All configs have total rank = 12, different allocations
# ═══════════════════════════════════════════════════════════════

def exp6_budget_matched():
    print("\n" + "=" * 60)
    print("  EXP 6: BUDGET-MATCHED COMPARISON (MNIST)")
    print("  Total rank = 12 for all configs")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    setup_cfg()

    # All have total rank = 12 across 4 layers (input + 2 hidden + output)
    # vanilla r=4: 4+4+4 = 12 (but only 3 unique shapes, so 4+4+4 = 12)
    # Note: 3 hidden layers share shape, so total = input + hidden*2 + output
    # Actually with 3 hidden layers: input + 3*hidden + output
    # vanilla r=4: 4 + 4 + 4 + 4 = 16 total (4 per layer, but per-shape)
    # Per-shape: (256,784)=4, (256,256)=4, (10,256)=4 → covers all layers
    # Total unique-shape budget: 4+4+4 = 12

    configs = {
        "vanilla_r4":         4,                                                    # 4+4+4 = 12
        "lwr_8_4_0":          {INPUT_SHAPE: 8, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 0},   # 8+4+0 = 12
        "lwr_4_4_4":          {INPUT_SHAPE: 4, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 4},   # 4+4+4 = 12
        "lwr_0_4_8":          {INPUT_SHAPE: 0, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 8},   # 0+4+8 = 12
        "lwr_8_2_2":          {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 2},   # 8+2+2 = 12
    }

    for name, rank_spec in configs.items():
        print(f"\n--- {name} ---")
        for seed in SEEDS_3:
            fname = Path(f"results/rank_study/budget_matched/{name}/seed{seed}.json")
            if fname.exists():
                print(f"[skip] {fname}"); continue

            print(f"\n[{name} seed={seed}]")
            t0 = time.time()
            if isinstance(rank_spec, int):
                result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank_spec)
            else:
                result = train_lwr_eggroll(seed, X_train, y_train, X_test, y_test,
                                           rank_spec=rank_spec, label=name)

            save_result(result, {"experiment": "budget_matched", "config": name,
                                 "seed": seed, "wall_s": time.time() - t0}, fname)


# ═══════════════════════════════════════════════════════════════
#  EXP 7: PER-GENERATION WALL-CLOCK COST VS RANK
# ═══════════════════════════════════════════════════════════════

def exp7_timing():
    print("\n" + "=" * 60)
    print("  EXP 7: PER-GENERATION COST VS RANK (MNIST)")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist()
    setup_cfg(wall=60.0)  # short runs, just timing

    timing_results = {}

    for rank in [1, 2, 4, 8, 16, 32]:
        print(f"\n--- vanilla rank={rank} ---")
        t0 = time.time()
        result = train_eggroll(0, X_train, y_train, X_test, y_test, rank=rank)
        total_wall = time.time() - t0
        hist = result.get("history", [])
        n_gens = len(hist) * 20 if hist else 0  # logged every 20 gens
        cost_per_gen = total_wall / max(n_gens, 1)
        timing_results[f"vanilla_r{rank}"] = {
            "rank": rank, "total_wall": total_wall,
            "n_gens": n_gens, "cost_per_gen_ms": cost_per_gen * 1000,
        }
        print(f"  rank={rank}: {n_gens} gens in {total_wall:.1f}s = {cost_per_gen*1000:.1f}ms/gen")

    # LWR configs
    lwr_configs = {
        "lwr_8_2_0": {INPUT_SHAPE: 8, HIDDEN_SHAPE: 2, OUTPUT_SHAPE: 0},
        "lwr_8_4_0": {INPUT_SHAPE: 8, HIDDEN_SHAPE: 4, OUTPUT_SHAPE: 0},
        "lwr_8_0_0": {INPUT_SHAPE: 8, HIDDEN_SHAPE: 0, OUTPUT_SHAPE: 0},
    }
    for name, rspec in lwr_configs.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        result = train_lwr_eggroll(0, X_train, y_train, X_test, y_test,
                                   rank_spec=rspec, label=name)
        total_wall = time.time() - t0
        hist = result.get("history", [])
        n_gens = len(hist) * 20 if hist else 0
        cost_per_gen = total_wall / max(n_gens, 1)
        timing_results[name] = {
            "config": name, "total_wall": total_wall,
            "n_gens": n_gens, "cost_per_gen_ms": cost_per_gen * 1000,
        }
        print(f"  {name}: {n_gens} gens in {total_wall:.1f}s = {cost_per_gen*1000:.1f}ms/gen")

    # Save timing results
    fname = Path("results/rank_study/timing/timing_results.json")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fname.write_text(json.dumps(timing_results, indent=2))
    print(f"\n[saved] {fname}")

    setup_cfg()  # reset


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
                hist = d.get("history", [])
                if hist:
                    best = max(h[2] for h in hist) if isinstance(hist[0], list) else 0
                    accs.append(best)
                elif "best_acc" in d:
                    accs.append(d["best_acc"])
                elif "test_acc" in d:
                    accs.append(d["test_acc"])
            except Exception:
                pass
        if accs:
            print(f"  {label}: {np.mean(accs):.4f} +/- {np.std(accs):.4f}  (n={len(accs)})")

    # Exp 1
    for ds in ["fashion_mnist", "kmnist"]:
        print(f"\n--- Four-method: {ds} ---")
        for m in ["backprop", "openai_es", "sep_cma_es", "eggroll_r4"]:
            summarise(f"results/four_method/{ds}/{m}/seed*.json", m)

    # Exp 2
    print("\n--- Sigma x LWR ---")
    for sigma in [0.01, 0.03, 0.05, 0.1]:
        for cfg in ["vanilla_r4", "lwr_8_2_0"]:
            summarise(f"results/sigma_lwr/sigma{sigma}/{cfg}/seed*.json", f"s={sigma} {cfg}")

    # Exp 3
    print("\n--- Wall budget EMNIST ---")
    for budget in [60, 120, 300]:
        for cfg in ["vanilla_r4", "lwr_8_2_0", "lwr_8_4_0"]:
            summarise(f"results/wall_budget_emnist/budget{budget}s/{cfg}/seed*.json", f"{budget}s {cfg}")

    # Exp 4
    print("\n--- Input rank sweep ---")
    for r in [1, 2, 4, 8]:
        summarise(f"results/rank_study/input_sweep/lwr_{r}_2_0/seed*.json", f"lwr_{r}_2_0")

    # Exp 5
    print("\n--- Output rank sweep ---")
    for r in [0, 1, 2, 4, 8]:
        summarise(f"results/rank_study/output_sweep/lwr_8_2_{r}/seed*.json", f"lwr_8_2_{r}")

    # Exp 6
    print("\n--- Budget-matched (total=12) ---")
    for cfg in ["vanilla_r4", "lwr_8_4_0", "lwr_4_4_4", "lwr_0_4_8", "lwr_8_2_2"]:
        summarise(f"results/rank_study/budget_matched/{cfg}/seed*.json", cfg)

    # Exp 7
    timing_file = Path("results/rank_study/timing/timing_results.json")
    if timing_file.exists():
        print("\n--- Per-gen timing ---")
        t = json.loads(timing_file.read_text())
        for name, data in t.items():
            print(f"  {name}: {data.get('cost_per_gen_ms', 0):.1f} ms/gen")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print(f"[env] JAX {jax.__version__} on {jax.default_backend()}")
    print(f"[env] Devices: {jax.devices()}")
    print("=" * 60)
    print("  EXTENDED OVERNIGHT SUITE")
    print("=" * 60)

    # Quick experiments first (rank study)
    exp4_input_rank_sweep()       # ~30 min
    exp5_output_rank_sweep()      # ~30 min
    exp6_budget_matched()         # ~30 min
    exp7_timing()                 # ~15 min

    # Medium experiments
    exp2_sigma_lwr()              # ~1.5 hrs

    # Longer experiments
    exp1_four_method()            # ~2 hrs
    exp3_wall_budget_emnist()     # ~2 hrs

    print_summary()
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
