"""Wide-hidden causal ablation: disentangling sensitivity from parameter count.

Architecture: [784, 128, 1024, 10]
  Dense_0: (128, 784)   = 100,352 params  ← INPUT
  Dense_1: (1024, 128)  = 131,072 params  ← HIDDEN (MORE params than input)
  Dense_2: (10, 1024)   =  10,240 params  ← OUTPUT

If sensitivity were purely about parameter count, hidden should dominate.
If sensitivity reflects functional position, input should still dominate.

Method: Phase A causal ablation.
  Baseline: all layers at rank=4
  Ablated:  drop ONE layer to rank=1, others stay at rank=4
  Measure:  degradation = baseline_accuracy - ablated_accuracy

3 seeds, MNIST only. Quick run (~30-45 min).
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

import json
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# If running from experiments/, adjust path
if not (Path(__file__).resolve().parent / "compare_4_methods_mnist.py").exists():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from compare_4_methods_mnist import (
    CFG, load_mnist, _train_hyperscalees_common,
)
import hyperscalees as hs
from hyperscalees.noiser.lwr_eggroll import LWREggRoll

PROJECT_ROOT = Path(__file__).resolve().parent
# Try to find eggroll-diss root
for p in [PROJECT_ROOT, PROJECT_ROOT.parent, Path.home() / "dissertation" / "eggroll-diss"]:
    if (p / "results").exists() or (p / "experiments").exists():
        PROJECT_ROOT = p
        break

RESULTS_DIR = PROJECT_ROOT / "results" / "wide_hidden_ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Architecture
HIDDEN_DIMS = (128, 1024)
# Shapes as stored by Flax: (out_features, in_features)
INPUT_SHAPE  = (128, 784)    # Dense_0: 100,352 params
HIDDEN_SHAPE = (1024, 128)   # Dense_1: 131,072 params (MORE than input)
OUTPUT_SHAPE = (10, 1024)    # Dense_2:  10,240 params

SEEDS = (0, 1, 2)
LAYER_GROUPS = {
    "input":  INPUT_SHAPE,
    "hidden": HIDDEN_SHAPE,
    "output": OUTPUT_SHAPE,
}

BASELINE_RANK = 4
ABLATED_RANK = 1
SIGMA = 0.05
POP = 2048
WALL = 300.0


def setup_cfg():
    CFG.hidden_dims = HIDDEN_DIMS
    CFG.n_classes = 10
    CFG.eggroll_sigma_init = SIGMA
    CFG.eggroll_pop = POP
    CFG.max_wall_seconds = WALL


def train_with_rank_spec(seed, X_train, y_train, X_test, y_test, rank_spec, label):
    return _train_hyperscalees_common(
        seed, X_train, y_train, X_test, y_test,
        noiser_class=LWREggRoll,
        rank_spec=rank_spec,
        label=label,
    )


def main():
    setup_cfg()
    X_train, y_train, X_test, y_test = load_mnist()

    print("=" * 60)
    print("Wide-Hidden Causal Ablation")
    print(f"Architecture: [784, {', '.join(str(h) for h in HIDDEN_DIMS)}, 10]")
    print(f"Baseline rank: {BASELINE_RANK}, Ablated rank: {ABLATED_RANK}")
    print(f"Seeds: {SEEDS}, σ={SIGMA}, N={POP}, wall={WALL}s")
    print("=" * 60)

    all_results = {}

    # --- Baseline: uniform rank=4 ---
    print("\n--- Baseline: uniform rank=4 ---")
    baseline_accs = []
    uniform_spec = {
        INPUT_SHAPE: BASELINE_RANK,
        HIDDEN_SHAPE: BASELINE_RANK,
        OUTPUT_SHAPE: BASELINE_RANK,
    }
    for seed in SEEDS:
        print(f"  Baseline seed={seed}...")
        t0 = time.time()
        result = train_with_rank_spec(
            seed, X_train, y_train, X_test, y_test,
            rank_spec=uniform_spec,
            label=f"baseline_r{BASELINE_RANK}_seed{seed}",
        )
        elapsed = time.time() - t0
        acc = result["best_test_acc"]
        baseline_accs.append(acc)
        print(f"    acc={acc:.4f}  ({elapsed:.1f}s)")

        run_data = {
            "label": f"baseline_r{BASELINE_RANK}",
            "seed": seed,
            "best_test_acc": acc,
            "wall_seconds": elapsed,
            "rank_spec": {str(k): v for k, v in uniform_spec.items()},
        }
        fname = RESULTS_DIR / f"baseline_r{BASELINE_RANK}_seed{seed}.json"
        with open(fname, "w") as f:
            json.dump(run_data, f, indent=2)

    baseline_mean = float(np.mean(baseline_accs))
    baseline_std = float(np.std(baseline_accs))
    print(f"  Baseline mean: {baseline_mean:.4f} ± {baseline_std:.4f}")

    all_results["baseline"] = {
        "mean": baseline_mean,
        "std": baseline_std,
        "per_seed": baseline_accs,
    }

    # --- Ablation: drop each layer group to rank=1 ---
    for group_name, group_shape in LAYER_GROUPS.items():
        print(f"\n--- Ablate {group_name} ({group_shape}) to rank={ABLATED_RANK} ---")
        ablated_accs = []

        ablated_spec = {
            INPUT_SHAPE: BASELINE_RANK,
            HIDDEN_SHAPE: BASELINE_RANK,
            OUTPUT_SHAPE: BASELINE_RANK,
        }
        ablated_spec[group_shape] = ABLATED_RANK

        for seed in SEEDS:
            print(f"  {group_name} ablated seed={seed}...")
            t0 = time.time()
            result = train_with_rank_spec(
                seed, X_train, y_train, X_test, y_test,
                rank_spec=ablated_spec,
                label=f"ablate_{group_name}_seed{seed}",
            )
            elapsed = time.time() - t0
            acc = result["best_test_acc"]
            ablated_accs.append(acc)
            print(f"    acc={acc:.4f}  ({elapsed:.1f}s)")

            run_data = {
                "label": f"ablate_{group_name}",
                "seed": seed,
                "best_test_acc": acc,
                "wall_seconds": elapsed,
                "rank_spec": {str(k): v for k, v in ablated_spec.items()},
                "ablated_layer": group_name,
                "ablated_shape": list(group_shape),
            }
            fname = RESULTS_DIR / f"ablate_{group_name}_seed{seed}.json"
            with open(fname, "w") as f:
                json.dump(run_data, f, indent=2)

        abl_mean = float(np.mean(ablated_accs))
        abl_std = float(np.std(ablated_accs))
        degradation = baseline_mean - abl_mean
        print(f"  {group_name} mean: {abl_mean:.4f} ± {abl_std:.4f}")
        print(f"  Degradation: {degradation:+.4f}pp")

        all_results[group_name] = {
            "mean": abl_mean,
            "std": abl_std,
            "per_seed": ablated_accs,
            "degradation_from_baseline": degradation,
        }

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Architecture: [784, 128, 1024, 10]")
    print(f"Baseline (uniform r={BASELINE_RANK}): {all_results['baseline']['mean']:.4f} ± {all_results['baseline']['std']:.4f}")
    print()
    print(f"{'Layer':<10} {'Shape':<15} {'Params':<10} {'Ablated':<12} {'Degradation':<12}")
    print("-" * 60)
    param_counts = {
        "input": INPUT_SHAPE[0] * INPUT_SHAPE[1],
        "hidden": HIDDEN_SHAPE[0] * HIDDEN_SHAPE[1],
        "output": OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1],
    }
    for group_name in ["input", "hidden", "output"]:
        r = all_results[group_name]
        shape = LAYER_GROUPS[group_name]
        params = param_counts[group_name]
        print(f"{group_name:<10} {str(shape):<15} {params:<10} {r['mean']:.4f}±{r['std']:.4f} {r['degradation_from_baseline']:+.4f}pp")

    print()
    ordering = sorted(
        ["input", "hidden", "output"],
        key=lambda g: all_results[g]["degradation_from_baseline"],
        reverse=True,
    )
    print(f"Sensitivity ordering: {' > '.join(ordering)}")
    print()
    if ordering[0] == "input":
        print("RESULT: Input layer is most sensitive despite hidden having more parameters.")
        print("        Sensitivity reflects functional position, not parameter count.")
    elif ordering[0] == "hidden":
        print("RESULT: Hidden layer is most sensitive — parameter count may be a confound.")
        print("        Further investigation needed.")
    print("=" * 60)

    # Save summary
    summary = {
        "experiment": "wide_hidden_causal_ablation",
        "architecture": [784, 128, 1024, 10],
        "hidden_dims": list(HIDDEN_DIMS),
        "baseline_rank": BASELINE_RANK,
        "ablated_rank": ABLATED_RANK,
        "sigma": SIGMA,
        "population": POP,
        "wall_seconds": WALL,
        "seeds": list(SEEDS),
        "dataset": "MNIST",
        "results": all_results,
        "sensitivity_ordering": ordering,
        "param_counts": param_counts,
    }
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
