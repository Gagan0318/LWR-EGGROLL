"""Overnight run: corrected Phase 1 pilots with 5 seeds.

1. Standard MLP [784, 256, 256, 256, 10] — 5 seeds (3 exist, 2 new)
2. Tapered MLP [784, 512, 256, 128, 10] — 5 seeds (all new or partial)
"""

import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.compare_4_methods_mnist import load_mnist, train_lwr_eggroll, CFG
from experiments.moving_hidden_layer import setup_cfg

SEEDS = [0, 1, 2, 3, 4]
ELEVATED_RANK = 8
BACKGROUND_RANK = 1


def run_condition(cond_name, rank_spec, seeds, results_dir, X_train, y_train, X_test, y_test):
    print(f"\n{'='*60}")
    print(f"CONDITION: {cond_name}")
    print(f"Rank spec: {rank_spec}")
    print(f"{'='*60}")

    cond_results = []
    for seed in seeds:
        out_file = results_dir / f"{cond_name}_seed{seed}.json"

        if out_file.exists():
            print(f"  seed={seed} — already exists, loading.")
            with open(out_file) as f:
                result = json.load(f)
        else:
            print(f"  seed={seed} — running...", flush=True)
            t0 = time.time()
            result = train_lwr_eggroll(
                seed=seed,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                rank_spec=rank_spec,
                label=f"{cond_name}",
            )
            elapsed = time.time() - t0
            print(f"    acc={result['best_test_acc']:.4f}  ({elapsed:.1f}s)")

            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

        cond_results.append(result)
    return cond_results


def summarise(all_results, title, results_dir):
    print(f"\n{'='*60}")
    print(title)
    print(f"{'='*60}")

    print(f"\n{'Layer':<28} {'Accuracy':>18} {'Fitness Var':>22}")
    print("-" * 68)

    sorted_by_fvar = sorted(
        all_results.items(),
        key=lambda x: x[1]["mean_fitness_var"] if x[1]["mean_fitness_var"] else 0,
        reverse=True,
    )
    for cond_name, stats in sorted_by_fvar:
        acc_str = f"{stats['mean_acc']:.4f} \u00b1 {stats['std_acc']:.4f}"
        if stats["mean_fitness_var"] is not None:
            fvar_str = f"{stats['mean_fitness_var']:.2f} \u00b1 {stats['std_fitness_var']:.2f}"
        else:
            fvar_str = "N/A"
        print(f"{cond_name:<28} {acc_str:>18} {fvar_str:>22}")

    ordering = [name.replace("_elevated", "") for name, _ in sorted_by_fvar]
    print(f"\nSensitivity ordering (by fitness variance): {' >> '.join(ordering)}")

    with open(results_dir / "summary_5seed.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {results_dir}/summary_5seed.json")


def compute_stats(cond_results):
    accs = [r["best_test_acc"] for r in cond_results]
    mean_acc = float(np.mean(accs))
    std_acc = float(np.std(accs))

    if "fitness_variance_history" in cond_results[0]:
        fvars = [float(np.mean(r["fitness_variance_history"])) for r in cond_results]
        mean_fvar = float(np.mean(fvars))
        std_fvar = float(np.std(fvars))
    else:
        mean_fvar = None
        std_fvar = None

    return {
        "accs": accs,
        "mean_acc": mean_acc,
        "std_acc": std_acc,
        "mean_fitness_var": mean_fvar,
        "std_fitness_var": std_fvar,
    }


def main():
    X_train, y_train, X_test, y_test = load_mnist()

    # ── PART 1: Standard MLP [784, 256, 256, 256, 10] ───────────
    print("\n" + "#" * 60)
    print("# PART 1: STANDARD MLP — 5 SEEDS")
    print("#" * 60)

    # Reset CFG to standard architecture
    CFG.hidden_dims = (256, 256, 256)

    std_dir = Path("results/corrected_pilot_phase1")
    std_dir.mkdir(parents=True, exist_ok=True)

    STANDARD_CONDITIONS = {
        "input_elevated": {
            (256, 784): ELEVATED_RANK,
            (256, 256): BACKGROUND_RANK,
            (10, 256):  BACKGROUND_RANK,
        },
        "hidden_elevated": {
            (256, 784): BACKGROUND_RANK,
            (256, 256): ELEVATED_RANK,
            (10, 256):  BACKGROUND_RANK,
        },
        "output_elevated": {
            (256, 784): BACKGROUND_RANK,
            (256, 256): BACKGROUND_RANK,
            (10, 256):  ELEVATED_RANK,
        },
    }

    std_results = {}
    for cond_name, rank_spec in STANDARD_CONDITIONS.items():
        cond_results = run_condition(
            cond_name, rank_spec, SEEDS, std_dir,
            X_train, y_train, X_test, y_test,
        )
        std_results[cond_name] = compute_stats(cond_results)

    summarise(std_results, "STANDARD MLP — CORRECTED PHASE 1 (5 SEEDS)", std_dir)

    # ── PART 2: Tapered MLP [784, 512, 256, 128, 10] ────────────
    print("\n" + "#" * 60)
    print("# PART 2: TAPERED MLP — 5 SEEDS")
    print("#" * 60)

    setup_cfg((512, 256, 128))

    tap_dir = Path("results/corrected_pilot_phase1_tapered")
    tap_dir.mkdir(parents=True, exist_ok=True)

    TAPERED_CONDITIONS = {
        "tapered_input_elevated": {
            (512, 784): ELEVATED_RANK,
            (256, 512): BACKGROUND_RANK,
            (128, 256): BACKGROUND_RANK,
            (10, 128):  BACKGROUND_RANK,
        },
        "tapered_hidden1_elevated": {
            (512, 784): BACKGROUND_RANK,
            (256, 512): ELEVATED_RANK,
            (128, 256): BACKGROUND_RANK,
            (10, 128):  BACKGROUND_RANK,
        },
        "tapered_hidden2_elevated": {
            (512, 784): BACKGROUND_RANK,
            (256, 512): BACKGROUND_RANK,
            (128, 256): ELEVATED_RANK,
            (10, 128):  BACKGROUND_RANK,
        },
        "tapered_output_elevated": {
            (512, 784): BACKGROUND_RANK,
            (256, 512): BACKGROUND_RANK,
            (128, 256): BACKGROUND_RANK,
            (10, 128):  ELEVATED_RANK,
        },
    }

    tap_results = {}
    for cond_name, rank_spec in TAPERED_CONDITIONS.items():
        cond_results = run_condition(
            cond_name, rank_spec, SEEDS, tap_dir,
            X_train, y_train, X_test, y_test,
        )
        tap_results[cond_name] = compute_stats(cond_results)

    summarise(tap_results, "TAPERED MLP — CORRECTED PHASE 1 (5 SEEDS)", tap_dir)

    print("\n" + "#" * 60)
    print("# ALL DONE")
    print("#" * 60)


if __name__ == "__main__":
    main()
