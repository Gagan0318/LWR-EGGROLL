"""Corrected Phase 1 Sensitivity Pilot — tapered architecture [784, 512, 256, 128, 10]."""

import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.compare_4_methods_mnist import load_mnist, train_lwr_eggroll
from experiments.moving_hidden_layer import setup_cfg

RESULTS_DIR = Path("results/corrected_pilot_phase1_tapered")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
ELEVATED_RANK = 8
BACKGROUND_RANK = 1

CONDITIONS = {
    "input_elevated": {
        (512, 784): ELEVATED_RANK,
        (256, 512): BACKGROUND_RANK,
        (128, 256): BACKGROUND_RANK,
        (10, 128):  BACKGROUND_RANK,
    },
    "hidden1_elevated": {
        (512, 784): BACKGROUND_RANK,
        (256, 512): ELEVATED_RANK,
        (128, 256): BACKGROUND_RANK,
        (10, 128):  BACKGROUND_RANK,
    },
    "hidden2_elevated": {
        (512, 784): BACKGROUND_RANK,
        (256, 512): BACKGROUND_RANK,
        (128, 256): ELEVATED_RANK,
        (10, 128):  BACKGROUND_RANK,
    },
    "output_elevated": {
        (512, 784): BACKGROUND_RANK,
        (256, 512): BACKGROUND_RANK,
        (128, 256): BACKGROUND_RANK,
        (10, 128):  ELEVATED_RANK,
    },
}


def main():
    setup_cfg((512, 256, 128))
    X_train, y_train, X_test, y_test = load_mnist()

    all_results = {}

    for cond_name, rank_spec in CONDITIONS.items():
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond_name}")
        print(f"Rank spec: {rank_spec}")
        print(f"{'='*60}")

        cond_results = []
        for seed in SEEDS:
            out_file = RESULTS_DIR / f"{cond_name}_seed{seed}.json"

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
                    label=f"corrected_p1_tapered_{cond_name}",
                )
                elapsed = time.time() - t0
                print(f"    acc={result['best_test_acc']:.4f}  ({elapsed:.1f}s)")

                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)

            cond_results.append(result)

        accs = [r["best_test_acc"] for r in cond_results]
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        if "fitness_variance_history" in cond_results[0]:
            fvars = [np.mean(r["fitness_variance_history"]) for r in cond_results]
            mean_fvar = np.mean(fvars)
            std_fvar = np.std(fvars)
        else:
            mean_fvar = None
            std_fvar = None

        all_results[cond_name] = {
            "accs": accs,
            "mean_acc": float(mean_acc),
            "std_acc": float(std_acc),
            "mean_fitness_var": float(mean_fvar) if mean_fvar is not None else None,
            "std_fitness_var": float(std_fvar) if std_fvar is not None else None,
        }

    print(f"\n{'='*60}")
    print("TAPERED ARCHITECTURE — CORRECTED PHASE 1 SUMMARY")
    print(f"Architecture: [784, 512, 256, 128, 10]")
    print(f"Background rank: {BACKGROUND_RANK}, Elevated rank: {ELEVATED_RANK}")
    print(f"{'='*60}")

    print(f"\n{'Layer':<25} {'Accuracy':>18} {'Fitness Var':>22}")
    print("-" * 65)

    sorted_by_fvar = sorted(
        all_results.items(),
        key=lambda x: x[1]["mean_fitness_var"] if x[1]["mean_fitness_var"] else 0,
        reverse=True
    )
    for cond_name, stats in sorted_by_fvar:
        acc_str = f"{stats['mean_acc']:.4f} ± {stats['std_acc']:.4f}"
        if stats["mean_fitness_var"] is not None:
            fvar_str = f"{stats['mean_fitness_var']:.2f} ± {stats['std_fitness_var']:.2f}"
        else:
            fvar_str = "N/A"
        print(f"{cond_name:<25} {acc_str:>18} {fvar_str:>22}")

    ordering = [name.replace("_elevated", "") for name, _ in sorted_by_fvar]
    print(f"\nSensitivity ordering (by fitness variance): {' >> '.join(ordering)}")

    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {RESULTS_DIR}/summary.json")


if __name__ == "__main__":
    main()
