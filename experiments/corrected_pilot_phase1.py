"""Corrected Phase 1 Sensitivity Pilot — rank-1 background (Lehre feedback).

Runs three conditions on MNIST, 3 seeds each:
  1. Input elevated (rank 8), hidden + output at rank 1
  2. Hidden elevated (rank 8), input + output at rank 1
  3. Output elevated (rank 8), input + hidden at rank 1

Compares against the original pilot (rank-0 background) to confirm
the sensitivity ordering is unchanged.
"""

import sys, json, time
from pathlib import Path

# ── project imports ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from experiments.compare_4_methods_mnist import (
    load_mnist, train_lwr_eggroll, CFG
)

RESULTS_DIR = Path("results/corrected_pilot_phase1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 1, 2]
ELEVATED_RANK = 8
BACKGROUND_RANK = 1

# Layer group specs: target layer gets elevated rank, others get background rank 1
CONDITIONS = {
    "input_elevated": {
        (256, 784): ELEVATED_RANK,   # input → rank 8
        (256, 256): BACKGROUND_RANK, # hidden → rank 1
        (10, 256):  BACKGROUND_RANK, # output → rank 1
    },
    "hidden_elevated": {
        (256, 784): BACKGROUND_RANK, # input → rank 1
        (256, 256): ELEVATED_RANK,   # hidden → rank 8
        (10, 256):  BACKGROUND_RANK, # output → rank 1
    },
    "output_elevated": {
        (256, 784): BACKGROUND_RANK, # input → rank 1
        (256, 256): BACKGROUND_RANK, # hidden → rank 1
        (10, 256):  ELEVATED_RANK,   # output → rank 8
    },
}


def main():
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
                    label=f"corrected_p1_{cond_name}",
                )
                elapsed = time.time() - t0
                print(f"    acc={result['best_test_acc']:.4f}  ({elapsed:.1f}s)")

                with open(out_file, "w") as f:
                    json.dump(result, f, indent=2)

            cond_results.append(result)

        # Summarise
        accs = [r["best_test_acc"] for r in cond_results]
        import numpy as np
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        # Fitness variance if available
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

    # ── Final summary ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("CORRECTED PHASE 1 SENSITIVITY PILOT — SUMMARY")
    print(f"Background rank: {BACKGROUND_RANK} (corrected from 0)")
    print(f"Elevated rank: {ELEVATED_RANK}")
    print(f"{'='*60}")

    print(f"\n{'Layer':<20} {'Accuracy':>15} {'Fitness Var':>20}")
    print("-" * 55)

    # Sort by accuracy descending to show ordering
    sorted_conds = sorted(all_results.items(), key=lambda x: x[1]["mean_acc"], reverse=True)
    for cond_name, stats in sorted_conds:
        acc_str = f"{stats['mean_acc']:.4f} ± {stats['std_acc']:.4f}"
        if stats["mean_fitness_var"] is not None:
            fvar_str = f"{stats['mean_fitness_var']:.6f} ± {stats['std_fitness_var']:.6f}"
        else:
            fvar_str = "N/A"
        print(f"{cond_name:<20} {acc_str:>15} {fvar_str:>20}")

    ordering = [name.replace("_elevated", "") for name, _ in sorted_conds]
    print(f"\nSensitivity ordering: {' >> '.join(ordering)}")

    expected = ["input", "hidden", "output"]
    if ordering == expected:
        print("✓ ORDERING MATCHES original pilot. All downstream results remain valid.")
    else:
        print("✗ ORDERING DIFFERS from original pilot. Check results carefully.")

    # Save summary
    summary_file = RESULTS_DIR / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")


if __name__ == "__main__":
    main()
