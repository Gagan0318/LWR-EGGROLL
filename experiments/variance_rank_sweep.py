"""
2D sweep over (σ, rank) to reproduce and extend the July 22, 2026
variance-rank finding.

The July finding, on MNIST MLP [256, 256, 256] with vanilla EGGROLL:
- At σ=0.1, low rank (1-2) beat high rank (8+).
- At σ=0.03, the ordering flipped: rank 128 beat rank 1 by ~10pp.
- Interpretation: rank's benefit is conditional on σ being small
  enough that the fitness signal is not dominated by noise.

This experiment maps the full σ×rank landscape at n=5 seeds:
  σ    ∈ {0.01, 0.03, 0.05, 0.1, 0.3}
  rank ∈ {1, 2, 4, 8, 16}

25 configurations × 5 seeds = 125 runs. Est ~1.5 hours on RTX 5060.

Outputs per-run JSON to results/variance_rank/ and a summary to
results/variance_rank/summary.json for downstream analysis.
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"

import json
import time
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax

sys.path.insert(0, str(Path(__file__).parent))
from compare_4_methods_mnist import (
    CFG, load_mnist, _train_hyperscalees_common,
)

import hyperscalees as hs
from hyperscalees.noiser.lwr_eggroll import LWREggRoll


PROJECT_ROOT = Path(__file__).parent.parent
SWEEP_DIR = PROJECT_ROOT / "results" / "variance_rank"
SWEEP_DIR.mkdir(parents=True, exist_ok=True)


# --- Sweep configuration ---
SIGMAS = (0.01, 0.03, 0.05, 0.1, 0.3)
RANKS = (1, 2, 4, 8, 16)
SWEEP_SEEDS = (0, 1, 2, 3, 4)

# We override the CFG sigma per run. Store the original to restore later
# in case CFG is shared with other running scripts (it isn't, but be safe).
_ORIGINAL_SIGMA = CFG.eggroll_sigma_init


def save_result(result, subdir):
    (SWEEP_DIR / subdir).mkdir(exist_ok=True)
    fn = SWEEP_DIR / subdir / f"{result['method']}_seed{result['seed']}.json"
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] {fn}")


def run_config(sigma, rank, seed, X_train, y_train, X_test, y_test):
    """Run one (σ, rank, seed) configuration.

    Uses LWREggRoll with a uniform scalar rank — equivalent to vanilla
    EGGROLL, but we already have the LWR machinery loaded and it
    saves an import.
    """
    # Override the sigma in CFG for this run
    CFG.eggroll_sigma_init = sigma

    label = f"vsweep_sig{sigma}_r{rank}"

    result = _train_hyperscalees_common(
        seed=seed,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        noiser_class=LWREggRoll,
        rank_spec=rank,   # scalar → uniform rank across all layers
        label=label,
    )

    # Enrich result with sweep metadata
    result["sweep"] = "variance_rank"
    result["sigma_init"] = sigma
    result["rank"] = rank

    return result


def run_sweep():
    print("\n" + "=" * 70)
    print("VARIANCE-RANK SWEEP (σ × rank)")
    print("=" * 70)
    print(f"σ values: {SIGMAS}")
    print(f"Rank values: {RANKS}")
    print(f"Seeds: {SWEEP_SEEDS}")
    print(f"Total runs: {len(SIGMAS) * len(RANKS) * len(SWEEP_SEEDS)}")

    X_train, y_train, X_test, y_test = load_mnist()

    all_results = []

    for sigma in SIGMAS:
        for rank in RANKS:
            print("\n" + "-" * 70)
            print(f"CONFIG: σ={sigma}, rank={rank}")
            print("-" * 70)

            for seed in SWEEP_SEEDS:
                print(f"\n>>> σ={sigma}, r={rank}, seed={seed}")

                result = run_config(
                    sigma, rank, seed,
                    X_train, y_train, X_test, y_test,
                )
                save_result(result, f"sig{sigma}_r{rank}")
                all_results.append(result)

    # Restore original sigma
    CFG.eggroll_sigma_init = _ORIGINAL_SIGMA

    return all_results


def summarize_sweep(results):
    """Group by (σ, rank) and compute mean±std of accuracy."""
    grouped = defaultdict(list)
    for r in results:
        key = (r["sigma_init"], r["rank"])
        grouped[key].append(r)

    summary = {}
    for (sigma, rank), runs in grouped.items():
        accs = [r["best_test_acc"] for r in runs]
        stops = [r["stop_reason"] for r in runs]
        wall_times = [
            r.get("converged_at_wall_s") or r.get("total_wall_s") or 0
            for r in runs
        ]

        # Also extract mean fitness variance over gens 1-50 for each run
        early_variances = []
        for r in runs:
            if "fitness_variance_history" in r:
                gen_variances = [
                    v for (g, v) in r["fitness_variance_history"]
                    if 1 <= g <= 50
                ]
                if gen_variances:
                    early_variances.append(np.mean(gen_variances))

        entry = {
            "sigma": sigma,
            "rank": rank,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "wall_mean": float(np.mean(wall_times)),
            "n_seeds": len(runs),
            "n_converged": sum(1 for s in stops if s == "patience"),
            "n_wallcap": sum(1 for s in stops if s == "wall_cap"),
            "variance_mean": (
                float(np.mean(early_variances)) if early_variances else None
            ),
            "variance_std": (
                float(np.std(early_variances)) if early_variances else None
            ),
        }
        summary[f"sig{sigma}_r{rank}"] = entry

    # Print accuracy table
    print("\n" + "=" * 70)
    print("ACCURACY TABLE (mean across seeds)")
    print("=" * 70)
    print(f"{'σ \\ rank':<10}", end="")
    for rank in RANKS:
        print(f"{'r=' + str(rank):<12}", end="")
    print()
    print("-" * (10 + 12 * len(RANKS)))
    for sigma in SIGMAS:
        print(f"{sigma:<10}", end="")
        for rank in RANKS:
            key = f"sig{sigma}_r{rank}"
            if key in summary:
                v = summary[key]
                print(f"{v['acc_mean']:.4f}     ", end="")
            else:
                print(f"{'MISSING':<12}", end="")
        print()

    # Print variance table
    print("\n" + "=" * 70)
    print("VARIANCE TABLE (mean over gens 1-50)")
    print("=" * 70)
    print(f"{'σ \\ rank':<10}", end="")
    for rank in RANKS:
        print(f"{'r=' + str(rank):<14}", end="")
    print()
    print("-" * (10 + 14 * len(RANKS)))
    for sigma in SIGMAS:
        print(f"{sigma:<10}", end="")
        for rank in RANKS:
            key = f"sig{sigma}_r{rank}"
            if key in summary:
                v = summary[key]
                if v["variance_mean"] is not None:
                    print(f"{v['variance_mean']:.2e}    ", end="")
                else:
                    print(f"{'N/A':<14}", end="")
            else:
                print(f"{'MISSING':<14}", end="")
        print()

    # Print stop-reason table
    print("\n" + "=" * 70)
    print("CONVERGENCE TABLE (# converged / # wall-capped)")
    print("=" * 70)
    print(f"{'σ \\ rank':<10}", end="")
    for rank in RANKS:
        print(f"{'r=' + str(rank):<12}", end="")
    print()
    for sigma in SIGMAS:
        print(f"{sigma:<10}", end="")
        for rank in RANKS:
            key = f"sig{sigma}_r{rank}"
            if key in summary:
                v = summary[key]
                print(
                    f"{v['n_converged']}/{v['n_wallcap']}          ",
                    end="",
                )
            else:
                print(f"{'MISSING':<12}", end="")
        print()

    # Print interaction detection
    print("\n" + "=" * 70)
    print("σ × RANK INTERACTION CHECK")
    print("=" * 70)
    print("For each σ, is the best rank (highest acc) different from the")
    print("best rank at other σ values? If yes, the interaction is present.")
    print()
    best_rank_at_sigma = {}
    for sigma in SIGMAS:
        best_rank = None
        best_acc = -np.inf
        for rank in RANKS:
            key = f"sig{sigma}_r{rank}"
            if key in summary and summary[key]["acc_mean"] > best_acc:
                best_acc = summary[key]["acc_mean"]
                best_rank = rank
        best_rank_at_sigma[sigma] = (best_rank, best_acc)
        print(f"  σ={sigma}: best rank = {best_rank}, acc = {best_acc:.4f}")

    # Was the July finding reproduced?
    print("\n" + "=" * 70)
    print("JULY 22 FINDING CHECK")
    print("=" * 70)
    sigma_low = 0.03
    sigma_high = 0.1
    if (sigma_low in best_rank_at_sigma and
            sigma_high in best_rank_at_sigma):
        best_low_r, _ = best_rank_at_sigma[sigma_low]
        best_high_r, _ = best_rank_at_sigma[sigma_high]
        print(f"  At σ={sigma_low}: best rank = {best_low_r}")
        print(f"  At σ={sigma_high}: best rank = {best_high_r}")
        if best_low_r > best_high_r:
            print("  ✓ Ordering FLIPS as σ increases: July finding reproduced.")
        elif best_low_r < best_high_r:
            print(
                "  ✗ Ordering inverts opposite July finding "
                "(higher σ prefers higher rank)."
            )
        else:
            print("  ~ Best rank is the same at both σ; interaction weak.")

    return summary


if __name__ == "__main__":
    t_start = time.perf_counter()

    all_results = run_sweep()
    summary = summarize_sweep(all_results)

    total_wall_min = (time.perf_counter() - t_start) / 60

    combined = {
        "summary": summary,
        "sigmas": list(SIGMAS),
        "ranks": list(RANKS),
        "seeds": list(SWEEP_SEEDS),
        "total_wall_minutes": total_wall_min,
        "architecture": [256, 256, 256],
    }
    with open(SWEEP_DIR / "summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 70)
    print(f"VARIANCE-RANK SWEEP COMPLETE — {total_wall_min:.1f} minutes")
    print("=" * 70)
    print(f"Summary written to: {SWEEP_DIR / 'summary.json'}")
