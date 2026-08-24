"""
Brax Ant — Multi-Seed Phase 3 Validation
==========================================
Tests whether Phase 3 binary inclusion (rank 0 vs rank 1 on hidden layers)
is stable across random initialisations when the frozen fraction is high.

Frozen fraction: hidden layers = 2×(256×256) = 131,072 / 140,032 = 93.6%
Pilot ordering: input > output > hidden (hidden = least sensitive)

Phase 3 conditions:
  rank_0_hidden: {input: 4, hidden: 0, output: 4}  — freeze hidden
  rank_1_hidden: {input: 4, hidden: 1, output: 4}  — minimal perturbation

Runs: 5 seeds × 2 conditions × 15 gens = ~2 hours on T4
Results save to Drive immediately per condition.

Usage on Colab:
  Save to Drive, then run with the standard Brax Ant setup cells.
"""

import os
import sys
import json
import time
import math
from pathlib import Path
from functools import partial

os.environ['XLA_FLAGS'] = '--xla_gpu_enable_cublaslt=false'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import brax
from brax import envs
from brax.io import model as brax_model

import flax.linen as nn
import optax

import hyperscalees as hs
from hyperscalees.noiser.lwr_eggroll import LWREggRoll
from hyperscalees.noiser.eggroll import EggRoll

# ============================================================
# Configuration
# ============================================================
SEEDS = [0, 1, 2, 3, 4]
PHASE3_GENS = 15           # Same as pilot Phase 3
POP_SIZE = 2048
SIGMA = 0.2
LR = 0.1
SIGMA_DECAY = 0.999
LR_DECAY = 0.9995
VMAP_CHUNK = 128           # T4-safe; change to 256 for L4
ENV_NAME = "ant"
ENV_STEPS = 1000

# Layer shapes (transposed: out_dim, in_dim)
INPUT_SHAPE  = (256, 27)
HIDDEN_SHAPE = (256, 256)   # 2 layers share this shape
OUTPUT_SHAPE = (8, 256)

LAYER_SHAPES = {
    "input": INPUT_SHAPE,
    "hidden": HIDDEN_SHAPE,
    "output": OUTPUT_SHAPE,
}

# Phase 3 conditions
SPEC_RANK0 = {INPUT_SHAPE: 4, HIDDEN_SHAPE: 0, OUTPUT_SHAPE: 4}
SPEC_RANK1 = {INPUT_SHAPE: 4, HIDDEN_SHAPE: 1, OUTPUT_SHAPE: 4}

# Frozen parameter stats
HIDDEN_PARAMS = 2 * 256 * 256  # 131,072
TOTAL_PARAMS = 27*256 + 256 + 256*256 + 256 + 256*256 + 256 + 256*8 + 8  # 140,032
FROZEN_FRACTION = HIDDEN_PARAMS / TOTAL_PARAMS

# Results directory — goes to Drive via symlink
RESULTS_DIR = Path("results/brax_ant/phase3_multiseed")


# ============================================================
# MLP Model (same as brax_ant_experiment.py)
# ============================================================
class AntMLP(nn.Module):
    hidden_dims: tuple = (256, 256, 256)
    action_dim: int = 8

    @nn.compact
    def __call__(self, x):
        for hd in self.hidden_dims:
            x = nn.Dense(hd)(x)
            x = nn.tanh(x)
        x = nn.Dense(self.action_dim)(x)
        return jnp.tanh(x)


MODEL = AntMLP()


# ============================================================
# Safe key wrapper (JAX legacy vs typed PRNG)
# ============================================================
def _safe_wrap_key(key):
    """Make key compatible with both legacy and typed PRNG."""
    if hasattr(key, 'ndim') and key.ndim >= 1:
        try:
            return jax.random.wrap_key_data(key)
        except Exception:
            return key
    return key


# ============================================================
# Environment setup
# ============================================================
def make_env():
    env = envs.get_environment(ENV_NAME)
    return env


def eval_population(env, params_batch, rng, chunk_size=VMAP_CHUNK):
    """Evaluate a batch of parameter sets on Brax Ant with VMAP chunking."""
    n = jax.tree.leaves(params_batch)[0].shape[0]
    n_chunks = math.ceil(n / chunk_size)
    all_rewards = []

    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, n)
        chunk_params = jax.tree.map(lambda x: x[start:end], params_batch)
        chunk_n = end - start

        chunk_rng = jax.random.fold_in(rng, chunk_idx)
        chunk_rngs = jax.random.split(chunk_rng, chunk_n)

        def rollout_single(p, r):
            state = env.reset(r)
            def step_fn(state, _):
                obs = state.obs
                action = MODEL.apply(p, obs)
                state = env.step(state, action)
                return state, state.reward
            final_state, rewards = jax.lax.scan(step_fn, state, None, length=ENV_STEPS)
            total = jnp.where(jnp.isnan(jnp.sum(rewards)), -1e6, jnp.sum(rewards))
            return total

        chunk_rewards = jax.vmap(rollout_single)(chunk_params, chunk_rngs)
        all_rewards.append(chunk_rewards)

    return jnp.concatenate(all_rewards)


# ============================================================
# Training function (simplified for Phase 3 — no checkpoints needed)
# ============================================================
def train(seed, noiser_class, rank_spec, label, max_gens=PHASE3_GENS):
    """Train for max_gens generations. Returns result dict."""
    rng = jax.random.PRNGKey(seed)
    rng = _safe_wrap_key(rng)

    env = make_env()

    # Init params
    rng, init_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros(27)
    params = MODEL.init(init_rng, dummy_obs)

    # Init optimizer
    optimizer = optax.adam(LR)
    opt_state = optimizer.init(params)

    # Init noiser
    rng, noiser_rng = jax.random.split(rng)
    n_params = sum(p.size for p in jax.tree.leaves(params))

    if noiser_class == LWREggRoll:
        frozen_noiser_params, current_noiser_params = hs.noiser.init(
            noiser_class, params, noiser_rng, rank_spec,
            {"sigma": SIGMA, "pop_size": POP_SIZE}
        )
    else:
        frozen_noiser_params, current_noiser_params = hs.noiser.init(
            noiser_class, params, noiser_rng, rank_spec,
            {"sigma": SIGMA, "pop_size": POP_SIZE}
        )

    print(f"  [{label} seed={seed}] initialising... params={n_params:,}", flush=True)

    history = []
    best_fitness = -1e10
    t0 = time.time()
    sigma = SIGMA
    lr = LR

    for gen in range(max_gens):
        rng, gen_rng = jax.random.split(rng)
        gen_rng = _safe_wrap_key(gen_rng)

        # Generate population
        population = hs.noiser.get_pop(
            noiser_class, frozen_noiser_params, current_noiser_params,
            params, gen_rng
        )

        # Evaluate
        rng, eval_rng = jax.random.split(rng)
        fitnesses = eval_population(env, population, eval_rng)

        # Handle NaNs
        fitnesses = jnp.where(jnp.isnan(fitnesses), -1e6, fitnesses)

        mean_fit = float(jnp.mean(fitnesses))
        best_fit = float(jnp.max(fitnesses))
        best_fitness = max(best_fitness, best_fit)

        # Update
        converted = hs.noiser.convert_fit(
            noiser_class, frozen_noiser_params, fitnesses
        )
        grad = hs.noiser.get_grad(
            noiser_class, frozen_noiser_params, current_noiser_params,
            params, gen_rng, converted
        )
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Decay
        sigma *= SIGMA_DECAY
        lr *= LR_DECAY
        current_noiser_params = hs.noiser.do_updates(
            noiser_class, frozen_noiser_params, current_noiser_params,
            {"sigma": sigma}
        )

        elapsed = time.time() - t0
        history.append({
            "wall_s": elapsed, "gen": gen,
            "mean_fitness": mean_fit, "best_fitness": best_fit,
            "best_so_far": best_fitness,
        })

        if gen % 5 == 0 or gen == max_gens - 1:
            print(f"  [{label} seed={seed}] gen={gen:4d}  "
                  f"best={best_fit:.1f}  mean={mean_fit:.1f}  "
                  f"best_so_far={best_fitness:.1f}  wall={elapsed:.0f}s",
                  flush=True)

    total_time = time.time() - t0
    result = {
        "method": label, "seed": seed,
        "best_fitness": best_fitness,
        "final_mean_fitness": mean_fit,
        "generations": len(history),
        "wall_seconds": total_time,
        "history": history,
    }
    return result


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("BRAX ANT — MULTI-SEED PHASE 3 VALIDATION")
    print(f"Architecture: [27, 256, 256, 256, 8]  ({TOTAL_PARAMS:,} params)")
    print(f"Target layers: hidden {HIDDEN_SHAPE} (×2)")
    print(f"Frozen fraction (rank 0 on hidden): {FROZEN_FRACTION:.1%}")
    print(f"POP={POP_SIZE}, SIGMA={SIGMA}, GENS={PHASE3_GENS}")
    print(f"Seeds: {SEEDS}")
    print(f"VMAP_CHUNK: {VMAP_CHUNK}")
    print(f"Conditions: rank_0_hidden vs rank_1_hidden")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_r0 = {}
    results_r1 = {}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        # --- Rank 0 on hidden ---
        label_r0 = "phase3_rank0_hidden"
        result_file_r0 = RESULTS_DIR / f"{label_r0}_seed{seed}.json"

        if result_file_r0.exists():
            with open(result_file_r0) as f:
                r0 = json.load(f)
            print(f"  [SKIP] {label_r0} seed={seed} — "
                  f"best={r0['best_fitness']:.1f}")
        else:
            r0 = train(seed, LWREggRoll, SPEC_RANK0, label_r0)
            with open(result_file_r0, "w") as f:
                json.dump(r0, f, indent=2, default=str)
            print(f"  [SAVED] {result_file_r0}")
        results_r0[seed] = r0

        # --- Rank 1 on hidden ---
        label_r1 = "phase3_rank1_hidden"
        result_file_r1 = RESULTS_DIR / f"{label_r1}_seed{seed}.json"

        if result_file_r1.exists():
            with open(result_file_r1) as f:
                r1 = json.load(f)
            print(f"  [SKIP] {label_r1} seed={seed} — "
                  f"best={r1['best_fitness']:.1f}")
        else:
            r1 = train(seed, LWREggRoll, SPEC_RANK1, label_r1)
            with open(result_file_r1, "w") as f:
                json.dump(r1, f, indent=2, default=str)
            print(f"  [SAVED] {result_file_r1}")
        results_r1[seed] = r1

    # ============================================================
    # Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    bests_r0 = []
    bests_r1 = []
    recommendations = []

    print(f"\n{'Seed':<6} {'Rank 0':<12} {'Rank 1':<12} {'Diff':<10} {'Recommends'}")
    print("-" * 54)

    for seed in SEEDS:
        b0 = results_r0[seed]["best_fitness"]
        b1 = results_r1[seed]["best_fitness"]
        diff = b0 - b1
        rec = "rank 0" if b0 >= b1 else "rank 1"

        bests_r0.append(b0)
        bests_r1.append(b1)
        recommendations.append(rec)

        print(f"{seed:<6} {b0:<12.1f} {b1:<12.1f} {diff:+.1f}      {rec}")

    mean_r0 = np.mean(bests_r0)
    std_r0 = np.std(bests_r0)
    mean_r1 = np.mean(bests_r1)
    std_r1 = np.std(bests_r1)

    rank0_count = recommendations.count("rank 0")
    rank1_count = recommendations.count("rank 1")
    consistency = max(rank0_count, rank1_count) / len(SEEDS)

    print(f"\n{'Mean':<6} {mean_r0:<12.1f} {mean_r1:<12.1f} {mean_r0-mean_r1:+.1f}")
    print(f"{'Std':<6} {std_r0:<12.1f} {std_r1:<12.1f}")
    print(f"{'Range':<6} {max(bests_r0)-min(bests_r0):<12.1f} "
          f"{max(bests_r1)-min(bests_r1):<12.1f}")

    print(f"\nRecommendation split: rank 0 = {rank0_count}/{len(SEEDS)}, "
          f"rank 1 = {rank1_count}/{len(SEEDS)}")
    print(f"Consistency: {consistency:.0%}")
    print(f"Frozen parameter fraction: {FROZEN_FRACTION:.1%}")

    if consistency < 0.8:
        verdict = ("UNSTABLE — recommendation varies across seeds. "
                   "Multi-seed Phase 3 is REQUIRED at this frozen fraction.")
    elif consistency < 1.0:
        verdict = (f"MOSTLY STABLE — one recommendation wins {max(rank0_count,rank1_count)}/{len(SEEDS)}, "
                   "but not unanimous. Multi-seed Phase 3 recommended.")
    else:
        winner = "rank 0" if rank0_count == len(SEEDS) else "rank 1"
        verdict = (f"STABLE — {winner} wins across all seeds. "
                   "Single-seed Phase 3 would be sufficient here.")

    print(f"\nVerdict: {verdict}")

    # ============================================================
    # Cross-environment comparison
    # ============================================================
    print("\n" + "=" * 60)
    print("CROSS-ENVIRONMENT COMPARISON")
    print("=" * 60)
    print(f"{'':20} {'MNIST (output)':<20} {'Brax Ant (hidden)'}")
    print(f"{'Frozen fraction':<20} {'0.8%':<20} {FROZEN_FRACTION:.1%}")
    print(f"{'Phase 3 consistency':<20} {'100% (5/5 rank 0)':<20} "
          f"{consistency:.0%} ({rank0_count}/{len(SEEDS)} rank 0)")
    print(f"{'Cross-seed range':<20} {'0.053 (accuracy)':<20} "
          f"{max(bests_r0)-min(bests_r0):.1f} (fitness)")
    print(f"\nThreshold finding: multi-seed Phase 3 is "
          f"{'needed' if consistency < 1.0 else 'not needed even'} "
          f"at {FROZEN_FRACTION:.1%} frozen fraction.")

    # Save summary
    summary = {
        "experiment": "brax_ant_phase3_multiseed_validation",
        "architecture": [27, 256, 256, 256, 8],
        "target_layers": "hidden",
        "target_shape": list(HIDDEN_SHAPE),
        "frozen_fraction": FROZEN_FRACTION,
        "seeds": SEEDS,
        "gens": PHASE3_GENS,
        "pop": POP_SIZE,
        "sigma": SIGMA,
        "vmap_chunk": VMAP_CHUNK,
        "per_seed": {
            str(s): {
                "rank_0_best": bests_r0[i],
                "rank_1_best": bests_r1[i],
                "diff": bests_r0[i] - bests_r1[i],
                "recommends": recommendations[i]
            }
            for i, s in enumerate(SEEDS)
        },
        "summary": {
            "rank_0_mean": mean_r0, "rank_0_std": std_r0,
            "rank_1_mean": mean_r1, "rank_1_std": std_r1,
            "rank_0_range": max(bests_r0) - min(bests_r0),
            "rank_1_range": max(bests_r1) - min(bests_r1),
            "consistency": consistency,
            "verdict": verdict
        }
    }

    summary_file = RESULTS_DIR / "phase3_multiseed_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
