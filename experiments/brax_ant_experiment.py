"""Brax Ant — Full LWR-EGGROLL experiment.

Matches the EGGROLL paper's Table 19 (brax/ant) hyperparameters exactly.

Hyperparameters from paper:
  activation: pqn
  deterministic_policy: false
  learning_rate: 0.1
  lr_decay: 0.9995
  layer_size: 256
  n_layers: 3
  pop_size: 2048
  optimizer: sgd
  rank: 4
  sigma: 0.2
  sigma_decay: 0.999

Architecture: MLP [27, 256, 256, 256, 8] (obs_dim=27, act_dim=8)
  Layer shapes: {(256,27): 'input', (256,256): 'hidden' (×2, shared shape), (8,256): 'output'}

Differences from MNIST/Gymnasium experiments (with rationale):
  1. Uses HyperscaleES natively (EggRoll/LWREggRoll noiser + MLP model)
     — perturbation mechanism identical to the paper's implementation.
  2. Brax environments are JAX-native (GPU evaluation, deterministic physics).
     No multiprocessing/numpy conversion needed.
  3. PQN activation (relu after layer norm) to match the paper.
  4. SGD optimizer (not Adam) for Ant, matching the paper's Table 19.
  5. sigma=0.2 (not 0.05) for Ant, matching the paper.
  6. Continuous actions clipped to [-1, 1].
  7. Sensitivity pilot uses Phase 2 + Phase 3 only (same as Gymnasium RL).

Seeds: 3. Episode length: 1000 steps.
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import sys
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from brax import envs

import hyperscalees as hs
from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.noiser.lwr_eggroll import LWREggRoll

# ── Configuration ───────────────────────────────────────────────────
ENV_NAME = "ant"
EPISODE_LENGTH = 1000

# Paper's Table 19: brax/ant
POP_SIZE = 2048
SIGMA = 0.2
LR = 0.1
SIGMA_DECAY = 0.999
LR_DECAY = 0.9995
RANK = 4
N_LAYERS = 3
LAYER_SIZE = 256
ACTIVATION = "pqn"
OPTIMIZER = optax.sgd

MAX_GENS = 500
SEEDS = [0, 1, 2]

# Pilot settings
PILOT_GENS = 50          # Enough to establish sensitivity ordering
PILOT_SEEDS = [0, 1, 2]  # Match main experiment seeds

RESULTS_DIR = Path("results/brax_ant")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Named layer shapes for this architecture
LAYER_SHAPES = {
    "input":  (256, 27),
    "hidden": (256, 256),   # Both hidden layers share this shape
    "output": (8, 256),
}

MODEL = hs.models.common.MLP

# ── Environment Setup ──────────────────────────────────────────────
def make_env():
    return envs.get_environment(ENV_NAME)

def get_dims():
    env = make_env()
    return env.observation_size, env.action_size

OBS_DIM, ACT_DIM = get_dims()

# ── Model + Noiser Init ───────────────────────────────────────────
def init_model(key):
    frozen_params, params, scan_map, es_map = MODEL.rand_init(
        key,
        in_dim=OBS_DIM,
        out_dim=ACT_DIM,
        hidden_dims=[LAYER_SIZE] * N_LAYERS,
        use_bias=True,
        activation=ACTIVATION,
        dtype="float32",
    )
    return frozen_params, params, scan_map, es_map

def init_noiser(params, noiser_class, rank_spec, sigma=SIGMA, lr=LR):
    frozen_noiser_params, noiser_params = noiser_class.init_noiser(
        params, sigma, lr,
        solver=OPTIMIZER,
        solver_kwargs={},
        rank=rank_spec,
    )
    return frozen_noiser_params, noiser_params

# ── Brax Episode Evaluation ───────────────────────────────────────
def evaluate_single_rollout(env, policy_fn, key, episode_length=EPISODE_LENGTH):
    """Run one Brax episode, return total reward."""
    state = env.reset(key)

    def scan_step(carry, _):
        state, total_reward, done = carry
        obs = state.obs
        action = policy_fn(obs)
        action = jnp.clip(action, -1.0, 1.0)
        next_state = env.step(state, action)
        reward = next_state.reward * (1.0 - done)
        done = jnp.logical_or(done, next_state.done)
        return (next_state, total_reward + reward, done), None

    init_carry = (state, jnp.float32(0.0), jnp.bool_(False))
    (_, total_reward, _), _ = jax.lax.scan(
        scan_step, init_carry, None, length=episode_length
    )
    return total_reward

# ── Training Loop ─────────────────────────────────────────────────
def train(seed, noiser_class, rank_spec, label, max_gens=MAX_GENS,
          sigma=SIGMA, lr=LR, sigma_decay=SIGMA_DECAY, lr_decay=LR_DECAY):
    """Train using HyperscaleES native API with Brax evaluation.

    Returns dict with per-generation history.
    """
    print(f"\n  [{label} seed={seed}] initialising...", flush=True)
    NOISER = noiser_class
    env = make_env()

    key = jax.random.key(seed)
    key, model_key, es_key = jax.random.split(key, 3)
    frozen_params, params, scan_map, es_map = init_model(model_key)
    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"  [{label} seed={seed}] params: {n_params:,}", flush=True)

    frozen_noiser_params, current_noiser_params = init_noiser(
        params, noiser_class, rank_spec, sigma=sigma, lr=lr
    )
    current_params = params

    history = []
    best_fitness = -float("inf")
    t0 = time.time()

    for gen in range(max_gens):
        t_gen = time.time()

        # Generate episode keys for each population member
        key, *member_keys = jax.random.split(key, POP_SIZE + 1)

        gen_fitnesses = []
        for mid in range(POP_SIZE):
            def member_policy(obs, mid=mid, gen=gen):
                iterinfo = (jnp.int32(gen), jnp.int32(mid))
                output = MODEL.forward(
                    NOISER, frozen_noiser_params, current_noiser_params,
                    frozen_params, current_params, es_tree_key, iterinfo, obs
                )
                return jnp.clip(output, -1.0, 1.0)

            reward = evaluate_single_rollout(env, member_policy, member_keys[mid])
            gen_fitnesses.append(float(reward))

        gen_fitnesses = jnp.array(gen_fitnesses)
        gen_mean = float(jnp.mean(gen_fitnesses))
        gen_best = float(jnp.max(gen_fitnesses))
        gen_var = float(jnp.var(gen_fitnesses))
        if gen_best > best_fitness:
            best_fitness = gen_best

        history.append({
            "gen": gen,
            "mean_fitness": gen_mean,
            "best_fitness": gen_best,
            "best_so_far": best_fitness,
            "fitness_variance": gen_var,
            "wall_s": time.time() - t0,
        })

        # Update
        iterinfo = (
            jnp.full(POP_SIZE, gen, dtype=jnp.int32),
            jnp.arange(POP_SIZE),
        )
        converted = NOISER.convert_fitnesses(
            frozen_noiser_params, current_noiser_params, gen_fitnesses
        )
        current_noiser_params, current_params = NOISER.do_updates(
            frozen_noiser_params, current_noiser_params, current_params,
            es_tree_key, converted, iterinfo, es_map
        )

        # Sigma decay
        if sigma_decay < 1.0:
            current_noiser_params['sigma'] *= sigma_decay

        gen_elapsed = time.time() - t_gen
        if gen % 5 == 0 or gen == max_gens - 1:
            print(f"    gen {gen:4d}  mean={gen_mean:.1f}  best_so_far={best_fitness:.1f}"
                  f"  sigma={float(current_noiser_params['sigma']):.4f}"
                  f"  ({gen_elapsed:.1f}s)", flush=True)

    total_time = time.time() - t0
    final_mean = history[-1]["mean_fitness"] if history else 0.0
    print(f"  [{label} seed={seed}] done {total_time:.0f}s, best={best_fitness:.1f}", flush=True)

    return {
        "method": label,
        "seed": seed,
        "best_fitness": best_fitness,
        "final_mean_fitness": final_mean,
        "generations": len(history),
        "wall_seconds": total_time,
        "history": history,
    }


# ── Sensitivity Pilot (Phase 2 + Phase 3) ─────────────────────────
def run_sensitivity_pilot():
    """Phase 2: causal ablation. Phase 3: binary inclusion check.

    Phase 2: Train baseline at uniform r=4 for PILOT_GENS gens.
    Then for each layer shape, retrain with that layer dropped to r=1.
    Layers with biggest fitness drop are most sensitive.

    Phase 3: For the least sensitive layer from Phase 2, compare
    r=1 vs r=0 to determine if freezing is justified.
    """
    print(f"\n{'='*60}")
    print("SENSITIVITY PILOT — Phase 2 (Causal Ablation)")
    print(f"{'='*60}")
    print(f"  Pilot gens: {PILOT_GENS}, Seeds: {PILOT_SEEDS}")
    print(f"  Baseline rank: {RANK} (uniform)")
    print(f"  Ablation rank: 1 (per layer)")

    unique_shapes = list(LAYER_SHAPES.items())  # (name, shape) pairs
    shape_to_name = {v: k for k, v in LAYER_SHAPES.items()}

    # --- Phase 2: Baseline at uniform r=4 ---
    print(f"\n  Phase 2 — Baseline (uniform r={RANK}):", flush=True)
    baseline_fitnesses = []
    for seed in PILOT_SEEDS:
        r = train(seed, EggRoll, RANK, f"pilot_baseline_r{RANK}", max_gens=PILOT_GENS)
        baseline_fitnesses.append(r["final_mean_fitness"])
    baseline_mean = np.mean(baseline_fitnesses)
    print(f"  Baseline mean fitness: {baseline_mean:.1f}", flush=True)

    # --- Phase 2: Ablate each layer to r=1 ---
    degradations = {}
    for layer_name, layer_shape in unique_shapes:
        print(f"\n  Phase 2 — Ablating {layer_name} {layer_shape} to r=1:", flush=True)
        ablated_spec = {s: RANK for s in LAYER_SHAPES.values()}
        ablated_spec[layer_shape] = 1
        ablated_fitnesses = []
        for seed in PILOT_SEEDS:
            r = train(seed, LWREggRoll, ablated_spec,
                      f"pilot_ablate_{layer_name}", max_gens=PILOT_GENS)
            ablated_fitnesses.append(r["final_mean_fitness"])
        ablated_mean = np.mean(ablated_fitnesses)
        degradation = baseline_mean - ablated_mean
        degradations[layer_name] = {
            "shape": layer_shape,
            "mean_fitness": ablated_mean,
            "degradation": degradation,
        }
        print(f"  {layer_name}: mean={ablated_mean:.1f}, degradation={degradation:.1f}", flush=True)

    # Order by degradation (most positive = most sensitive)
    ordering = sorted(degradations.keys(), key=lambda n: degradations[n]["degradation"], reverse=True)
    ordering_str = " > ".join(ordering)
    print(f"\n  Phase 2 sensitivity ordering: {ordering_str}")

    # --- Phase 3: Binary inclusion for least sensitive layer ---
    least_sensitive = ordering[-1]
    ls_shape = degradations[least_sensitive]["shape"]
    print(f"\n  Phase 3 — Binary inclusion: {least_sensitive} {ls_shape} at r=0 vs r=1")

    # r=1 for least sensitive (already done in Phase 2)
    r1_mean = degradations[least_sensitive]["mean_fitness"]

    # r=0 for least sensitive
    frozen_spec = {s: RANK for s in LAYER_SHAPES.values()}
    frozen_spec[ls_shape] = 0
    frozen_fitnesses = []
    for seed in PILOT_SEEDS:
        r = train(seed, LWREggRoll, frozen_spec,
                  f"pilot_freeze_{least_sensitive}", max_gens=PILOT_GENS)
        frozen_fitnesses.append(r["final_mean_fitness"])
    r0_mean = np.mean(frozen_fitnesses)

    freeze_justified = r0_mean >= r1_mean - 5.0  # Allow small tolerance
    phase3_decision = 0 if freeze_justified else 1
    print(f"  {least_sensitive} at r=1: {r1_mean:.1f}, at r=0: {r0_mean:.1f}")
    print(f"  Freeze justified: {freeze_justified} → assign rank {phase3_decision}")

    # --- Build allocation ---
    rank_tiers = [4, 2]  # Most sensitive gets 4, second gets 2
    allocation = {}
    alloc_named = {}
    for i, layer_name in enumerate(ordering):
        shape = LAYER_SHAPES[layer_name]
        if i == len(ordering) - 1:
            # Least sensitive: use Phase 3 decision
            allocation[shape] = phase3_decision
        else:
            allocation[shape] = rank_tiers[min(i, len(rank_tiers) - 1)]
        alloc_named[layer_name] = allocation[shape]

    alloc_label = "_".join(str(alloc_named[n]) for n in ["input", "hidden", "output"])

    pilot_results = {
        "baseline_mean": baseline_mean,
        "degradations": {k: {**v, "shape": str(v["shape"])} for k, v in degradations.items()},
        "ordering": ordering_str,
        "phase3": {
            "least_sensitive": least_sensitive,
            "r1_mean": r1_mean,
            "r0_mean": r0_mean,
            "freeze_justified": freeze_justified,
        },
        "allocation": alloc_named,
        "rank_spec": {str(k): v for k, v in allocation.items()},
    }

    # Save pilot results
    with open(RESULTS_DIR / "pilot_results.json", "w") as f:
        json.dump(pilot_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  PILOT SUMMARY")
    print(f"  Ordering: {ordering_str}")
    print(f"  Allocation: {alloc_named} (label: {alloc_label})")
    print(f"{'='*60}\n", flush=True)

    return allocation, alloc_named, ordering_str, alloc_label


# ── Main Experiment ───────────────────────────────────────────────
def main():
    tp = sum(s[0] * s[1] for s in LAYER_SHAPES.values())
    # hidden counted twice since there are 2 hidden layers
    actual_params = LAYER_SHAPES["input"][0] * LAYER_SHAPES["input"][1] + \
                    2 * LAYER_SHAPES["hidden"][0] * LAYER_SHAPES["hidden"][1] + \
                    LAYER_SHAPES["output"][0] * LAYER_SHAPES["output"][1]
    print("=" * 60)
    print(f"BRAX ANT — FULL LWR-EGGROLL EXPERIMENT")
    print(f"Architecture: [{OBS_DIM}, {LAYER_SIZE}, {LAYER_SIZE}, {LAYER_SIZE}, {ACT_DIM}]"
          f"  ({actual_params:,} weight params)")
    print(f"POP={POP_SIZE}, SIGMA={SIGMA}, LR={LR}, MAX_GENS={MAX_GENS}")
    print(f"SIGMA_DECAY={SIGMA_DECAY}, LR_DECAY={LR_DECAY}")
    print(f"Seeds: {SEEDS}")
    print("=" * 60, flush=True)

    # ── Step 1: Sensitivity Pilot ──
    pilot_file = RESULTS_DIR / "pilot_results.json"
    if pilot_file.exists():
        print("\nPilot results found, loading...", flush=True)
        with open(pilot_file) as f:
            pr = json.load(f)
        alloc_named = pr["allocation"]
        ordering_str = pr["ordering"]
        # Reconstruct rank_spec with tuple keys
        allocation = {}
        for layer_name, rank_val in alloc_named.items():
            allocation[LAYER_SHAPES[layer_name]] = rank_val
        alloc_label = "_".join(str(alloc_named[n]) for n in ["input", "hidden", "output"])
        print(f"  Ordering: {ordering_str}")
        print(f"  Allocation: {alloc_named}")
    else:
        allocation, alloc_named, ordering_str, alloc_label = run_sensitivity_pilot()

    # ── Step 2: Build methods ──
    ordered_layers = ordering_str.split(" > ")

    # Binary inclusion: rank 1 on sensitive layers, rank 0 on least sensitive
    binary_alloc = {}
    for i, layer_name in enumerate(ordered_layers):
        shape = LAYER_SHAPES[layer_name]
        binary_alloc[shape] = 0 if i == len(ordered_layers) - 1 else 1
    binary_label = "_".join(str(binary_alloc[LAYER_SHAPES[n]]) for n in ["input", "hidden", "output"])

    # Elevated LWR: pilot-derived allocation
    elevated_label = alloc_label

    METHODS = {
        # Baselines
        "eggroll_r1": (EggRoll, 1),
        "eggroll_r4": (EggRoll, RANK),
        "openai_es":  (EggRoll, None),              # Full-rank (None = no low-rank structure)
        # Three-way test
        f"lwr_binary_{binary_label}":    (LWREggRoll, binary_alloc),
        f"lwr_elevated_{elevated_label}": (LWREggRoll, allocation),
    }

    print(f"\nMethods to run:")
    for mn, (nc, rs) in METHODS.items():
        if rs is None:
            budget = "full"
        elif isinstance(rs, int):
            budget = rs * len(LAYER_SHAPES)
        elif isinstance(rs, dict):
            budget = sum(rs.values())
        else:
            budget = "?"
        print(f"  {mn}: noiser={nc.__name__}, rank_spec={rs}, budget={budget}")

    # ── Step 3: Run all methods ──
    all_results = {
        "pilot": {"allocation": alloc_named, "ordering": ordering_str},
        "config": {
            "pop_size": POP_SIZE, "sigma": SIGMA, "lr": LR,
            "sigma_decay": SIGMA_DECAY, "lr_decay": LR_DECAY,
            "max_gens": MAX_GENS, "episode_length": EPISODE_LENGTH,
            "activation": ACTIVATION, "optimizer": "sgd",
            "architecture": [OBS_DIM] + [LAYER_SIZE] * N_LAYERS + [ACT_DIM],
        },
    }

    for mn, (noiser_class, rank_spec) in METHODS.items():
        print(f"\n{'─'*60}")
        print(f"METHOD: {mn}")
        print(f"{'─'*60}", flush=True)

        method_results = []
        for seed in SEEDS:
            # Skip if result already exists
            result_file = RESULTS_DIR / f"{mn}_seed{seed}.json"
            if result_file.exists():
                print(f"  [{mn} seed={seed}] found existing result, skipping.", flush=True)
                with open(result_file) as f:
                    r = json.load(f)
                method_results.append(r)
                continue

            # Handle full-rank (openai_es): use a very high rank
            if rank_spec is None:
                # For "full rank" OpenAI-ES style, use rank equal to min dimension
                # of the largest layer. This effectively makes it full-rank.
                max_dim = max(min(s) for s in LAYER_SHAPES.values())
                actual_spec = max_dim  # e.g., rank=27 for input layer's min dim
            else:
                actual_spec = rank_spec

            r = train(seed, noiser_class, actual_spec, mn,
                      sigma_decay=SIGMA_DECAY, lr_decay=LR_DECAY)

            with open(result_file, "w") as f:
                json.dump(r, f, indent=2)
            method_results.append(r)

        bests = [r["best_fitness"] for r in method_results]
        means = [r["final_mean_fitness"] for r in method_results]
        all_results[mn] = {
            "mean_best": float(np.mean(bests)),
            "std_best": float(np.std(bests)),
            "mean_final": float(np.mean(means)),
            "per_seed_best": bests,
        }

    # ── Step 4: Summary ──
    print(f"\n{'='*60}")
    print("BRAX ANT — RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Pilot ordering: {ordering_str}")
    print(f"Pilot allocation: {alloc_named}")
    print(f"\n{'Method':<35} {'Mean Best':>12} {'Std':>8}")
    print("─" * 55)
    for m, s in all_results.items():
        if m in ("pilot", "config"):
            continue
        print(f"{m:<35} {s['mean_best']:>12.1f} {s['std_best']:>8.1f}")

    # Three-way comparison
    bl = [k for k in all_results if k.startswith("lwr_binary_")]
    el = [k for k in all_results if k.startswith("lwr_elevated_")]
    print(f"\n--- THREE-WAY COMPARISON ---")
    r1 = all_results.get("eggroll_r1", {})
    print(f"  Uniform r=1:       {r1.get('mean_best', 'N/A')}")
    if bl:
        b = all_results[bl[0]]
        print(f"  Binary inclusion:  {b['mean_best']:.1f} ± {b['std_best']:.1f}")
    if el:
        e = all_results[el[0]]
        print(f"  Elevated LWR:      {e['mean_best']:.1f} ± {e['std_best']:.1f}")

    # Save summary
    with open(RESULTS_DIR / "summary.json", "w") as f:
        json.dump({
            "env": ENV_NAME,
            "architecture": [OBS_DIM] + [LAYER_SIZE] * N_LAYERS + [ACT_DIM],
            "note": "Brax Ant full LWR-EGGROLL experiment matching paper Table 19",
            "results": all_results,
        }, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
