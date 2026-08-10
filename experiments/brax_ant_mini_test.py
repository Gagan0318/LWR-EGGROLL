"""Brax Ant — Mini signal test.
Quick check: does LWR help on deterministic RL?
Hypothesis: yes, because Brax has clean fitness (like MNIST, unlike Gymnasium).

Miniature settings: POP=64, GENS=50, 1 seed.
Runs ~3 methods: uniform r=1, uniform r=4, pilot-derived LWR.
Expected runtime: 1-2 hours on RTX 5060 (including JIT).

This is NOT the full experiment — just a directional signal check.
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import json, time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from brax import envs

import hyperscalees as hs
from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.noiser.lwr_eggroll import LWREggRoll

# ── Mini config ─────────────────────────────────────────
ENV_NAME = "ant"
EPISODE_LENGTH = 1000
POP_SIZE = 64
SIGMA = 0.2
LR = 0.1
SIGMA_DECAY = 0.999
LR_DECAY = 0.9995
RANK = 4
N_LAYERS = 3
LAYER_SIZE = 256
ACTIVATION = "pqn"
OPTIMIZER = optax.sgd

MAX_GENS = 50
PILOT_GENS = 20
SEED = 0

LAYER_SHAPES = {
    "input":  (256, 27),
    "hidden": (256, 256),
    "output": (8, 256),
}

MODEL = hs.models.common.MLP
RESULTS_DIR = Path("results/brax_ant_mini")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment ────────────────────────────────────────
def make_env():
    return envs.get_environment(ENV_NAME)

env = make_env()
OBS_DIM = env.observation_size
ACT_DIM = env.action_size

# ── Model + Noiser ─────────────────────────────────────
def init_model(key):
    return MODEL.rand_init(
        key, in_dim=OBS_DIM, out_dim=ACT_DIM,
        hidden_dims=[LAYER_SIZE] * N_LAYERS,
        use_bias=True, activation=ACTIVATION, dtype="float32",
    )

def init_noiser(params, noiser_class, rank_spec):
    return noiser_class.init_noiser(
        params, SIGMA, LR,
        solver=OPTIMIZER, solver_kwargs={}, rank=rank_spec,
    )

# ── Episode rollout ────────────────────────────────────
def evaluate_single_rollout(env, policy_fn, key):
    state = env.reset(key)
    def scan_step(carry, _):
        state, total_reward, done = carry
        action = jnp.clip(policy_fn(state.obs), -1.0, 1.0)
        ns = env.step(state, action)
        reward = ns.reward * (1.0 - done)
        done = jnp.logical_or(done, ns.done)
        return (ns, total_reward + reward, done), None
    (_, total_reward, _), _ = jax.lax.scan(
        scan_step, (state, jnp.float32(0.0), jnp.bool_(False)),
        None, length=EPISODE_LENGTH,
    )
    return total_reward

# ── Training ───────────────────────────────────────────
def train(seed, noiser_class, rank_spec, label, max_gens=MAX_GENS):
    NOISER = noiser_class
    env = make_env()

    key = jax.random.key(seed)
    key, model_key, es_key = jax.random.split(key, 3)
    frozen_params, params, scan_map, es_map = init_model(model_key)
    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)

    frozen_noiser_params, current_noiser_params = init_noiser(params, noiser_class, rank_spec)
    current_params = params

    history = []
    best_fitness = -float("inf")
    t0 = time.time()

    for gen in range(max_gens):
        t_gen = time.time()
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
        if gen_best > best_fitness:
            best_fitness = gen_best

        history.append({"gen": gen, "mean": gen_mean, "best": gen_best, "best_so_far": best_fitness})

        iterinfo = (jnp.full(POP_SIZE, gen, dtype=jnp.int32), jnp.arange(POP_SIZE))
        converted = NOISER.convert_fitnesses(frozen_noiser_params, current_noiser_params, gen_fitnesses)
        current_noiser_params, current_params = NOISER.do_updates(
            frozen_noiser_params, current_noiser_params, current_params,
            es_tree_key, converted, iterinfo, es_map
        )
        if SIGMA_DECAY < 1.0:
            current_noiser_params['sigma'] *= SIGMA_DECAY

        gen_time = time.time() - t_gen
        if gen % 5 == 0:
            print(f"  [{label}] gen {gen:3d}  mean={gen_mean:.1f}  best_so_far={best_fitness:.1f}  ({gen_time:.1f}s)", flush=True)

    total = time.time() - t0
    final_mean = history[-1]["mean"]
    print(f"  [{label}] done {total:.0f}s, best={best_fitness:.1f}, final_mean={final_mean:.1f}", flush=True)
    return {"method": label, "seed": seed, "best_fitness": best_fitness,
            "final_mean": final_mean, "wall_seconds": total, "history": history}


# ── Mini Pilot (Phase 2 only — quick) ─────────────────
def mini_pilot():
    # Check for cached pilot results
    pilot_file = RESULTS_DIR / "pilot_result.json"
    if pilot_file.exists():
        print(f"\nFound existing pilot result, loading from {pilot_file}", flush=True)
        with open(pilot_file) as f:
            cached = json.load(f)
        allocation = {tuple(int(x) for x in k.split(",")): v for k, v in cached["allocation_shape_keyed"].items()}
        return allocation, cached["alloc_named"], cached["ordering"]

    print(f"\n{'='*60}")
    print("MINI PILOT — Phase 2 causal ablation")
    print(f"POP={POP_SIZE}, GENS={PILOT_GENS}, seed={SEED}")
    print(f"{'='*60}\n", flush=True)

    # Baseline at uniform r=4
    print("Baseline (uniform r=4):", flush=True)
    bl = train(SEED, EggRoll, RANK, "pilot_baseline", max_gens=PILOT_GENS)
    baseline_mean = bl["final_mean"]

    # Ablate each layer to r=1
    degradations = {}
    for layer_name, layer_shape in LAYER_SHAPES.items():
        print(f"\nAblating {layer_name} to r=1:", flush=True)
        spec = {s: RANK for s in LAYER_SHAPES.values()}
        spec[layer_shape] = 1
        ab = train(SEED, LWREggRoll, spec, f"pilot_ablate_{layer_name}", max_gens=PILOT_GENS)
        deg = baseline_mean - ab["final_mean"]
        degradations[layer_name] = deg
        print(f"  {layer_name}: degradation = {deg:.1f}", flush=True)

    ordering = sorted(degradations.keys(), key=lambda n: degradations[n], reverse=True)
    print(f"\nOrdering: {' > '.join(ordering)}")
    print(f"Degradations: {degradations}")

    # Simple allocation: most sensitive=4, middle=2, least=0
    rank_tiers = [4, 2, 0]
    allocation = {}
    alloc_named = {}
    for i, name in enumerate(ordering):
        allocation[LAYER_SHAPES[name]] = rank_tiers[i]
        alloc_named[name] = rank_tiers[i]

    print(f"Allocation: {alloc_named}")

    # Cache pilot results
    with open(pilot_file, "w") as f:
        json.dump({
            "ordering": ordering, "alloc_named": alloc_named,
            "degradations": degradations, "baseline_mean": baseline_mean,
            "allocation_shape_keyed": {f"{k[0]},{k[1]}": v for k, v in allocation.items()},
        }, f, indent=2)
    print(f"Pilot cached to {pilot_file}", flush=True)

    return allocation, alloc_named, ordering


# ── Main ───────────────────────────────────────────────
def main():
    print("=" * 60)
    print("BRAX ANT — MINI SIGNAL TEST")
    print(f"POP={POP_SIZE}, GENS={MAX_GENS}, PILOT_GENS={PILOT_GENS}, seed={SEED}")
    print("Question: does LWR help on deterministic RL?")
    print("=" * 60, flush=True)

    allocation, alloc_named, ordering = mini_pilot()
    alloc_label = "_".join(str(alloc_named[n]) for n in ["input", "hidden", "output"])

    METHODS = {
        "eggroll_r1": (EggRoll, 1),
        "eggroll_r4": (EggRoll, RANK),
        f"lwr_{alloc_label}": (LWREggRoll, allocation),
    }

    results = {}
    for mn, (noiser_class, rank_spec) in METHODS.items():
        result_file = RESULTS_DIR / f"{mn}.json"
        if result_file.exists():
            print(f"\n[{mn}] found existing result, skipping.", flush=True)
            with open(result_file) as f:
                r = json.load(f)
        else:
            print(f"\n{'─'*60}")
            print(f"METHOD: {mn}")
            print(f"{'─'*60}", flush=True)
            r = train(SEED, noiser_class, rank_spec, mn)
            with open(result_file, "w") as f:
                json.dump(r, f, indent=2)
        results[mn] = r

    print(f"\n{'='*60}")
    print("MINI SIGNAL TEST RESULTS")
    print(f"{'='*60}")
    print(f"Pilot ordering: {' > '.join(ordering)}")
    print(f"Pilot allocation: {alloc_named}")
    print(f"\n{'Method':<30} {'Final Mean':>12} {'Best':>10}")
    print("─" * 52)
    for mn, r in results.items():
        print(f"{mn:<30} {r['final_mean']:>12.1f} {r['best_fitness']:>10.1f}")

    # Verdict
    lwr_key = [k for k in results if k.startswith("lwr_")][0]
    r4_mean = results["eggroll_r4"]["final_mean"]
    lwr_mean = results[lwr_key]["final_mean"]
    delta = lwr_mean - r4_mean
    print(f"\nLWR vs uniform r=4: {delta:+.1f}")
    if delta > 5:
        print("SIGNAL: LWR helps on deterministic RL — full experiment justified")
    elif delta > -5:
        print("SIGNAL: Inconclusive — similar performance, full experiment may clarify")
    else:
        print("SIGNAL: LWR does not help here — investigate before committing T4 time")

    with open(RESULTS_DIR / "mini_summary.json", "w") as f:
        json.dump({"ordering": ordering, "allocation": alloc_named,
                   "results": {k: {"final_mean": v["final_mean"], "best": v["best_fitness"]}
                               for k, v in results.items()},
                   "delta_vs_r4": delta}, f, indent=2)

    print(f"\nSaved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
