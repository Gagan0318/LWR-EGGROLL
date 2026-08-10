"""Brax Ant — Full LWR-EGGROLL experiment (final, 20-hour GPU budget).

Paper Table 19 hyperparameters: activation=pqn, lr=0.1, lr_decay=0.9995,
layer_size=256, n_layers=3, pop_size=2048, optimizer=sgd, rank=4,
sigma=0.2, sigma_decay=0.999.

Architecture: MLP [27, 256, 256, 256, 8]
Layer shapes: {(256,27):'input', (256,256):'hidden' (x2 shared), (8,256):'output'}

Three-phase sensitivity pilot (Phase 1 + 2 + 3).
Main: eggroll_r4 (budget 16), eggroll_r1 (budget 4), lwr (pilot-derived).

CRASH RECOVERY: Every train() checkpoints every 10 min. Every pilot
sub-result cached individually. Session crash loses <=10 min of compute.
Delete results/brax_ant/ to force full restart.

IMPORTANT: Changing any config constants (POP_SIZE, pilot gens, seeds, etc.)
after partial completion requires clearing cache: rm -rf results/brax_ant/cache/
"""
import os
# XLA persistent compilation cache — survives session restarts on cluster
try:
    os.makedirs("/jupyter/work/jax_cache", exist_ok=True)
    os.environ.setdefault("XLA_FLAGS",
        "--xla_gpu_persistent_cache_dir=/jupyter/work/jax_cache")
except OSError:
    pass  # Not on cluster — skip XLA cache
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"

import ast
import json
import pickle
import sys
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

MAX_GENS = 150
MAX_GENS_FLOOR = 30
SEEDS = [0, 1, 2]

PHASE1_CHECKPOINT_GENS = 25
PHASE1_ELEVATION_GENS = 10
PHASE2_GENS = 15
PHASE3_GENS = 15
PILOT_SEEDS = [0, 1]

NAN_REPLACEMENT = -1000.0
CHECKPOINT_INTERVAL_S = 600

RESULTS_DIR = Path("results/brax_ant")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = RESULTS_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LAYER_SHAPES = {
    "input":  (256, 27),
    "hidden": (256, 256),
    "output": (8, 256),
}

MODEL = hs.models.common.MLP

# ── Environment ────────────────────────────────────────────────────
def make_env():
    return envs.get_environment(ENV_NAME)

def get_dims():
    env = make_env()
    return env.observation_size, env.action_size

OBS_DIM, ACT_DIM = get_dims()

# ── Checkpoint I/O (numpy-safe for cross-session pickle) ──────────
def _to_numpy_leaf(x):
    """Convert JAX arrays to numpy for pickle. Leave others unchanged."""
    if hasattr(x, 'dtype') and hasattr(x, 'shape') and not isinstance(x, np.ndarray):
        try:
            return np.asarray(x)
        except Exception:
            return x
    return x

def _to_jax_leaf(x):
    """Convert numpy arrays back to JAX on load."""
    if isinstance(x, np.ndarray):
        return jnp.asarray(x)
    return x

def save_pickle(filepath, data):
    filepath = Path(filepath)
    np_data = jax.tree_util.tree_map(_to_numpy_leaf, data)
    tmp = filepath.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        pickle.dump(np_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(filepath)

def load_pickle(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    try:
        with open(filepath, 'rb') as f:
            np_data = pickle.load(f)
        return jax.tree_util.tree_map(_to_jax_leaf, np_data)
    except Exception as e:
        print(f"  WARNING: corrupt checkpoint {filepath}, ignoring: {e}", flush=True)
        return None

def save_json(filepath, data):
    filepath = Path(filepath)
    tmp = filepath.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2)
    tmp.rename(filepath)

def load_json(filepath):
    filepath = Path(filepath)
    if not filepath.exists():
        return None
    try:
        with open(filepath) as f:
            return json.load(f)
    except Exception:
        return None

# ── Model + Noiser Init ───────────────────────────────────────────
def init_model(key):
    fp, p, sm, em = MODEL.rand_init(
        key, in_dim=OBS_DIM, out_dim=ACT_DIM,
        hidden_dims=[LAYER_SIZE] * N_LAYERS,
        use_bias=True, activation=ACTIVATION, dtype="float32")
    return fp, p, sm, em

def init_noiser(params, noiser_class, rank_spec, sigma=SIGMA, lr=LR):
    fnp, np_ = noiser_class.init_noiser(
        params, sigma, lr, solver=OPTIMIZER, solver_kwargs={}, rank=rank_spec)
    return fnp, np_

# ── Training Loop ─────────────────────────────────────────────────
def train(seed, noiser_class, rank_spec, label, max_gens=MAX_GENS,
          sigma=SIGMA, lr=LR, sigma_decay=SIGMA_DECAY, lr_decay=LR_DECAY,
          initial_state=None, return_checkpoint_at=None):
    """Train with 10-minute crash-recovery checkpoints.

    If return_checkpoint_at is set, returns (result_dict, checkpoint_state).
    Otherwise returns result_dict.
    """
    NOISER = noiser_class
    env = make_env()

    ckpt_file = CACHE_DIR / f"train_{label}_seed{seed}.pkl"
    resumed_gen = 0
    history = []
    best_fitness = -float("inf")

    existing_ckpt = load_pickle(ckpt_file)
    if existing_ckpt is not None and initial_state is None:
        resumed_gen = existing_ckpt['gen'] + 1
        history = existing_ckpt['history']
        best_fitness = existing_ckpt['best_fitness']
        frozen_params = existing_ckpt['frozen_params']
        current_params = existing_ckpt['params']
        scan_map = existing_ckpt['scan_map']
        es_map = existing_ckpt['es_map']
        es_tree_key = existing_ckpt['es_tree_key']
        frozen_noiser_params = existing_ckpt['frozen_noiser_params']
        current_noiser_params = existing_ckpt['noiser_params']
        episode_key = existing_ckpt['episode_key']
        print(f"\n  [{label} seed={seed}] RESUMING from gen {resumed_gen} "
              f"(best={best_fitness:.1f})", flush=True)
    else:
        if initial_state is not None:
            frozen_params = initial_state['frozen_params']
            current_params = initial_state['params']
            scan_map = initial_state['scan_map']
            es_map = initial_state['es_map']
            es_tree_key = initial_state['es_tree_key']
            frozen_noiser_params, current_noiser_params = init_noiser(
                current_params, noiser_class, rank_spec, sigma=sigma, lr=lr)
        else:
            key = jax.random.PRNGKey(seed)
            key, model_key, es_key = jax.random.split(key, 3)
            frozen_params, current_params, scan_map, es_map = init_model(model_key)
            es_tree_key = hs.models.common.simple_es_tree_key(
                current_params, es_key, scan_map)
            frozen_noiser_params, current_noiser_params = init_noiser(
                current_params, noiser_class, rank_spec, sigma=sigma, lr=lr)
        episode_key = jax.random.PRNGKey(seed + 10000)
        print(f"\n  [{label} seed={seed}] initialising...", flush=True)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(current_params))
    print(f"  [{label} seed={seed}] params: {n_params:,}", flush=True)

    # Two evaluation strategies: JIT (fast) and closure fallback (slow but safe)
    use_jit = globals().get('USE_JIT_EVAL', True)

    if use_jit:
        # JIT: gen_i and mid_i are traced abstract ints → ONE kernel compiled.
        @jax.jit
        def _eval_member(gen_i, mid_i, ep_key, fnp, cnp, fp, cp, estk):
            iterinfo = (gen_i, mid_i)
            def policy(obs):
                return jnp.clip(
                    MODEL.forward(NOISER, fnp, cnp, fp, cp, estk,
                                  iterinfo, obs), -1.0, 1.0)
            state = env.reset(ep_key)
            def scan_step(carry, _):
                st, total_reward, done = carry
                action = policy(st.obs)
                ns = env.step(st, action)
                reward = ns.reward * (1.0 - done)
                done = jnp.logical_or(done, ns.done)
                return (ns, total_reward + reward, done), None
            (_, total_reward, _), _ = jax.lax.scan(
                scan_step, (state, jnp.float32(0.0), jnp.bool_(False)),
                None, length=EPISODE_LENGTH)
            return total_reward

    def _eval_member_closure(gen, mid, ep_key):
        """Fallback: closure-based, one JIT trace per (gen, mid) combo."""
        iterinfo = (jnp.int32(gen), jnp.int32(mid))
        def policy(obs):
            return jnp.clip(
                MODEL.forward(NOISER, frozen_noiser_params,
                              current_noiser_params, frozen_params,
                              current_params, es_tree_key, iterinfo, obs),
                -1.0, 1.0)
        state = env.reset(ep_key)
        def scan_step(carry, _):
            st, total_reward, done = carry
            action = policy(st.obs)
            ns = env.step(st, action)
            reward = ns.reward * (1.0 - done)
            done = jnp.logical_or(done, ns.done)
            return (ns, total_reward + reward, done), None
        (_, total_reward, _), _ = jax.lax.scan(
            scan_step, (state, jnp.float32(0.0), jnp.bool_(False)),
            None, length=EPISODE_LENGTH)
        return total_reward

    checkpoint_state_for_return = None
    t0 = time.time()
    last_ckpt_time = time.time()

    for gen in range(resumed_gen, max_gens):
        t_gen = time.time()

        all_keys = jax.random.split(episode_key, POP_SIZE + 1)
        episode_key = all_keys[0]
        member_keys = all_keys[1:]

        rewards = []
        if use_jit:
            gen_i = jnp.int32(gen)
            for mid in range(POP_SIZE):
                r = _eval_member(
                    gen_i, jnp.int32(mid), member_keys[mid],
                    frozen_noiser_params, current_noiser_params,
                    frozen_params, current_params, es_tree_key)
                rewards.append(r)
        else:
            for mid in range(POP_SIZE):
                r = _eval_member_closure(gen, mid, member_keys[mid])
                rewards.append(float(r))

        gen_fitnesses_arr = jnp.array(rewards) if not use_jit else jnp.stack(rewards)

        # NaN handling
        nan_mask = jnp.isnan(gen_fitnesses_arr)
        nan_count = int(jnp.sum(nan_mask))
        if nan_count > 0:
            print(f"  \u26a0 gen {gen}: {nan_count}/{POP_SIZE} NaN fitnesses, "
                  f"replacing with {NAN_REPLACEMENT}", flush=True)
            gen_fitnesses_arr = jnp.where(nan_mask, NAN_REPLACEMENT, gen_fitnesses_arr)

        gen_mean = float(jnp.mean(gen_fitnesses_arr))
        gen_best = float(jnp.max(gen_fitnesses_arr))
        gen_var = float(jnp.var(gen_fitnesses_arr))
        if gen_best > best_fitness:
            best_fitness = gen_best

        history.append({
            "gen": gen, "mean_fitness": gen_mean, "best_fitness": gen_best,
            "best_so_far": best_fitness, "fitness_variance": gen_var,
            "nan_count": nan_count, "wall_s": time.time() - t0,
        })

        # ES update
        iterinfo = (jnp.full(POP_SIZE, gen, dtype=jnp.int32),
                    jnp.arange(POP_SIZE))
        converted = NOISER.convert_fitnesses(
            frozen_noiser_params, current_noiser_params, gen_fitnesses_arr)
        current_noiser_params, current_params = NOISER.do_updates(
            frozen_noiser_params, current_noiser_params, current_params,
            es_tree_key, converted, iterinfo, es_map)

        # Decay
        if sigma_decay < 1.0:
            current_noiser_params['sigma'] = current_noiser_params['sigma'] * sigma_decay
        if lr_decay < 1.0 and 'lr' in current_noiser_params:
            current_noiser_params['lr'] = current_noiser_params['lr'] * lr_decay

        # Return checkpoint (Phase 1)
        if return_checkpoint_at is not None and gen == return_checkpoint_at:
            checkpoint_state_for_return = {
                'frozen_params': frozen_params, 'params': current_params,
                'scan_map': scan_map, 'es_map': es_map,
                'es_tree_key': es_tree_key,
                'frozen_noiser_params': frozen_noiser_params,
                'noiser_params': current_noiser_params,
            }
            print(f"  [{label} seed={seed}] return-checkpoint at gen {gen}", flush=True)

        # Crash-recovery checkpoint every 10 min
        now = time.time()
        if now - last_ckpt_time >= CHECKPOINT_INTERVAL_S:
            save_pickle(ckpt_file, {
                'gen': gen, 'history': history, 'best_fitness': best_fitness,
                'frozen_params': frozen_params, 'params': current_params,
                'scan_map': scan_map, 'es_map': es_map,
                'es_tree_key': es_tree_key,
                'frozen_noiser_params': frozen_noiser_params,
                'noiser_params': current_noiser_params,
                'episode_key': episode_key,
            })
            last_ckpt_time = now
            print(f"  [{label} seed={seed}] checkpoint at gen {gen} "
                  f"({now - t0:.0f}s elapsed)", flush=True)

        gen_elapsed = time.time() - t_gen
        if gen % 5 == 0 or gen == max_gens - 1:
            print(f"    gen {gen:4d}  mean={gen_mean:.1f}  best_so_far={best_fitness:.1f}"
                  f"  sigma={float(current_noiser_params['sigma']):.4f}"
                  f"  ({gen_elapsed:.1f}s)", flush=True)

    total_time = time.time() - t0
    final_mean = history[-1]["mean_fitness"] if history else 0.0
    print(f"  [{label} seed={seed}] done {total_time:.0f}s, best={best_fitness:.1f}",
          flush=True)

    # Clean up training checkpoint
    if ckpt_file.exists():
        ckpt_file.unlink()

    result = {
        "method": label, "seed": seed, "best_fitness": best_fitness,
        "final_mean_fitness": final_mean, "generations": len(history),
        "wall_seconds": total_time, "history": history,
    }
    if return_checkpoint_at is not None:
        return result, checkpoint_state_for_return
    return result


# ── Phase 1: Elevation ────────────────────────────────────────────
def run_phase1():
    print(f"\n{'='*60}")
    print("SENSITIVITY PILOT — Phase 1 (Elevation / Magnitude)")
    print(f"{'='*60}")
    print(f"  Checkpoint gens: {PHASE1_CHECKPOINT_GENS}, "
          f"Elevation gens: {PHASE1_ELEVATION_GENS}, Seeds: {PILOT_SEEDS}")

    unique_shapes = list(LAYER_SHAPES.items())
    phase1_results = {}

    for seed in PILOT_SEEDS:
        ckpt_cache = CACHE_DIR / f"p1_checkpoint_seed{seed}.json"
        ckpt_state_file = CACHE_DIR / f"p1_checkpoint_state_seed{seed}.pkl"
        cached = load_json(ckpt_cache)

        if cached is not None and load_pickle(ckpt_state_file) is not None:
            checkpoint_fitness = cached["checkpoint_fitness"]
            checkpoint_state = load_pickle(ckpt_state_file)
            print(f"\n  Phase 1 — Checkpoint seed={seed} from cache "
                  f"(fitness={checkpoint_fitness:.1f})", flush=True)
        else:
            print(f"\n  Phase 1 — Training shared checkpoint (seed={seed})...",
                  flush=True)
            checkpoint_result, checkpoint_state = train(
                seed, EggRoll, RANK, f"phase1_checkpoint",
                max_gens=PHASE1_CHECKPOINT_GENS,
                return_checkpoint_at=PHASE1_CHECKPOINT_GENS - 1)
            checkpoint_fitness = checkpoint_result["final_mean_fitness"]
            save_json(ckpt_cache, {"checkpoint_fitness": checkpoint_fitness})
            save_pickle(ckpt_state_file, checkpoint_state)
            print(f"  Checkpoint fitness (seed={seed}): {checkpoint_fitness:.1f}",
                  flush=True)

        if checkpoint_state is None:
            print(f"  ERROR: checkpoint not captured for seed={seed}!", flush=True)
            continue

        for layer_name, layer_shape in unique_shapes:
            elev_cache = CACHE_DIR / f"p1_elevate_{layer_name}_seed{seed}.json"
            cached_elev = load_json(elev_cache)
            if cached_elev is not None:
                phase1_results[f"{layer_name}_seed{seed}"] = cached_elev
                print(f"  Phase 1 — {layer_name} seed={seed} from cache "
                      f"(|delta|={cached_elev['abs_delta']:.1f})", flush=True)
                continue

            print(f"\n  Phase 1 — Elevating {layer_name} {layer_shape} to r=8 "
                  f"(seed={seed}):", flush=True)
            elevated_spec = {s: RANK for s in LAYER_SHAPES.values()}
            elevated_spec[layer_shape] = 8
            elevated_result = train(
                seed, LWREggRoll, elevated_spec,
                f"phase1_elevate_{layer_name}",
                max_gens=PHASE1_ELEVATION_GENS,
                initial_state=checkpoint_state)
            elevated_fitness = elevated_result["final_mean_fitness"]
            delta = abs(elevated_fitness - checkpoint_fitness)
            entry = {
                "layer": layer_name, "seed": seed,
                "checkpoint_fitness": checkpoint_fitness,
                "elevated_fitness": elevated_fitness, "abs_delta": delta,
            }
            phase1_results[f"{layer_name}_seed{seed}"] = entry
            save_json(elev_cache, entry)
            print(f"  {layer_name} (seed={seed}): elevated={elevated_fitness:.1f}, "
                  f"|delta|={delta:.1f}", flush=True)

    phase1_summary = {}
    for layer_name, _ in unique_shapes:
        deltas = [phase1_results[f"{layer_name}_seed{s}"]["abs_delta"]
                  for s in PILOT_SEEDS
                  if f"{layer_name}_seed{s}" in phase1_results]
        phase1_summary[layer_name] = {
            "mean_abs_delta": float(np.mean(deltas)),
            "per_seed_deltas": deltas,
        }

    phase1_ordering = sorted(phase1_summary.keys(),
                             key=lambda n: phase1_summary[n]["mean_abs_delta"],
                             reverse=True)
    phase1_ordering_str = " > ".join(phase1_ordering)
    save_json(CACHE_DIR / "phase1_complete.json",
              {"ordering": phase1_ordering_str, "summary": phase1_summary})

    print(f"\n  Phase 1 ordering: {phase1_ordering_str}")
    for name in phase1_ordering:
        s = phase1_summary[name]
        print(f"    {name}: mean |delta| = {s['mean_abs_delta']:.1f}")
    return phase1_ordering_str, phase1_summary


# ── Phase 2: Causal Ablation ──────────────────────────────────────
def run_phase2():
    print(f"\n{'='*60}")
    print("SENSITIVITY PILOT — Phase 2 (Causal Ablation)")
    print(f"{'='*60}")
    print(f"  Gens: {PHASE2_GENS}, Seeds: {PILOT_SEEDS}, "
          f"Baseline: r={RANK}, Ablation: r=1")

    unique_shapes = list(LAYER_SHAPES.items())

    baseline_fitnesses = []
    for seed in PILOT_SEEDS:
        bl_cache = CACHE_DIR / f"p2_baseline_seed{seed}.json"
        cached = load_json(bl_cache)
        if cached is not None:
            baseline_fitnesses.append(cached["final_mean_fitness"])
            print(f"  Baseline seed={seed} from cache "
                  f"({cached['final_mean_fitness']:.1f})", flush=True)
        else:
            r = train(seed, EggRoll, RANK, f"pilot_baseline_r{RANK}",
                      max_gens=PHASE2_GENS)
            baseline_fitnesses.append(r["final_mean_fitness"])
            save_json(bl_cache, {"final_mean_fitness": r["final_mean_fitness"]})
    baseline_mean = float(np.mean(baseline_fitnesses))
    print(f"  Baseline mean: {baseline_mean:.1f}", flush=True)

    degradations = {}
    for layer_name, layer_shape in unique_shapes:
        abl_fitnesses = []
        for seed in PILOT_SEEDS:
            abl_cache = CACHE_DIR / f"p2_ablate_{layer_name}_seed{seed}.json"
            cached = load_json(abl_cache)
            if cached is not None:
                abl_fitnesses.append(cached["final_mean_fitness"])
                print(f"  Ablate {layer_name} seed={seed} from cache", flush=True)
            else:
                ablated_spec = {s: RANK for s in LAYER_SHAPES.values()}
                ablated_spec[layer_shape] = 1
                r = train(seed, LWREggRoll, ablated_spec,
                          f"pilot_ablate_{layer_name}", max_gens=PHASE2_GENS)
                abl_fitnesses.append(r["final_mean_fitness"])
                save_json(abl_cache,
                          {"final_mean_fitness": r["final_mean_fitness"]})
        ablated_mean = float(np.mean(abl_fitnesses))
        degradation = baseline_mean - ablated_mean
        degradations[layer_name] = {
            "shape": layer_shape,
            "mean_fitness": ablated_mean,
            "degradation": degradation,
        }
        print(f"  {layer_name}: mean={ablated_mean:.1f}, "
              f"degradation={degradation:.1f}", flush=True)

    ordering = sorted(degradations.keys(),
                      key=lambda n: degradations[n]["degradation"], reverse=True)
    ordering_str = " > ".join(ordering)
    save_json(CACHE_DIR / "phase2_complete.json", {
        "baseline_mean": baseline_mean,
        "degradations": {k: {**v, "shape": str(v["shape"])}
                         for k, v in degradations.items()},
        "ordering": ordering_str,
    })
    print(f"\n  Phase 2 ordering: {ordering_str}")
    return ordering, degradations, baseline_mean


# ── Phase 3: Binary Inclusion ─────────────────────────────────────
def run_phase3(ordering, degradations):
    least_sensitive = ordering[-1]
    ls_shape = degradations[least_sensitive]["shape"]
    print(f"\n{'='*60}")
    print(f"SENSITIVITY PILOT — Phase 3 (Binary Inclusion)")
    print(f"{'='*60}")
    print(f"  Layer: {least_sensitive} {ls_shape}, r=0 vs r=1")

    r1_mean = degradations[least_sensitive]["mean_fitness"]

    frozen_fitnesses = []
    for seed in PILOT_SEEDS:
        p3_cache = CACHE_DIR / f"p3_freeze_{least_sensitive}_seed{seed}.json"
        cached = load_json(p3_cache)
        if cached is not None:
            frozen_fitnesses.append(cached["final_mean_fitness"])
            print(f"  Freeze {least_sensitive} seed={seed} from cache", flush=True)
        else:
            frozen_spec = {s: RANK for s in LAYER_SHAPES.values()}
            frozen_spec[ls_shape] = 0
            r = train(seed, LWREggRoll, frozen_spec,
                      f"pilot_freeze_{least_sensitive}", max_gens=PHASE3_GENS)
            frozen_fitnesses.append(r["final_mean_fitness"])
            save_json(p3_cache, {"final_mean_fitness": r["final_mean_fitness"]})

    r0_mean = float(np.mean(frozen_fitnesses))
    freeze_justified = r0_mean >= r1_mean - 5.0
    phase3_decision = 0 if freeze_justified else 1

    save_json(CACHE_DIR / "phase3_complete.json", {
        "least_sensitive": least_sensitive,
        "r1_mean": r1_mean, "r0_mean": r0_mean,
        "freeze_justified": freeze_justified, "decision": phase3_decision,
    })
    print(f"  r=1: {r1_mean:.1f}, r=0: {r0_mean:.1f} -> rank {phase3_decision}")
    return least_sensitive, phase3_decision, r0_mean, r1_mean


# ── Full Pilot ─────────────────────────────────────────────────────
def run_sensitivity_pilot():
    # Phase 1
    p1c = load_json(CACHE_DIR / "phase1_complete.json")
    if p1c is not None:
        p1_ord_str, p1_summary = p1c["ordering"], p1c["summary"]
        print(f"\n  Phase 1 from cache: {p1_ord_str}")
    else:
        p1_ord_str, p1_summary = run_phase1()

    # Phase 2
    p2c = load_json(CACHE_DIR / "phase2_complete.json")
    if p2c is not None:
        ordering_str = p2c["ordering"]
        ordering = ordering_str.split(" > ")
        baseline_mean = p2c["baseline_mean"]
        degradations = {}
        for k, v in p2c["degradations"].items():
            shape = ast.literal_eval(v["shape"])
            degradations[k] = {**v, "shape": shape}
        print(f"\n  Phase 2 from cache: {ordering_str}")
    else:
        ordering, degradations, baseline_mean = run_phase2()
        ordering_str = " > ".join(ordering)

    # Phase 3
    p3c = load_json(CACHE_DIR / "phase3_complete.json")
    if p3c is not None:
        least_sensitive = p3c["least_sensitive"]
        phase3_decision = p3c["decision"]
        r0_mean, r1_mean = p3c["r0_mean"], p3c["r1_mean"]
        print(f"\n  Phase 3 from cache: {least_sensitive} -> rank {phase3_decision}")
    else:
        least_sensitive, phase3_decision, r0_mean, r1_mean = \
            run_phase3(ordering, degradations)

    # Build allocation (rank 4 cap)
    rank_tiers = [4, 2]
    allocation = {}
    alloc_named = {}
    for i, layer_name in enumerate(ordering):
        shape = LAYER_SHAPES[layer_name]
        if i == len(ordering) - 1:
            allocation[shape] = phase3_decision
        else:
            allocation[shape] = rank_tiers[min(i, len(rank_tiers) - 1)]
        alloc_named[layer_name] = allocation[shape]

    actual_budget = (alloc_named["input"] + 2 * alloc_named["hidden"]
                     + alloc_named["output"])
    alloc_label = "_".join(
        str(alloc_named[n]) for n in ["input", "hidden", "output"])

    save_json(RESULTS_DIR / "pilot_results.json", {
        "phase1": {"ordering": p1_ord_str, "summary": p1_summary},
        "phase2": {
            "baseline_mean": baseline_mean,
            "degradations": {k: {**v, "shape": str(v["shape"])}
                             for k, v in degradations.items()},
            "ordering": ordering_str,
        },
        "phase3": {
            "least_sensitive": least_sensitive,
            "r1_mean": r1_mean, "r0_mean": r0_mean,
            "freeze_justified": phase3_decision == 0,
        },
        "allocation": alloc_named, "allocation_label": alloc_label,
        "rank_budget": actual_budget,
        "rank_spec": {str(k): v for k, v in allocation.items()},
    })

    print(f"\n{'='*60}")
    print(f"  PILOT SUMMARY")
    print(f"  Phase 1 (magnitude): {p1_ord_str}")
    print(f"  Phase 2 (direction): {ordering_str}")
    print(f"  Phase 3: {least_sensitive} -> rank {phase3_decision}")
    print(f"  Allocation: {alloc_named} (label: {alloc_label})")
    print(f"  Budget: {actual_budget} (vs uniform r=4: 16)")
    print(f"{'='*60}\n", flush=True)
    return allocation, alloc_named, ordering_str, alloc_label


# ── Main ──────────────────────────────────────────────────────────
def main():
    actual_params = (LAYER_SHAPES["input"][0] * LAYER_SHAPES["input"][1]
                     + 2 * LAYER_SHAPES["hidden"][0] * LAYER_SHAPES["hidden"][1]
                     + LAYER_SHAPES["output"][0] * LAYER_SHAPES["output"][1])
    print("=" * 60)
    print("BRAX ANT — FULL LWR-EGGROLL EXPERIMENT")
    arch = [OBS_DIM] + [LAYER_SIZE] * N_LAYERS + [ACT_DIM]
    print(f"Architecture: {arch}  ({actual_params:,} weight params)")
    print(f"POP={POP_SIZE}, SIGMA={SIGMA}, LR={LR}, MAX_GENS={MAX_GENS}")
    print(f"SIGMA_DECAY={SIGMA_DECAY}, LR_DECAY={LR_DECAY}")
    print(f"Seeds: {SEEDS}, Checkpoint: {CHECKPOINT_INTERVAL_S}s")
    print("=" * 60, flush=True)

    # ── Pre-flight: verify JIT evaluation works ──
    print("\nPre-flight: testing JIT evaluation (should complete in 1-3 min)...",
          flush=True)
    print("  (If stuck >5 min, press Ctrl+C to use fallback mode)", flush=True)
    JIT_EVAL_WORKS = False
    try:
        _pf_env = make_env()
        _pf_key = jax.random.PRNGKey(9999)
        _pf_key, _pf_mk, _pf_ek = jax.random.split(_pf_key, 3)
        _pf_fp, _pf_cp, _pf_sm, _pf_em = init_model(_pf_mk)
        _pf_estk = hs.models.common.simple_es_tree_key(_pf_cp, _pf_ek, _pf_sm)
        _pf_keys = jax.random.split(jax.random.PRNGKey(42), 4)

        # Test 1: EggRoll (uniform rank)
        _pf_fnp, _pf_cnp = init_noiser(_pf_cp, EggRoll, RANK)
        @jax.jit
        def _pf_eval_egg(gen_i, mid_i, ep_key, fnp, cnp, fp, cp, estk):
            iterinfo = (gen_i, mid_i)
            def policy(obs):
                return jnp.clip(MODEL.forward(
                    EggRoll, fnp, cnp, fp, cp, estk, iterinfo, obs), -1.0, 1.0)
            state = _pf_env.reset(ep_key)
            def scan_step(carry, _):
                st, tr, done = carry
                ns = _pf_env.step(st, policy(st.obs))
                return (ns, tr + ns.reward * (1.0 - done),
                        jnp.logical_or(done, ns.done)), None
            (_, tr, _), _ = jax.lax.scan(
                scan_step, (state, jnp.float32(0.0), jnp.bool_(False)),
                None, length=10)
            return tr

        t_pf = time.time()
        _r0 = float(_pf_eval_egg(jnp.int32(0), jnp.int32(0), _pf_keys[0],
                                  _pf_fnp, _pf_cnp, _pf_fp, _pf_cp, _pf_estk))
        _r1 = float(_pf_eval_egg(jnp.int32(0), jnp.int32(1), _pf_keys[1],
                                  _pf_fnp, _pf_cnp, _pf_fp, _pf_cp, _pf_estk))
        print(f"  EggRoll JIT OK: m0={_r0:.2f}, m1={_r1:.2f} "
              f"({time.time()-t_pf:.1f}s)", flush=True)

        # Test 2: LWREggRoll (per-shape rank dict)
        _pf_lwr_spec = {s: RANK for s in LAYER_SHAPES.values()}
        _pf_lwr_spec[LAYER_SHAPES["output"]] = 1  # Simulate ablation
        _pf_fnp2, _pf_cnp2 = init_noiser(_pf_cp, LWREggRoll, _pf_lwr_spec)
        @jax.jit
        def _pf_eval_lwr(gen_i, mid_i, ep_key, fnp, cnp, fp, cp, estk):
            iterinfo = (gen_i, mid_i)
            def policy(obs):
                return jnp.clip(MODEL.forward(
                    LWREggRoll, fnp, cnp, fp, cp, estk, iterinfo, obs),
                    -1.0, 1.0)
            state = _pf_env.reset(ep_key)
            def scan_step(carry, _):
                st, tr, done = carry
                ns = _pf_env.step(st, policy(st.obs))
                return (ns, tr + ns.reward * (1.0 - done),
                        jnp.logical_or(done, ns.done)), None
            (_, tr, _), _ = jax.lax.scan(
                scan_step, (state, jnp.float32(0.0), jnp.bool_(False)),
                None, length=10)
            return tr

        t_pf2 = time.time()
        _r2 = float(_pf_eval_lwr(jnp.int32(0), jnp.int32(0), _pf_keys[2],
                                  _pf_fnp2, _pf_cnp2, _pf_fp, _pf_cp, _pf_estk))
        print(f"  LWREggRoll JIT OK: m0={_r2:.2f} ({time.time()-t_pf2:.1f}s)",
              flush=True)

        JIT_EVAL_WORKS = True
        print(f"  Pre-flight PASSED ({time.time()-t_pf:.0f}s total)", flush=True)
    except KeyboardInterrupt:
        print(f"\n  Pre-flight interrupted — using closure fallback.", flush=True)
    except Exception as e:
        print(f"\n  Pre-flight FAILED: {e}", flush=True)
        print(f"  Using closure-based evaluation (slower but safe).", flush=True)

    global USE_JIT_EVAL
    USE_JIT_EVAL = JIT_EVAL_WORKS

    # Pilot
    pilot_file = RESULTS_DIR / "pilot_results.json"
    if pilot_file.exists():
        print("\nPilot results found, loading...", flush=True)
        with open(pilot_file) as f:
            pr = json.load(f)
        alloc_named = pr["allocation"]
        ordering_str = pr["phase2"]["ordering"]
        allocation = {}
        for ln, rv in alloc_named.items():
            allocation[LAYER_SHAPES[ln]] = rv
        alloc_label = pr["allocation_label"]
        print(f"  Ordering: {ordering_str}")
        print(f"  Allocation: {alloc_named}")
    else:
        allocation, alloc_named, ordering_str, alloc_label = \
            run_sensitivity_pilot()

    # Methods
    uniform_budget = 4 * 4
    lwr_budget = (alloc_named["input"] + 2 * alloc_named["hidden"]
                  + alloc_named["output"])
    METHODS = [
        ("eggroll_r4", EggRoll, RANK, MAX_GENS),
        ("eggroll_r1", EggRoll, 1, MAX_GENS_FLOOR),
        (f"lwr_{alloc_label}", LWREggRoll, allocation, MAX_GENS),
    ]

    print(f"\nMethods:")
    for mn, nc, rs, mg in METHODS:
        if isinstance(rs, int):
            b = rs * 4
        elif isinstance(rs, dict):
            b = (rs.get(LAYER_SHAPES["input"], 0)
                 + 2 * rs.get(LAYER_SHAPES["hidden"], 0)
                 + rs.get(LAYER_SHAPES["output"], 0))
        else:
            b = "?"
        print(f"  {mn}: {nc.__name__}, budget={b}, gens={mg}")

    # Run
    all_results = {
        "pilot": {"allocation": alloc_named, "ordering": ordering_str},
        "config": {
            "pop_size": POP_SIZE, "sigma": SIGMA, "lr": LR,
            "sigma_decay": SIGMA_DECAY, "lr_decay": LR_DECAY,
            "max_gens": MAX_GENS, "max_gens_floor": MAX_GENS_FLOOR,
            "episode_length": EPISODE_LENGTH,
            "activation": ACTIVATION, "optimizer": "sgd",
            "architecture": arch,
        },
    }

    for mn, noiser_class, rank_spec, method_max_gens in METHODS:
        print(f"\n{'~'*60}")
        print(f"METHOD: {mn} ({method_max_gens} gens)")
        print(f"{'~'*60}", flush=True)

        method_results = []
        for seed in SEEDS:
            result_file = RESULTS_DIR / f"{mn}_seed{seed}.json"
            if result_file.exists():
                print(f"  [{mn} seed={seed}] exists, skipping.", flush=True)
                with open(result_file) as f:
                    r = json.load(f)
                method_results.append(r)
                continue

            r = train(seed, noiser_class, rank_spec, mn,
                      max_gens=method_max_gens,
                      sigma_decay=SIGMA_DECAY, lr_decay=LR_DECAY)
            save_json(result_file, r)
            method_results.append(r)

        bests = [r["best_fitness"] for r in method_results]
        means = [r["final_mean_fitness"] for r in method_results]
        all_results[mn] = {
            "mean_best": float(np.mean(bests)),
            "std_best": float(np.std(bests)),
            "mean_final": float(np.mean(means)),
            "per_seed_best": bests,
        }

    # Summary
    saving_pct = (1 - lwr_budget / uniform_budget) * 100
    print(f"\n{'='*60}")
    print("BRAX ANT — RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Pilot ordering: {ordering_str}")
    print(f"Pilot allocation: {alloc_named}")
    print(f"\n{'Method':35s} {'Budget':>8s} {'Mean Best':>12s} "
          f"{'Std':>8s} {'Gens':>6s}")
    print("-" * 70)
    for mn, _, _, mg in METHODS:
        if mn not in all_results:
            continue
        s = all_results[mn]
        if mn == "eggroll_r4":
            b = uniform_budget
        elif mn == "eggroll_r1":
            b = 1 * 4
        else:
            b = lwr_budget
        print(f"{mn:35s} {b:>8d} {s['mean_best']:>12.1f} "
              f"{s['std_best']:>8.1f} {mg:>6d}")
    print(f"\nEfficiency: LWR budget {lwr_budget} vs uniform {uniform_budget} "
          f"= {saving_pct:.0f}% reduction")

    save_json(RESULTS_DIR / "summary.json", {
        "env": ENV_NAME, "architecture": arch,
        "note": "Brax Ant LWR-EGGROLL, rank 4 cap, 3-phase pilot",
        "efficiency": {"lwr_budget": lwr_budget,
                       "uniform_budget": uniform_budget,
                       "saving_pct": saving_pct},
        "results": all_results,
    })
    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
