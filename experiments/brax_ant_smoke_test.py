"""Brax Ant — MINIATURE SMOKE TEST
Small config to verify the training loop works locally before university GPU.
POP=32, MAX_GENS=5, 1 seed. Should complete in ~2-5 minutes.
"""
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.50"

import time
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax

from brax import envs
import hyperscalees as hs
from hyperscalees.noiser.eggroll import EggRoll

# ── Miniature config ───────────────────────────────────────────────
ENV_NAME = "ant"
POP_SIZE = 32
SIGMA = 0.2
LR = 0.1
RANK = 4
MAX_GENS = 5
EPISODE_LENGTH = 200  # short episodes for smoke test
N_LAYERS = 3
LAYER_SIZE = 256
ACTIVATION = "pqn"

MODEL = hs.models.common.MLP
NOISER = EggRoll

print("=" * 60)
print("BRAX ANT — MINIATURE SMOKE TEST")
print(f"POP={POP_SIZE}, GENS={MAX_GENS}, EP_LEN={EPISODE_LENGTH}")
print("=" * 60, flush=True)

# ── Step 1: Environment ───────────────────────────────────────────
print("\n[1] Loading Brax Ant...", flush=True)
env = envs.get_environment(ENV_NAME)
OBS_DIM = env.observation_size
ACT_DIM = env.action_size
print(f"    obs_dim={OBS_DIM}, act_dim={ACT_DIM}")

# ── Step 2: Model init ───────────────────────────────────────────
print("\n[2] Initialising model...", flush=True)
key = jax.random.key(0)
model_key = jax.random.fold_in(key, 0)
es_key = jax.random.fold_in(key, 1)

frozen_params, params, scan_map, es_map = MODEL.rand_init(
    model_key,
    in_dim=OBS_DIM,
    out_dim=ACT_DIM,
    hidden_dims=[LAYER_SIZE] * N_LAYERS,
    use_bias=True,
    activation=ACTIVATION,
    dtype="float32",
)
es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)

# Print parameter shapes
print("    Parameter shapes:")
def print_shapes(prefix, tree):
    if isinstance(tree, dict):
        for k in tree:
            print_shapes(f"{prefix}/{k}", tree[k])
    elif hasattr(tree, 'shape'):
        print(f"      {prefix}: {tree.shape} (ndim={tree.ndim})")
print_shapes("params", params)

# Collect 2D shapes for layer identification
layer_shapes = {}
def collect_shapes(tree):
    if isinstance(tree, dict):
        for k in tree:
            collect_shapes(tree[k])
    elif hasattr(tree, 'shape') and tree.ndim == 2:
        shape = tuple(tree.shape)
        if shape not in layer_shapes:
            if shape[1] == OBS_DIM:
                layer_shapes[shape] = "input"
            elif shape[0] == ACT_DIM:
                layer_shapes[shape] = "output"
            else:
                layer_shapes[shape] = "hidden"
collect_shapes(params)
print(f"    Layer shapes for LWR: {layer_shapes}")

n_params = sum(x.size for x in jax.tree.leaves(params))
print(f"    Total params: {n_params:,}")

# ── Step 3: Noiser init ──────────────────────────────────────────
print("\n[3] Initialising EggRoll noiser (rank={})...".format(RANK), flush=True)
frozen_noiser_params, noiser_params = NOISER.init_noiser(
    params, SIGMA, LR,
    solver=optax.sgd,
    solver_kwargs={},
    rank=RANK,
)
print("    Noiser initialised OK")
print(f"    noiser_params type: {type(noiser_params)}")

# ── Step 4: Single episode rollout test ──────────────────────────
print("\n[4] Testing single episode rollout...", flush=True)

def run_episode(policy_fn, eval_key, episode_length):
    """Run one Brax episode, return total reward."""
    state = env.reset(eval_key)
    
    def scan_step(carry, _):
        state, total_reward, done = carry
        obs = state.obs
        action = policy_fn(obs)
        action = jnp.clip(action, -1.0, 1.0)
        next_state = env.step(state, action)
        reward = next_state.reward * (1.0 - done.astype(jnp.float32))
        done = jnp.logical_or(done, next_state.done)
        return (next_state, total_reward + reward, done), None
    
    init_carry = (state, jnp.float32(0.0), jnp.bool_(False))
    (_, total_reward, _), _ = jax.lax.scan(
        scan_step, init_carry, None, length=episode_length
    )
    return total_reward

# Clean policy (no noise)
def clean_policy(obs):
    output = MODEL.forward(
        NOISER, frozen_noiser_params, noiser_params,
        frozen_params, params, es_tree_key, None, obs
    )
    return jnp.clip(output, -1.0, 1.0)

eval_key = jax.random.fold_in(key, 99)
t0 = time.time()
reward = run_episode(clean_policy, eval_key, EPISODE_LENGTH)
elapsed = time.time() - t0
print(f"    Clean policy reward: {float(reward):.1f} ({elapsed:.1f}s, includes JIT)")

# ── Step 5: Noised policy test (single member) ──────────────────
print("\n[5] Testing noised policy (single member)...", flush=True)

def noised_policy(obs, epoch, member_id):
    iterinfo = (epoch, member_id)
    output = MODEL.forward(
        NOISER, frozen_noiser_params, noiser_params,
        frozen_params, params, es_tree_key, iterinfo, obs
    )
    return jnp.clip(output, -1.0, 1.0)

t0 = time.time()
reward_noised = run_episode(
    lambda obs: noised_policy(obs, jnp.int32(0), jnp.int32(0)),
    eval_key, EPISODE_LENGTH
)
elapsed = time.time() - t0
print(f"    Noised policy reward: {float(reward_noised):.1f} ({elapsed:.1f}s)")

# ── Step 6: Population evaluation ────────────────────────────────
print(f"\n[6] Evaluating population of {POP_SIZE} (sequential)...", flush=True)
t0 = time.time()

eval_keys = jax.random.split(jax.random.fold_in(key, 42), POP_SIZE)
fitnesses = []

for member_id in range(POP_SIZE):
    def member_policy(obs, mid=member_id):
        return noised_policy(obs, jnp.int32(0), jnp.int32(mid))
    
    reward = run_episode(member_policy, eval_keys[member_id], EPISODE_LENGTH)
    fitnesses.append(float(reward))

fitnesses = jnp.array(fitnesses)
elapsed = time.time() - t0
print(f"    Pop eval: mean={float(jnp.mean(fitnesses)):.1f}, "
      f"best={float(jnp.max(fitnesses)):.1f}, "
      f"std={float(jnp.std(fitnesses)):.1f} ({elapsed:.1f}s)")

# ── Step 7: Update test ─────────────────────────────────────────
print("\n[7] Testing noiser update...", flush=True)
t0 = time.time()

iterinfo = (
    jnp.full(POP_SIZE, 0, dtype=jnp.int32),
    jnp.arange(POP_SIZE),
)
converted_fitnesses = NOISER.convert_fitnesses(
    frozen_noiser_params, noiser_params, fitnesses
)
new_noiser_params, new_params = NOISER.do_updates(
    frozen_noiser_params, noiser_params, params,
    es_tree_key, converted_fitnesses, iterinfo, es_map
)
elapsed = time.time() - t0
print(f"    Update completed ({elapsed:.1f}s)")

# ── Step 8: Mini training loop ───────────────────────────────────
print(f"\n[8] Mini training loop ({MAX_GENS} generations)...", flush=True)

current_noiser_params = noiser_params
current_params = params
t_total = time.time()

for gen in range(MAX_GENS):
    t_gen = time.time()
    
    # Generate eval keys
    gen_key = jax.random.fold_in(key, gen + 1000)
    member_keys = jax.random.split(gen_key, POP_SIZE)
    
    # Evaluate population (sequential for smoke test)
    gen_fitnesses = []
    for mid in range(POP_SIZE):
        def member_policy(obs, mid=mid, gen=gen):
            iterinfo = (jnp.int32(gen), jnp.int32(mid))
            output = MODEL.forward(
                NOISER, frozen_noiser_params, current_noiser_params,
                frozen_params, current_params, es_tree_key, iterinfo, obs
            )
            return jnp.clip(output, -1.0, 1.0)
        
        reward = run_episode(member_policy, member_keys[mid], EPISODE_LENGTH)
        gen_fitnesses.append(float(reward))
    
    gen_fitnesses = jnp.array(gen_fitnesses)
    gen_mean = float(jnp.mean(gen_fitnesses))
    gen_best = float(jnp.max(gen_fitnesses))
    
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
    
    gen_elapsed = time.time() - t_gen
    print(f"    gen {gen}  mean={gen_mean:.1f}  best={gen_best:.1f}  ({gen_elapsed:.1f}s)", flush=True)

total_elapsed = time.time() - t_total
print(f"\n    Training loop completed in {total_elapsed:.1f}s")

# ── Summary ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SMOKE TEST SUMMARY")
print(f"{'='*60}")
print(f"  Environment: {ENV_NAME} (obs={OBS_DIM}, act={ACT_DIM})")
print(f"  Params: {n_params:,}")
print(f"  Layer shapes: {layer_shapes}")
print(f"  All steps passed ✓")
print(f"  Total time: {time.time() - t0:.0f}s")
print(f"\n  Ready for full run on university GPU.")
print(f"  Change: POP_SIZE=2048, MAX_GENS=500, EPISODE_LENGTH=1000, SEEDS=[0,1,2]")
