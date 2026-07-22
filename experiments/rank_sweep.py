import os
import json
import time
import jax
from huggingface_hub.constants import HF_HOME
jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import optax
import jax.numpy as jnp
import hyperscalees as hs
from torchvision import datasets

mnist_root = os.path.expanduser("~/dissertation/data/mnist")
train_ds = datasets.MNIST(root=mnist_root, train=True, download=True)
test_ds = datasets.MNIST(root=mnist_root, train=False, download=True)
train_images = jnp.array(train_ds.data.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
train_labels = jnp.array(train_ds.targets.numpy(), dtype=jnp.int32)
test_images = jnp.array(test_ds.data.numpy().reshape(-1, 784) / 255.0, dtype=jnp.float32)
test_labels = jnp.array(test_ds.targets.numpy(), dtype=jnp.int32)

SIGMA, LR, NUM_ENVS, BATCH_SIZE = 0.03, 0.01, 256, 512
NUM_GENERATIONS, EVAL_EVERY = 5000, 50
RANK_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]

NOISER = hs.noiser.eggroll.EggRoll
MODEL = hs.models.common.MLP


def train_at_rank(rank, seed=0):
    key = jax.random.key(seed)
    model_key = jax.random.fold_in(key, 0)
    es_key = jax.random.fold_in(key, 1)
    data_key = jax.random.fold_in(key, 2)

    frozen_params, params, scan_map, es_map = MODEL.rand_init(
        model_key, in_dim=784, out_dim=10, hidden_dims=[128, 128],
        use_bias=True, activation="relu", dtype="float32",
    )
    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)
    frozen_noiser_params, noiser_params = NOISER.init_noiser(
        params, SIGMA, LR, solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999}, rank=rank,
    )

    _forward_single = lambda n, p, i, x: MODEL.forward(
        NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, i, x
    )
    _forward_over_batch = jax.vmap(_forward_single, in_axes=(None, None, None, 0))
    jit_forward = jax.jit(jax.vmap(_forward_over_batch, in_axes=(None, None, 0, None)))
    _eval_single = lambda n, p, x: MODEL.forward(
        NOISER, frozen_noiser_params, n, frozen_params, p, es_tree_key, None, x
    )
    jit_forward_eval = jax.jit(jax.vmap(_eval_single, in_axes=(None, None, 0)))
    jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
        frozen_noiser_params, n, p, es_tree_key, f, i, es_map
    ))

    @jax.jit
    def batch_fitness(logits, labels):
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        batch_idx = jnp.arange(labels.shape[0])
        picked = log_probs[:, batch_idx, labels]
        return jnp.mean(picked, axis=-1)

    history = {"gen": [], "test_acc": [], "wallclock": []}
    t0 = time.time()
    for gen in range(NUM_GENERATIONS):
        data_key, batch_key = jax.random.split(data_key)
        batch_idx = jax.random.choice(batch_key, 60000, (BATCH_SIZE,), replace=False)
        x_batch = train_images[batch_idx]
        y_batch = train_labels[batch_idx]
        iterinfo = (jnp.full(NUM_ENVS, gen, dtype=jnp.int32), jnp.arange(NUM_ENVS))
        logits = jit_forward(noiser_params, params, iterinfo, x_batch)
        raw_fitness = batch_fitness(logits, y_batch)
        shaped_fitness = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_fitness)
        noiser_params, params = jit_update(noiser_params, params, shaped_fitness, iterinfo)
        if gen % EVAL_EVERY == 0 or gen == NUM_GENERATIONS - 1:
            test_logits = jit_forward_eval(noiser_params, params, test_images[:2000])
            test_acc = float(jnp.mean(jnp.argmax(test_logits, axis=-1) == test_labels[:2000]))
            history["gen"].append(gen)
            history["test_acc"].append(test_acc)
            history["wallclock"].append(time.time() - t0)
    return history


rank_results = {}
for rank in RANK_VALUES:
    print(f"\n=== rank {rank} ===", flush=True)
    h = train_at_rank(rank=rank)
    rank_results[rank] = h
    print(f"  peak: {max(h['test_acc']):.4f}, "
          f"final: {h['test_acc'][-1]:.4f}, "
          f"time: {h['wallclock'][-1]:.1f}s", flush=True)

# Save results to disk so you can plot from a notebook later
results_path = os.path.expanduser("~/dissertation/eggroll-diss/results/rank_sweep.json")
os.makedirs(os.path.dirname(results_path), exist_ok=True)
with open(results_path, "w") as f:
    json.dump({str(k): v for k, v in rank_results.items()}, f, indent=2)
print(f"\nSaved results to {results_path}", flush=True)

print(f"\n{'Rank':<6} {'Peak acc':<10} {'Final acc':<10} {'Time (s)':<10}", flush=True)
print("-" * 40, flush=True)
for rank in RANK_VALUES:
    h = rank_results[rank]
    print(f"{rank:<6} {max(h['test_acc']):.4f}    {h['test_acc'][-1]:.4f}    {h['wallclock'][-1]:.1f}", flush=True)
