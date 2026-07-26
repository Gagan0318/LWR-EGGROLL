"""
Layer-Wise Rank EGGROLL (LWR-EGGROLL)
=====================================

A variant of EGGROLL that allows different perturbation ranks for different
parameter matrices, rather than a single uniform rank across all layers.

Motivation
----------
The sigma-x-rank interaction observed empirically on MNIST (2026-07-22) suggests
that the "correct" rank depends on the fitness landscape at that layer's scale.
Vanilla EGGROLL applies uniform rank across every Dense layer; LWR-EGGROLL
generalises this by allowing per-layer rank allocation.

Interface
---------
`rank` in `init_noiser` can be:
  - int (scalar): uniform rank across all layers. Numerically equivalent
    to `EggRoll`. Useful for sanity-checking the plumbing.
  - dict {shape_tuple: rank}: shape-keyed rank lookup. E.g.
      {(256, 784): 8, (128, 256): 4, (10, 128): 2}
    Requires all MM param shapes present as keys; raises otherwise.
  - callable(shape) -> rank: rank policy function.

Limitations
-----------
Shape-keyed lookup collapses layers that share a shape (e.g. transformer
blocks). For the MNIST MLP and small Brax MLPs used in the dissertation
pilots, all MM shapes are unique, so this is fine. Extending to transformer
architectures would require threading a layer identifier through
`call_submodule` -- deferred as future work.

Author: Gagan Deep Singh (MSc dissertation, University of Birmingham, 2026)
"""
from collections import defaultdict
from functools import partial

import jax
import jax.numpy as jnp
import optax
from jax.tree_util import tree_flatten, tree_unflatten

from .base_noiser import Noiser
from .eggroll import (
    _noop_update,
    _simple_full_update,
    get_nonlora_update_params,
)


# ---------------------------------------------------------------------------
# Rank resolution
# ---------------------------------------------------------------------------

def _get_rank(rank_spec, param_shape):
    """Resolve rank for a single parameter given the spec."""
    if isinstance(rank_spec, int):
        return rank_spec
    if isinstance(rank_spec, dict):
        if param_shape not in rank_spec:
            raise KeyError(
                f"LWR-EGGROLL: no rank specified for param shape {param_shape}. "
                f"Configured shapes: {list(rank_spec.keys())}"
            )
        return rank_spec[param_shape]
    if callable(rank_spec):
        return rank_spec(param_shape)
    raise TypeError(f"Unrecognised rank spec type: {type(rank_spec).__name__}")


# ---------------------------------------------------------------------------
# LoRA update generation (LWR-aware)
# ---------------------------------------------------------------------------

def get_lora_update_params_lwr(frozen_noiser_params, base_sigma, iterinfo, param, key):
    epoch, thread_id = iterinfo

    true_epoch = (
        0 if frozen_noiser_params["noise_reuse"] == 0
        else epoch // frozen_noiser_params["noise_reuse"]
    )
    true_thread_idx = thread_id // 2
    sigma = jnp.where(thread_id % 2 == 0, base_sigma, -base_sigma)

    a, b = param.shape
    rank = _get_rank(frozen_noiser_params["rank"], param.shape)

    lora_params = jax.random.normal(
        jax.random.fold_in(jax.random.fold_in(key, true_epoch), true_thread_idx),
        (a + b, rank),
        dtype=param.dtype,
    )
    B = lora_params[:b]
    A = lora_params[b:]
    return A * sigma, B


def _simple_lora_update_lwr(base_sigma, param, key, scores, iterinfo, frozen_noiser_params):
    rank = _get_rank(frozen_noiser_params["rank"], param.shape)
    scaled_sigma = base_sigma / jnp.sqrt(rank)

    A, B = jax.vmap(
        partial(get_lora_update_params_lwr, frozen_noiser_params),
        in_axes=(None, 0, None, None),
    )(scaled_sigma, iterinfo, param, key)

    broadcasted_scores = jnp.reshape(scores, scores.shape + (1, 1))
    A = broadcasted_scores * A
    num_envs = scores.shape[0]
    return jnp.einsum("nir,njr->ij", A, B) / num_envs


# ---------------------------------------------------------------------------
# Noiser class
# ---------------------------------------------------------------------------

class LWREggRoll(Noiser):
    """Layer-Wise Rank EGGROLL."""

    @classmethod
    def init_noiser(
        cls, params, sigma, lr, *args,
        solver=None, solver_kwargs=None,
        group_size=0, freeze_nonlora=False, noise_reuse=0,
        rank=1, use_batched_update: bool = False,
        **kwargs,
    ):
        if solver is None:
            solver = optax.sgd
        if solver_kwargs is None:
            solver_kwargs = {}
        true_solver = solver(lr, **solver_kwargs)
        opt_state = true_solver.init(params)

        if isinstance(rank, dict):
            flat_params, _ = tree_flatten(params)
            mm_shapes = {
                tuple(p.shape) for p in flat_params
                if hasattr(p, "shape") and len(p.shape) == 2
            }
            missing = mm_shapes - set(rank.keys())
            if missing:
                raise ValueError(
                    f"LWR-EGGROLL: rank dict missing entries for MM param shapes: "
                    f"{missing}. Provided: {list(rank.keys())}"
                )

        frozen = {
            "group_size": group_size,
            "freeze_nonlora": freeze_nonlora,
            "noise_reuse": noise_reuse,
            "solver": true_solver,
            "rank": rank,
            "use_batched_update": use_batched_update,
        }
        mutable = {"sigma": sigma, "opt_state": opt_state}
        return frozen, mutable

    @classmethod
    def do_mm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = x @ param.T
        if iterinfo is None:
            return base_ans
        rank = _get_rank(frozen_noiser_params["rank"], param.shape)
        A, B = get_lora_update_params_lwr(
            frozen_noiser_params,
            noiser_params["sigma"] / jnp.sqrt(rank),
            iterinfo, param, base_key,
        )
        return base_ans + x @ B @ A.T

    @classmethod
    def do_Tmm(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        base_ans = x @ param
        if iterinfo is None:
            return base_ans
        rank = _get_rank(frozen_noiser_params["rank"], param.shape)
        A, B = get_lora_update_params_lwr(
            frozen_noiser_params,
            noiser_params["sigma"] / jnp.sqrt(rank),
            iterinfo, param, base_key,
        )
        return base_ans + x @ A @ B.T

    @classmethod
    def do_emb(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo, x):
        raise NotImplementedError("Embedding is not implemented for LWR-EGGROLL")

    @classmethod
    def get_noisy_standard(cls, frozen_noiser_params, noiser_params, param, base_key, iterinfo):
        if iterinfo is None or frozen_noiser_params["freeze_nonlora"]:
            return param
        return param + get_nonlora_update_params(
            frozen_noiser_params, noiser_params["sigma"], iterinfo, param, base_key,
        )

    @classmethod
    def convert_fitnesses(cls, frozen_noiser_params, noiser_params, raw_scores, num_episodes_list=None):
        group_size = frozen_noiser_params["group_size"]
        if group_size == 0:
            true_scores = (raw_scores - jnp.mean(raw_scores, keepdims=True)) / jnp.sqrt(
                jnp.var(raw_scores, keepdims=True) + 1e-5
            )
        else:
            group_scores = raw_scores.reshape((-1, group_size))
            true_scores = (
                group_scores - jnp.mean(group_scores, axis=-1, keepdims=True)
            ) / jnp.sqrt(jnp.var(raw_scores, keepdims=True) + 1e-5)
            true_scores = true_scores.ravel()
        return true_scores

    @classmethod
    def _do_update(cls, param, base_key, fitnesses, iterinfos, map_classification,
                   sigma, frozen_noiser_params, **kwargs):
        update_fn = [
            _simple_full_update,
            _simple_lora_update_lwr,
            _noop_update,
            _noop_update,
        ][map_classification]

        if len(base_key.shape) == 0:
            new_grad = update_fn(sigma, param, base_key, fitnesses, iterinfos, frozen_noiser_params)
        else:
            new_grad = jax.lax.scan(
                lambda _, x: (0, update_fn(sigma, x[0], x[1], fitnesses, iterinfos, frozen_noiser_params)),
                0, xs=(param, base_key),
            )[1]

        return -(new_grad * jnp.sqrt(fitnesses.size)).astype(param.dtype)

    @classmethod
    def do_updates(cls, frozen_noiser_params, noiser_params, params, base_keys,
                   fitnesses, iterinfos, es_map):
        if frozen_noiser_params["use_batched_update"]:
            return cls._do_updates_batched(
                frozen_noiser_params, noiser_params, params, base_keys,
                fitnesses, iterinfos, es_map,
            )
        return cls._do_updates_original(
            frozen_noiser_params, noiser_params, params, base_keys,
            fitnesses, iterinfos, es_map,
        )

    @classmethod
    def _do_updates_original(cls, frozen_noiser_params, noiser_params, params,
                             base_keys, fitnesses, iterinfos, es_map):
        new_grad = jax.tree.map(
            lambda p, k, m: cls._do_update(
                p, k, fitnesses, iterinfos, m, noiser_params["sigma"], frozen_noiser_params,
            ),
            params, base_keys, es_map,
        )
        updates, noiser_params["opt_state"] = frozen_noiser_params["solver"].update(
            new_grad, noiser_params["opt_state"], params,
        )
        return noiser_params, optax.apply_updates(params, updates)

    @classmethod
    def _do_updates_batched(cls, frozen_noiser_params, noiser_params, params,
                            base_keys, fitnesses, iterinfos, es_map):
        flat_params, treedef = tree_flatten(params)
        flat_keys, _ = tree_flatten(base_keys)
        flat_es, _ = tree_flatten(es_map)

        buckets = defaultdict(list)
        for i, (param, map_class) in enumerate(zip(flat_params, flat_es)):
            key = param.shape, map_class
            buckets[key].append(i)

        new_flat_grads = [None] * len(flat_params)

        for (_, map_class), indices in buckets.items():
            batched_params = jnp.stack([flat_params[i] for i in indices])
            batched_keys = jnp.stack([flat_keys[i] for i in indices])

            grads_for_this_batch = jax.vmap(
                lambda param, rng: cls._do_update(
                    param, rng, fitnesses, iterinfos, map_class,
                    noiser_params["sigma"], frozen_noiser_params,
                ),
            )(batched_params, batched_keys)

            for i, idx in enumerate(indices):
                new_flat_grads[idx] = grads_for_this_batch[i]

        new_grad = tree_unflatten(treedef, new_flat_grads)
        updates, noiser_params["opt_state"] = frozen_noiser_params["solver"].update(
            new_grad, noiser_params["opt_state"], params,
        )
        return noiser_params, optax.apply_updates(params, updates)
