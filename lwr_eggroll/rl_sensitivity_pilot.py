"""RL Sensitivity Pilot — thin wrapper for CPU-based RL environments.

Builds a CPU multiprocessing train_fn and passes it to
AdaptiveSensitivityPilot. All pilot logic (phases, ordering,
cross-referencing, Phase 3 gate) lives in the shared pilot.

For GPU-based RL (Brax), experiments call AdaptiveSensitivityPilot
directly with their own GPU train_fn.

Usage:
    from lwr_eggroll.rl_sensitivity_pilot import run_rl_pilot

    result = run_rl_pilot(
        init_fn=init_params,
        perturb_low_rank_fn=perturb,
        to_numpy_fn=to_np,
        eval_worker_fn=eval_worker,
        layer_shapes={"input": (64, 4), "hidden": (64, 64), "output": (2, 64)},
        env_name="CartPole-v1",
        rank_set=[1, 2, 4],   # stochastic → no rank 0
    )
"""
from multiprocessing import Pool
from typing import Dict, Tuple, Callable, Optional, List

import jax
import jax.numpy as jnp
import numpy as np

from lwr_eggroll.adaptive_sensitivity_pilot import (
    AdaptiveSensitivityPilot,
    DEFAULT_RANK_SET,
)


def run_rl_pilot(
    init_fn: Callable,
    perturb_low_rank_fn: Callable,
    to_numpy_fn: Callable,
    eval_worker_fn: Callable,
    layer_shapes: Dict[str, Tuple[int, int]],
    env_name: str,
    pop_size: int = 64,
    sigma: float = 0.05,
    lr: float = 0.01,
    max_gens: int = 100,
    n_eval_episodes: int = 3,
    n_seeds: int = 3,
    n_workers: int = 4,
    rank_set: Optional[List[int]] = None,
    baseline_rank: Optional[int] = None,
    max_rank: Optional[int] = None,
) -> dict:
    """Run sensitivity pilot for a CPU-based RL environment.

    Wraps the multiprocessing ES loop into a train_fn and delegates
    all pilot logic to AdaptiveSensitivityPilot.
    """
    if rank_set is None:
        rank_set = [1, 2, 4]  # default for stochastic RL

    names = list(layer_shapes.keys())
    shapes = list(layer_shapes.values())

    def train_fn(seed: int, rank_spec: dict, label: str, **kwargs) -> dict:
        """CPU multiprocessing ES training loop."""
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        params = init_fn(init_key)

        # Adam state
        adam_m = {n: jnp.zeros_like(params[n]) for n in params}
        adam_v = {n: jnp.zeros_like(params[n]) for n in params}
        b1, b2, eps_adam = 0.9, 0.999, 1e-8

        pool = Pool(n_workers)
        mean_history = []
        best_history = []

        try:
            for gen in range(max_gens):
                pop_params = []
                pop_noise = []
                for _ in range(pop_size):
                    key, pk = jax.random.split(key)
                    p, n = perturb_low_rank_fn(params, pk, sigma, rank_spec)
                    pop_params.append(p)
                    pop_noise.append(n)

                work = [(to_numpy_fn(p), env_name, n_eval_episodes)
                        for p in pop_params]
                fitnesses = np.array(pool.map(eval_worker_fn, work))

                mean_history.append(float(np.mean(fitnesses)))
                best_history.append(float(np.max(fitnesses)))

                f_norm = fitnesses - np.mean(fitnesses)
                f_std = np.std(fitnesses)
                if f_std > 1e-8:
                    f_norm = f_norm / f_std

                for name in params:
                    grad = jnp.zeros_like(params[name])
                    for i in range(pop_size):
                        grad = grad + f_norm[i] * pop_noise[i][name]
                    grad = grad / (pop_size * sigma)
                    adam_m[name] = b1 * adam_m[name] + (1 - b1) * grad
                    adam_v[name] = b2 * adam_v[name] + (1 - b2) * grad ** 2
                    mh = adam_m[name] / (1 - b1 ** (gen + 1))
                    vh = adam_v[name] / (1 - b2 ** (gen + 1))
                    params[name] = params[name] + lr * mh / (jnp.sqrt(vh) + eps_adam)
        finally:
            pool.close()
            pool.join()

        tail = mean_history[-10:] if len(mean_history) >= 10 else mean_history
        return {
            "final_mean_fitness": float(np.mean(tail)),
            "best_fitness": float(np.max(best_history)) if best_history else 0.0,
            "mean_history": mean_history,
            "best_history": best_history,
        }

    pilot = AdaptiveSensitivityPilot(
        train_fn=train_fn,
        layer_shapes=layer_shapes,
        rank_set=rank_set,
        fitness_key="final_mean_fitness",
        best_fitness_key="best_fitness",
        n_seeds=n_seeds,
        baseline_rank=baseline_rank,
        max_rank=max_rank,
    )

    return pilot.run()
