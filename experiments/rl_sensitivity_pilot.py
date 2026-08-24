"""RL Sensitivity Pilot — derives LWR allocation for RL architectures.

Runs all three phases using the standalone ES loop:
  Phase 1: Elevation — measures per-layer perturbation magnitude
           (shared-checkpoint, one-gen variance measurement).
  Phase 2: Causal ablation — drops each layer to r=1, measures degradation.
           Primary ordering mechanism.
  Phase 3: Binary inclusion — tests rank 0 vs rank 1 for least sensitive.

Phase 1 magnitudes cross-reference with Phase 2 ordering to assign rank 2
to middle-ranked layers with low perturbation magnitude (budget-saving).

For stochastic RL environments, rank set is {1, 2, 4} (rank 0 excluded
a priori due to noisy fitness signal — Phase 3 still runs but floor is r=1).

Usage:
    from rl_sensitivity_pilot import run_rl_pilot
    allocation = run_rl_pilot(
        init_fn=init_params,
        forward_fn=forward,
        np_forward_fn=np_forward,
        layer_shapes={"input": (64, 4), "hidden": (64, 64), "output": (2, 64)},
        env_name="CartPole-v1",
        ...
    )
    # allocation = {(64, 4): 4, (64, 64): 2, (2, 64): 1}
"""
import time
from multiprocessing import Pool
from typing import Dict, Tuple, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np

from strategy_selector import select_strategy_from_dict


FULL_RANK_SET = [0, 1, 2, 4, 8]


def run_rl_pilot(
    init_fn: Callable,
    perturb_low_rank_fn: Callable,
    perturb_full_rank_fn: Callable,
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
    baseline_rank: int = 4,
    max_rank: int = 8,
) -> Dict[Tuple[int, int], int]:
    """Run sensitivity pilot for an RL architecture.

    Phase 1: Elevation — per-layer fitness variance from shared checkpoint.
    Phase 2: Causal ablation — drop each layer from baseline_rank to 1.
    Phase 3: Binary inclusion — test rank 0 vs 1 for least sensitive.

    Phase 1 magnitudes are cross-referenced with Phase 2 ordering:
    middle-ranked layers with low Phase 1 magnitude get rank 2 instead
    of rank 4 (budget-saving). This makes the full rank set {1, 2, 4}
    actively used for stochastic RL environments.

    Returns: (shape_allocation, named_allocation, ordering, recommendation)
    """
    names = list(layer_shapes.keys())
    shapes = list(layer_shapes.values())
    seeds = list(range(n_seeds))

    print("=" * 60)
    print("RL SENSITIVITY PILOT")
    print(f"  Env: {env_name}")
    print(f"  Layers: {layer_shapes}")
    print(f"  Baseline rank: {baseline_rank}, Max rank: {max_rank}")
    print(f"  Pilot generations: {max_gens}, Seeds: {n_seeds}")
    print("=" * 60, flush=True)

    def run_es_short(seed, rank_spec, label):
        """Run a short ES training with Adam, return mean fitness at end."""
        key = jax.random.PRNGKey(seed)
        key, init_key = jax.random.split(key)
        params = init_fn(init_key)

        # Adam state
        adam_m = {n: jnp.zeros_like(params[n]) for n in params}
        adam_v = {n: jnp.zeros_like(params[n]) for n in params}
        b1, b2, eps_adam = 0.9, 0.999, 1e-8

        pool = Pool(n_workers)
        history = []

        try:
            for gen in range(max_gens):
                pop_params = []
                pop_noise = []
                for _ in range(pop_size):
                    key, pk = jax.random.split(key)
                    p, n = perturb_low_rank_fn(params, pk, sigma, rank_spec)
                    pop_params.append(p)
                    pop_noise.append(n)

                work = [(to_numpy_fn(p), env_name, n_eval_episodes) for p in pop_params]
                fitnesses = np.array(pool.map(eval_worker_fn, work))

                history.append(float(np.mean(fitnesses)))

                # ES update with Adam
                f_norm = (fitnesses - np.mean(fitnesses))
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

        final_mean = float(np.mean(history[-10:])) if len(history) >= 10 else float(np.mean(history))
        return final_mean

    # ── Phase 1: Shared-checkpoint Elevation ─────────────────
    # Run a short baseline, then measure fitness variance per layer
    # with that layer elevated to max_rank and others at r=1.
    # This gives us the magnitude signal for Phase 1 × Phase 2
    # cross-referencing (low magnitude middle layers get rank 2
    # instead of rank 4, saving rank budget).
    print("\n--- Phase 1: Elevation (fitness variance) ---")
    print(f"  Elevation rank: {max_rank}, Background: r=1", flush=True)

    phase1_magnitudes = {}
    for name, shape in zip(names, shapes):
        elevation_spec = {s: 1 for s in shapes}
        elevation_spec[shape] = min(max_rank, min(shape))  # cap at layer dims

        elevation_means = []
        print(f"\n  Elevating {name} {shape} to r={elevation_spec[shape]}:")
        for seed in seeds:
            print(f"    seed={seed}...", end="", flush=True)
            m = run_es_short(seed, elevation_spec, f"phase1_elevate_{name}")
            elevation_means.append(m)
            print(f" mean_fitness={m:.1f}")

        phase1_magnitudes[name] = float(np.var(elevation_means))
        print(f"  {name}: variance = {phase1_magnitudes[name]:.4f}")

    p1_ordering = sorted(phase1_magnitudes.keys(),
                         key=lambda n: phase1_magnitudes[n], reverse=True)
    print(f"\n  Phase 1 ordering: {' > '.join(p1_ordering)}")

    # ── Phase 2: Causal Ablation ──────────────────────────────
    print("\n--- Phase 2: Causal Ablation ---")
    print(f"  Baseline: all layers at r={baseline_rank}")
    print(f"  Ablation: drop each layer to r=1", flush=True)

    # Baseline
    uniform_spec = {s: baseline_rank for s in shapes}
    baseline_means = []
    for seed in seeds:
        print(f"  Baseline seed={seed}...", end="", flush=True)
        m = run_es_short(seed, uniform_spec, f"baseline_r{baseline_rank}")
        baseline_means.append(m)
        print(f" mean_fitness={m:.1f}")
    baseline_avg = float(np.mean(baseline_means))
    print(f"  Baseline average: {baseline_avg:.1f}")

    # Ablate each layer
    degradations = {}
    for name, shape in zip(names, shapes):
        ablated_spec = {s: baseline_rank for s in shapes}
        ablated_spec[shape] = 1

        ablated_means = []
        print(f"\n  Ablating {name} {shape} to r=1:")
        for seed in seeds:
            print(f"    seed={seed}...", end="", flush=True)
            m = run_es_short(seed, ablated_spec, f"ablate_{name}")
            ablated_means.append(m)
            print(f" mean_fitness={m:.1f}")

        ablated_avg = float(np.mean(ablated_means))
        degradation = baseline_avg - ablated_avg
        degradations[name] = degradation
        print(f"  {name}: degradation = {degradation:+.1f}")

    # Ordering from Phase 2
    ordering = sorted(degradations.keys(), key=lambda n: degradations[n], reverse=True)
    print(f"\n  Sensitivity ordering: {' > '.join(ordering)}")

    # ── Phase 3: Binary Inclusion ─────────────────────────────
    least_sensitive_name = ordering[-1]
    least_sensitive_shape = layer_shapes[least_sensitive_name]

    print(f"\n--- Phase 3: Binary Inclusion ({least_sensitive_name}) ---")
    print(f"  Testing rank 0 vs rank 1 for {least_sensitive_name}", flush=True)

    # Rank 1
    spec_r1 = {s: baseline_rank for s in shapes}
    spec_r1[least_sensitive_shape] = 1
    r1_means = []
    for seed in seeds:
        print(f"  Rank 1 seed={seed}...", end="", flush=True)
        m = run_es_short(seed, spec_r1, f"p3_{least_sensitive_name}_r1")
        r1_means.append(m)
        print(f" mean_fitness={m:.1f}")
    r1_avg = float(np.mean(r1_means))

    # Rank 0
    spec_r0 = {s: baseline_rank for s in shapes}
    spec_r0[least_sensitive_shape] = 0
    r0_means = []
    for seed in seeds:
        print(f"  Rank 0 seed={seed}...", end="", flush=True)
        m = run_es_short(seed, spec_r0, f"p3_{least_sensitive_name}_r0")
        r0_means.append(m)
        print(f" mean_fitness={m:.1f}")
    r0_avg = float(np.mean(r0_means))

    least_rank = 0 if r0_avg >= r1_avg else 1
    print(f"\n  Rank 0 avg: {r0_avg:.1f}, Rank 1 avg: {r1_avg:.1f}")
    print(f"  Decision: {least_sensitive_name} → rank {least_rank}")

    # ── Strategy selection ────────────────────────────────────
    # Override least-sensitive layer's degradation if Phase 3
    # showed rank 0 is viable (negative or zero cost).
    # This lets the selector see negative degradation → assign 0.
    if least_rank == 0 and degradations[least_sensitive_name] >= 0:
        # Phase 3 confirmed rank 0 is fine — make degradation negative
        # so the selector assigns rank 0 automatically.
        degradations[least_sensitive_name] = -999.0  # sentinel for strategy_selector Phase 3 confirmation

    rec = select_strategy_from_dict(
        degradation_scores=degradations,
        layer_shapes=layer_shapes,
        baseline_rank=baseline_rank,
        max_rank=max_rank,
        verbose=True,
        phase1_magnitudes=phase1_magnitudes,
    )

    # Build shape-keyed allocation for HyperscaleES
    allocation = {}
    for name in names:
        shape = layer_shapes[name]
        allocation[shape] = rec.rank_allocation[name]

    allocation_named = rec.rank_allocation
    total_budget = rec.total_budget

    print(f"\n{'=' * 60}")
    print("PILOT RESULT")
    print(f"{'=' * 60}")
    print(f"  Strategy:   {rec.strategy.upper()}")
    print(f"  Confidence: {rec.confidence}")
    print(f"  Ordering:   {' > '.join(ordering)}")
    print(f"  Allocation: {allocation_named}")
    print(f"  Budget:     {total_budget}")
    print(f"  Finding:    {rec.finding}")
    print(f"{'=' * 60}\n", flush=True)

    return allocation, allocation_named, ordering, rec
