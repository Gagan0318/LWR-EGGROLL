"""Adaptive Sensitivity Pilot — derives per-layer rank allocation.

Runs a three-phase sensitivity pilot to determine optimal per-layer
rank assignments for LWR-EGGROLL:

  Phase 1: Elevation — measures per-layer perturbation magnitude
           (shared-checkpoint, one-gen variance measurement).
  Phase 2: Causal ablation — drops each layer to floor rank, measures
           degradation. Primary ordering mechanism.
  Phase 3: Binary inclusion — tests rank 0 vs floor rank for least
           sensitive layer. Skipped if 0 is not in the rank set.

Phase 1 magnitudes cross-reference with Phase 2 ordering to save rank
budget on middle layers with low perturbation magnitude.

# ── RANK SET CONFIGURATION ────────────────────────────────────────
#
# The rank set controls all rank assignments throughout the pilot.
# Default: [0, 1, 2, 4, 8]
#
# All assigned ranks are drawn from this set. Changing it here
# propagates to every phase automatically — nothing is hardcoded.
#
# Guidance:
#   - Deterministic environments (MNIST, Brax): [0, 1, 2, 4, 8]
#     Rank 0 (freeze) is safe when fitness signal is clean.
#
#   - Stochastic environments (CartPole, LunarLander): [1, 2, 4]
#     Rank 0 excluded — noisy fitness signals cannot distinguish
#     genuine insensitivity from noise-masked contributions.
#     Phase 3 is skipped automatically when 0 is absent.
#
#   - Capped set for high-parameter RL: [0, 1, 2, 4]
#     Rank 8 may dilute gradient estimates when hidden layers
#     dominate the parameter count.
#
# Powers of two are recommended — each unique rank triggers a
# separate JAX JIT compilation.
# ──────────────────────────────────────────────────────────────────

Usage:
    from lwr_eggroll.adaptive_sensitivity_pilot import AdaptiveSensitivityPilot

    pilot = AdaptiveSensitivityPilot(
        train_fn=my_train_function,
        layer_shapes={"input": (256, 784), "hidden": (256, 256), "output": (10, 256)},
        rank_set=[0, 1, 2, 4, 8],
        fitness_key="best_test_acc",
        best_fitness_key="best_individual_fitness",  # for Phase 3 gate
    )
    result = pilot.run()
"""
import time
from typing import Dict, Tuple, Callable, Optional, List

import numpy as np

from lwr_eggroll.strategy_selector import (
    select_strategy_from_dict,
    phase3_freeze_decision,
)


DEFAULT_RANK_SET = [0, 1, 2, 4, 8]


class AdaptiveSensitivityPilot:
    """Three-phase sensitivity pilot for LWR-EGGROLL.

    Args:
        train_fn: Callable(seed, rank_spec_dict, label, **kwargs) -> dict.
            The result dict must contain the key specified by fitness_key.
            Optionally also contains best_fitness_key for Phase 3 gate.
        layer_shapes: {layer_name: (out_dim, in_dim)} for every layer.
        rank_set: List of allowed rank values. Default [0, 1, 2, 4, 8].
        fitness_key: Key in train_fn result for the mean fitness metric.
        best_fitness_key: Key for best-individual fitness (Phase 3 gate).
            If None, Phase 3 gate uses fitness_key for both checks.
        n_seeds: Number of seeds for Phase 2 and Phase 3 runs.
        baseline_rank: Uniform rank for Phase 2 baseline. Must be in rank_set.
        max_rank: Maximum rank for Phase 1 elevation. Must be in rank_set.
        train_kwargs: Extra kwargs passed to train_fn on every call.
    """

    def __init__(
        self,
        train_fn: Callable,
        layer_shapes: Dict[str, Tuple[int, int]],
        rank_set: Optional[List[int]] = None,
        fitness_key: str = "best_test_acc",
        best_fitness_key: Optional[str] = None,
        n_seeds: int = 3,
        baseline_rank: Optional[int] = None,
        max_rank: Optional[int] = None,
        train_kwargs: Optional[dict] = None,
    ):
        self.train_fn = train_fn
        self.layer_shapes = layer_shapes
        self.names = list(layer_shapes.keys())
        self.shapes_list = list(layer_shapes.values())
        self.fitness_key = fitness_key
        self.best_fitness_key = best_fitness_key
        self.n_seeds = n_seeds
        self.train_kwargs = train_kwargs or {}

        # ── Rank set configuration ────────────────────────────
        self.rank_set = sorted(rank_set or DEFAULT_RANK_SET)
        self.has_zero = 0 in self.rank_set

        # Non-zero ranks (sorted descending for tier assignment)
        self.nonzero_ranks = sorted(
            [r for r in self.rank_set if r > 0], reverse=True
        )
        if len(self.nonzero_ranks) == 0:
            raise ValueError("Rank set must contain at least one non-zero value.")

        # Derive key ranks from the set
        self.floor_rank = self.nonzero_ranks[-1]  # smallest non-zero
        self.max_rank = max_rank or self.nonzero_ranks[0]
        self.baseline_rank = baseline_rank or (
            self.nonzero_ranks[1] if len(self.nonzero_ranks) > 1
            else self.nonzero_ranks[0]
        )

        # Tier assignment (derived from rank set, never hardcoded):
        #   Tier 0 (most sensitive):  max non-zero rank
        #   Tier 1 (moderate):        one step down from max
        #   Tier 2 (low P1 magnitude): another step down
        #   Floor (least sensitive):  smallest non-zero rank → Phase 3 candidate
        self.tier_ranks = []
        for r in self.nonzero_ranks:
            if r not in self.tier_ranks:
                self.tier_ranks.append(r)
        # tier_ranks[0] = max, tier_ranks[1] = moderate, tier_ranks[2] = low-P1

        if self.max_rank not in self.rank_set:
            raise ValueError(
                f"max_rank={self.max_rank} not in rank_set={self.rank_set}"
            )
        if self.baseline_rank not in self.rank_set:
            raise ValueError(
                f"baseline_rank={self.baseline_rank} not in rank_set={self.rank_set}"
            )

    def _run_train(self, seed: int, rank_spec: dict, label: str) -> dict:
        """Run train_fn with the given spec and return result dict."""
        return self.train_fn(seed, rank_spec, label, **self.train_kwargs)

    def _get_fitness(self, result: dict) -> float:
        """Extract the mean fitness metric from a result dict."""
        return float(result[self.fitness_key])

    def _get_best_fitness(self, result: dict) -> Optional[float]:
        """Extract the best-individual fitness from a result dict."""
        if self.best_fitness_key is None:
            return None
        val = result.get(self.best_fitness_key)
        if val is None:
            return None
        # Handle history format [(gen, value), ...] or scalar
        if isinstance(val, list) and len(val) > 0:
            if isinstance(val[0], (list, tuple)):
                return float(max(v for _, v in val))
            return float(max(val))
        return float(val)

    def _make_spec(self, target_shape: tuple, target_rank: int,
                   background_rank: int) -> dict:
        """Build a rank spec dict with target layer at target_rank,
        all others at background_rank."""
        spec = {}
        for shape in self.shapes_list:
            spec[shape] = (
                min(target_rank, shape[0], shape[1])
                if shape == target_shape
                else background_rank
            )
        return spec

    def _make_uniform_spec(self, rank: int) -> dict:
        """Build a uniform rank spec."""
        return {s: rank for s in self.shapes_list}

    def run(self) -> dict:
        """Execute the full three-phase pilot.

        Returns:
            dict with keys: allocation, allocation_named, ordering,
            strategy_recommendation, phase1_magnitudes, phase2_degradations,
            phase3_decision.
        """
        t0 = time.time()
        seeds = list(range(self.n_seeds))

        print("=" * 60)
        print("ADAPTIVE SENSITIVITY PILOT")
        print(f"  Layers: {self.layer_shapes}")
        print(f"  Rank set: {self.rank_set}")
        print(f"  Baseline: r={self.baseline_rank}, Max: r={self.max_rank}")
        print(f"  Floor: r={self.floor_rank}, Has zero: {self.has_zero}")
        print(f"  Tier ranks: {self.tier_ranks}")
        print(f"  Fitness key: {self.fitness_key}")
        print(f"  Seeds: {self.n_seeds}")
        print("=" * 60, flush=True)

        # ── Phase 1: Elevation ────────────────────────────────
        print("\n--- Phase 1: Elevation (perturbation magnitude) ---")
        print(f"  Background: r={self.floor_rank}, "
              f"Elevation: r={self.max_rank}", flush=True)

        phase1_magnitudes = {}
        for name, shape in zip(self.names, self.shapes_list):
            spec = self._make_spec(shape, self.max_rank, self.floor_rank)

            elevation_values = []
            print(f"\n  Elevating {name} {shape} to r={spec[shape]}:")
            for seed in seeds:
                print(f"    seed={seed}...", end="", flush=True)
                result = self._run_train(seed, spec, f"phase1_elevate_{name}")
                val = self._get_fitness(result)
                elevation_values.append(val)
                print(f" {self.fitness_key}={val:.4f}")

            phase1_magnitudes[name] = float(np.var(elevation_values))
            print(f"  {name}: variance = {phase1_magnitudes[name]:.6f}")

        p1_ordering = sorted(phase1_magnitudes.keys(),
                             key=lambda n: phase1_magnitudes[n], reverse=True)
        print(f"\n  Phase 1 ordering: {' > '.join(p1_ordering)}")

        # ── Phase 2: Causal Ablation ──────────────────────────
        print(f"\n--- Phase 2: Causal Ablation ---")
        print(f"  Baseline: all at r={self.baseline_rank}")
        print(f"  Ablation: drop each to r={self.floor_rank}", flush=True)

        # Baseline
        baseline_spec = self._make_uniform_spec(self.baseline_rank)
        baseline_values = []
        for seed in seeds:
            print(f"  Baseline seed={seed}...", end="", flush=True)
            result = self._run_train(
                seed, baseline_spec, f"baseline_r{self.baseline_rank}")
            val = self._get_fitness(result)
            baseline_values.append(val)
            print(f" {self.fitness_key}={val:.4f}")
        baseline_avg = float(np.mean(baseline_values))
        print(f"  Baseline average: {baseline_avg:.4f}")

        # Ablate each layer
        degradations = {}
        for name, shape in zip(self.names, self.shapes_list):
            ablated_spec = self._make_uniform_spec(self.baseline_rank)
            ablated_spec[shape] = self.floor_rank

            ablated_values = []
            print(f"\n  Ablating {name} {shape} to r={self.floor_rank}:")
            for seed in seeds:
                print(f"    seed={seed}...", end="", flush=True)
                result = self._run_train(
                    seed, ablated_spec, f"ablate_{name}")
                val = self._get_fitness(result)
                ablated_values.append(val)
                print(f" {self.fitness_key}={val:.4f}")

            ablated_avg = float(np.mean(ablated_values))
            degradation = baseline_avg - ablated_avg
            degradations[name] = degradation
            print(f"  {name}: degradation = {degradation:+.4f}")

        ordering = sorted(degradations.keys(),
                          key=lambda n: degradations[n], reverse=True)
        print(f"\n  Sensitivity ordering: {' > '.join(ordering)}")

        # ── Phase 1 × Phase 2 cross-referencing ──────────────
        print("\n--- Phase 1 × Phase 2 Cross-Referencing ---")

        if len(self.tier_ranks) >= 3:
            moderate_rank = self.tier_ranks[1]
            low_p1_rank = self.tier_ranks[2]
        elif len(self.tier_ranks) == 2:
            moderate_rank = self.tier_ranks[1]
            low_p1_rank = self.tier_ranks[1]
        else:
            moderate_rank = self.tier_ranks[0]
            low_p1_rank = self.tier_ranks[0]

        p1_vals = list(phase1_magnitudes.values())
        p1_median = float(np.median(p1_vals))

        print(f"  Phase 1 median magnitude: {p1_median:.6f}")
        print(f"  Moderate rank: {moderate_rank}, "
              f"Low-P1 rank: {low_p1_rank}")

        for name in ordering[1:-1]:  # middle layers only
            mag = phase1_magnitudes[name]
            assigned = low_p1_rank if mag < p1_median else moderate_rank
            print(f"  {name}: P1 mag={mag:.6f} "
                  f"({'below' if mag < p1_median else 'above'} median) "
                  f"→ r={assigned}")

        # ── Phase 3: Binary Inclusion ─────────────────────────
        least_name = ordering[-1]
        least_shape = self.layer_shapes[least_name]

        if not self.has_zero:
            print(f"\n--- Phase 3: SKIPPED ---")
            print(f"  Rank set {self.rank_set} does not contain 0.")
            print(f"  {least_name} assigned floor rank {self.floor_rank}.")
            phase3_decision = self.floor_rank
        else:
            print(f"\n--- Phase 3: Binary Inclusion ({least_name}) ---")
            print(f"  Testing rank 0 vs rank {self.floor_rank} "
                  f"for {least_name}", flush=True)

            # Build specs: everything at derived allocation, vary least
            # Use baseline rank for other layers during Phase 3
            spec_r0 = self._make_uniform_spec(self.baseline_rank)
            spec_r0[least_shape] = 0
            spec_r1 = self._make_uniform_spec(self.baseline_rank)
            spec_r1[least_shape] = self.floor_rank

            # Rank 0 condition
            print(f"\n  Condition A: {least_name} at rank 0 (frozen)")
            r0_mean_vals = []
            r0_best_vals = []
            for seed in seeds:
                print(f"    seed={seed}...", end="", flush=True)
                result = self._run_train(
                    seed, spec_r0, f"phase3_{least_name}_r0")
                mean_val = self._get_fitness(result)
                best_val = self._get_best_fitness(result)
                r0_mean_vals.append(mean_val)
                if best_val is not None:
                    r0_best_vals.append(best_val)
                print(f" {self.fitness_key}={mean_val:.4f}"
                      + (f" best={best_val:.4f}" if best_val else ""))

            # Rank 1 condition
            print(f"\n  Condition B: {least_name} at rank {self.floor_rank}")
            r1_mean_vals = []
            r1_best_vals = []
            for seed in seeds:
                print(f"    seed={seed}...", end="", flush=True)
                result = self._run_train(
                    seed, spec_r1, f"phase3_{least_name}_r{self.floor_rank}")
                mean_val = self._get_fitness(result)
                best_val = self._get_best_fitness(result)
                r1_mean_vals.append(mean_val)
                if best_val is not None:
                    r1_best_vals.append(best_val)
                print(f" {self.fitness_key}={mean_val:.4f}"
                      + (f" best={best_val:.4f}" if best_val else ""))

            # Decide using shared gate
            phase3_decision = phase3_freeze_decision(
                r0_mean_fitnesses=r0_mean_vals,
                r1_mean_fitnesses=r1_mean_vals,
                candidate_shape=least_shape,
                all_layer_shapes=self.layer_shapes,
                r0_best_fitnesses=r0_best_vals or None,
                r1_best_fitnesses=r1_best_vals or None,
            )

            r0_avg = float(np.mean(r0_mean_vals))
            r1_avg = float(np.mean(r1_mean_vals))
            print(f"\n  Rank 0 mean: {r0_avg:.4f}")
            print(f"  Rank {self.floor_rank} mean: {r1_avg:.4f}")
            print(f"  Phase 3 decision: {least_name} → rank {phase3_decision}")

        # ── Strategy selection ────────────────────────────────
        # Mark least-sensitive layer if Phase 3 confirmed freeze
        modified_degradations = dict(degradations)
        if phase3_decision == 0:
            modified_degradations[least_name] = -999.0

        rec = select_strategy_from_dict(
            degradation_scores=modified_degradations,
            layer_shapes=self.layer_shapes,
            baseline_rank=self.baseline_rank,
            max_rank=self.max_rank,
            verbose=True,
            phase1_magnitudes=phase1_magnitudes,
            rank_set=self.rank_set,
        )

        elapsed = time.time() - t0
        print(f"\nPilot completed in {elapsed/60:.1f} minutes.")

        return {
            "allocation": rec.rank_allocation_shapes,
            "allocation_named": rec.rank_allocation,
            "ordering": ordering,
            "strategy_recommendation": rec,
            "phase1_magnitudes": phase1_magnitudes,
            "phase1_ordering": p1_ordering,
            "phase2_degradations": degradations,
            "phase3_decision": phase3_decision,
            "phase3_layer": least_name,
            "elapsed_seconds": elapsed,
        }
