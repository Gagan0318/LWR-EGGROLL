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

# ── ADAPTIVE METRIC SELECTION ────────────────────────────────────
#
# After Phase 1 elevation runs, the pilot checks whether the fitness
# landscape exhibits a "flat initial basin" — a condition where mean
# fitness is uninformative but best (max) fitness carries early signal.
#
# Detection: if best_fitness separates from mean_fitness by more than
# BASIN_DETECTION_K standard deviations of the fitness distribution,
# the pilot switches all subsequent decision-making (Phase 2 comparisons
# and Phase 3 gate) to use best_fitness instead of mean_fitness.
#
# Environments with flat basins (e.g. Brax Ant, where the agent must
# learn to stand before walking) benefit from this because the pilot
# window (10-15 gens) is shorter than the basin escape time. Best
# fitness detects the first perturbation direction that escapes the
# basin, while mean fitness remains stuck in noise.
#
# Environments with informative initial landscapes (e.g. MNIST, where
# random policies already classify some digits correctly) will not
# trigger the switch — best and mean track closely from the start.
#
# This is logged clearly so the decision is traceable in experiment
# output.
# ──────────────────────────────────────────────────────────────────

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
        best_fitness_key="best_individual_fitness",  # for adaptive gate
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

# Adaptive metric detection threshold: if best fitness exceeds mean
# fitness by more than K standard deviations of the fitness distribution,
# switch to best-fitness gating.
BASIN_DETECTION_K = 2.33  # 99th percentile of mean-fitness distribution


class AdaptiveSensitivityPilot:
    """Three-phase sensitivity pilot for LWR-EGGROLL.

    Args:
        train_fn: Callable(seed, rank_spec_dict, label, **kwargs) -> dict.
            The result dict must contain the key specified by fitness_key.
            Optionally also contains best_fitness_key for adaptive gate.
        layer_shapes: {layer_name: (out_dim, in_dim)} for every layer.
        rank_set: List of allowed rank values. Default [0, 1, 2, 4, 8].
        fitness_key: Key in train_fn result for the mean fitness metric.
        best_fitness_key: Key for best-individual fitness (adaptive gate).
            If None, adaptive detection is disabled and fitness_key is
            used throughout.
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

        # Active fitness key — may be switched adaptively after Phase 1
        self._active_fitness_key = fitness_key
        self._metric_switched = False

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
        self.tier_ranks = []
        for r in self.nonzero_ranks:
            if r not in self.tier_ranks:
                self.tier_ranks.append(r)

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
        """Extract fitness using the currently active metric key."""
        return float(result[self._active_fitness_key])

    def _get_mean_fitness(self, result: dict) -> float:
        """Always extract the mean fitness metric (for logging)."""
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

    def _detect_flat_basin(self, phase1_results: List[dict]) -> bool:
        """Check if the fitness landscape exhibits a flat initial basin.

        Compares best fitness vs mean fitness across all Phase 1 runs.
        If best separates from mean by more than K standard deviations,
        the landscape has a flat basin where mean fitness is uninformative.

        Args:
            phase1_results: List of result dicts from Phase 1 elevation runs.

        Returns:
            True if flat basin detected (should switch to best fitness).
        """
        if self.best_fitness_key is None:
            return False

        mean_vals = []
        best_vals = []
        for result in phase1_results:
            m = self._get_mean_fitness(result)
            b = self._get_best_fitness(result)
            if m is not None and b is not None:
                mean_vals.append(m)
                best_vals.append(b)

        if len(mean_vals) < 2:
            return False

        mean_arr = np.array(mean_vals)
        best_arr = np.array(best_vals)

        # Separation: how far best is from mean, relative to variance
        separations = best_arr - mean_arr
        mean_std = np.std(mean_arr)

        if mean_std < 1e-8:
            # Mean fitness has no variance at all — definitely flat
            avg_separation = np.mean(separations)
            if avg_separation > 0:
                return True
            return False

        # Average separation in units of mean-fitness std
        avg_separation_normalized = np.mean(separations) / mean_std

        detected = avg_separation_normalized > BASIN_DETECTION_K
        print(f"  Basin detection: avg separation = "
              f"{np.mean(separations):.4f}, mean std = {mean_std:.4f}, "
              f"normalized = {avg_separation_normalized:.2f} "
              f"(threshold = {BASIN_DETECTION_K})")
        return detected

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
            phase3_decision, metric_used, metric_switched.
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
        print(f"  Fitness key (mean): {self.fitness_key}")
        print(f"  Fitness key (best): {self.best_fitness_key}")
        print(f"  Adaptive metric detection: "
              f"{'enabled' if self.best_fitness_key else 'disabled'}")
        print(f"  Seeds: {self.n_seeds}")
        print("=" * 60, flush=True)

        # ── Phase 1: Elevation ────────────────────────────────
        print("\n--- Phase 1: Elevation (perturbation magnitude) ---")
        print(f"  Background: r={self.floor_rank}, "
              f"Elevation: r={self.max_rank}", flush=True)

        phase1_magnitudes = {}
        phase1_all_results = []  # collect for basin detection
        for name, shape in zip(self.names, self.shapes_list):
            spec = self._make_spec(shape, self.max_rank, self.floor_rank)

            elevation_values = []
            print(f"\n  Elevating {name} {shape} to r={spec[shape]}:")
            for seed in seeds:
                print(f"    seed={seed}...", end="", flush=True)
                result = self._run_train(seed, spec, f"phase1_elevate_{name}")
                phase1_all_results.append(result)
                val = self._get_mean_fitness(result)
                elevation_values.append(val)
                best_val = self._get_best_fitness(result)
                print(f" mean={val:.4f}"
                      + (f"  best={best_val:.4f}" if best_val is not None else ""))

            phase1_magnitudes[name] = float(np.var(elevation_values))
            print(f"  {name}: variance = {phase1_magnitudes[name]:.6f}")

        p1_ordering = sorted(phase1_magnitudes.keys(),
                             key=lambda n: phase1_magnitudes[n], reverse=True)
        print(f"\n  Phase 1 ordering: {' > '.join(p1_ordering)}")

        # ── Adaptive metric detection ─────────────────────────
        print("\n--- Adaptive Metric Detection ---")
        flat_basin = self._detect_flat_basin(phase1_all_results)
        if flat_basin and self.best_fitness_key is not None:
            self._active_fitness_key = self.best_fitness_key
            self._metric_switched = True
            print(f"  ** FLAT BASIN DETECTED **")
            print(f"  Switching pilot metric: {self.fitness_key} -> "
                  f"{self.best_fitness_key}")
            print(f"  Rationale: mean fitness is uninformative in the initial "
                  f"basin; best fitness detects early basin escape.")
        else:
            self._active_fitness_key = self.fitness_key
            self._metric_switched = False
            print(f"  No flat basin detected: best and mean fitness "
                  f"track closely.")
            print(f"  Keeping pilot metric: {self.fitness_key}")
        print(f"  Active metric for Phase 2 & 3: {self._active_fitness_key}",
              flush=True)

        # ── Phase 2: Causal Ablation ──────────────────────────
        print(f"\n--- Phase 2: Causal Ablation ---")
        print(f"  Baseline: all at r={self.baseline_rank}")
        print(f"  Ablation: drop each to r={self.floor_rank}")
        print(f"  Metric: {self._active_fitness_key}", flush=True)

        # Baseline
        baseline_spec = self._make_uniform_spec(self.baseline_rank)
        baseline_values = []
        baseline_mean_values = []
        baseline_best_values = []
        for seed in seeds:
            print(f"  Baseline seed={seed}...", end="", flush=True)
            result = self._run_train(
                seed, baseline_spec, f"baseline_r{self.baseline_rank}")
            val = self._get_fitness(result)
            baseline_values.append(val)
            baseline_mean_values.append(self._get_mean_fitness(result))
            _b = self._get_best_fitness(result)
            if _b is not None:
                baseline_best_values.append(_b)
            print(f" {self._active_fitness_key}={val:.4f}")
        baseline_avg = float(np.mean(baseline_values))
        baseline_mean_avg = float(np.mean(baseline_mean_values))
        baseline_best_avg = float(np.mean(baseline_best_values)) if baseline_best_values else None
        print(f"  Baseline average: {baseline_avg:.4f}")

        # Ablate each layer
        degradations = {}
        degradations_mean = {}
        degradations_best = {}
        ablated_raw = {}
        for name, shape in zip(self.names, self.shapes_list):
            ablated_spec = self._make_uniform_spec(self.baseline_rank)
            ablated_spec[shape] = self.floor_rank

            ablated_values = []
            ablated_mean_values = []
            ablated_best_values = []
            print(f"\n  Ablating {name} {shape} to r={self.floor_rank}:")
            for seed in seeds:
                print(f"    seed={seed}...", end="", flush=True)
                result = self._run_train(
                    seed, ablated_spec, f"ablate_{name}")
                val = self._get_fitness(result)
                ablated_values.append(val)
                ablated_mean_values.append(self._get_mean_fitness(result))
                _b = self._get_best_fitness(result)
                if _b is not None:
                    ablated_best_values.append(_b)
                print(f" {self._active_fitness_key}={val:.4f}")

            ablated_avg = float(np.mean(ablated_values))
            degradation = baseline_avg - ablated_avg
            degradations[name] = degradation

            _ab_mean = float(np.mean(ablated_mean_values))
            degradations_mean[name] = baseline_mean_avg - _ab_mean
            if ablated_best_values and baseline_best_avg is not None:
                _ab_best = float(np.mean(ablated_best_values))
                degradations_best[name] = baseline_best_avg - _ab_best
                ablated_raw[name] = {"mean": _ab_mean, "best": _ab_best}
            else:
                ablated_raw[name] = {"mean": _ab_mean, "best": None}
            print(f"  {name}: degradation = {degradation:+.4f}")

        ordering = sorted(degradations.keys(),
                          key=lambda n: degradations[n], reverse=True)
        print(f"\n  Sensitivity ordering: {' > '.join(ordering)}")

        # ── Phase 1 x Phase 2 cross-referencing ──────────────
        print("\n--- Phase 1 x Phase 2 Cross-Referencing ---")

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
                  f"-> r={assigned}")

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
                  f"for {least_name}")
            print(f"  Metric: {self._active_fitness_key}", flush=True)

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
                mean_val = self._get_mean_fitness(result)
                best_val = self._get_best_fitness(result)
                r0_mean_vals.append(mean_val)
                if best_val is not None:
                    r0_best_vals.append(best_val)
                print(f" mean={mean_val:.4f}"
                      + (f"  best={best_val:.4f}" if best_val is not None else ""))

            # Rank floor condition
            print(f"\n  Condition B: {least_name} at rank {self.floor_rank}")
            r1_mean_vals = []
            r1_best_vals = []
            for seed in seeds:
                print(f"    seed={seed}...", end="", flush=True)
                result = self._run_train(
                    seed, spec_r1, f"phase3_{least_name}_r{self.floor_rank}")
                mean_val = self._get_mean_fitness(result)
                best_val = self._get_best_fitness(result)
                r1_mean_vals.append(mean_val)
                if best_val is not None:
                    r1_best_vals.append(best_val)
                print(f" mean={mean_val:.4f}"
                      + (f"  best={best_val:.4f}" if best_val is not None else ""))

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
            print(f"  Phase 3 decision: {least_name} -> rank {phase3_decision}")

        # ── Strategy selection ────────────────────────────────
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
        if self._metric_switched:
            print(f"NOTE: Pilot used '{self._active_fitness_key}' (adaptive "
                  f"switch from '{self.fitness_key}' due to flat basin "
                  f"detection)")

        return {
            "allocation": rec.rank_allocation_shapes,
            "allocation_named": rec.rank_allocation,
            "ordering": ordering,
            "strategy_recommendation": rec,
            "phase1_magnitudes": phase1_magnitudes,
            "phase1_ordering": p1_ordering,
            "phase2_degradations": degradations,
            "phase2_degradations_mean": degradations_mean,
            "phase2_degradations_best": degradations_best,
            "phase2_baseline_mean": baseline_mean_avg,
            "phase2_baseline_best": baseline_best_avg,
            "phase2_ablated_raw": ablated_raw,
            "phase3_decision": phase3_decision,
            "phase3_layer": least_name,
            "elapsed_seconds": elapsed,
            "metric_used": self._active_fitness_key,
            "metric_switched": self._metric_switched,
        }
