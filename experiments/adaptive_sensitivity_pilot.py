"""Adaptive Sensitivity Pilot for LWR-EGGROLL.

Automatically determines the pilot regime based on model scale:
  - ALLOCATION mode (standard):  rank set {0, 1, 2, 4, 8}, Phase 2 drops target to r=1
  - BINARY INCLUSION mode (large-scale): rank set {0, 1}, Phase 2 drops target to r=0

The regime is determined per-layer by comparing min(m, n) of each weight
matrix against `max_rank`. If the effective maximum rank across all layers
is <= 1, the pilot switches to binary inclusion mode. This can also be
forced via `force_binary=True` for compute-constrained large-model settings
where only rank 0 or 1 is practical regardless of layer dimensions.

Usage:
    from adaptive_sensitivity_pilot import AdaptiveSensitivityPilot

    pilot = AdaptiveSensitivityPilot(
        train_fn=my_training_function,
        layer_shapes={(256, 784): "input", (256, 256): "hidden", (10, 256): "output"},
        dataset=(X_train, y_train, X_test, y_test),
    )
    result = pilot.run(n_seeds=5)
    print(result.rank_allocation)   # e.g. {(256, 784): 8, (256, 256): 2, (10, 256): 0}
    print(result.mode)              # "allocation" or "binary_inclusion"
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Rank set and regime detection
# ─────────────────────────────────────────────────────────────────────

FULL_RANK_SET = [0, 1, 2, 4, 8]
BINARY_RANK_SET = [0, 1]


def detect_regime(
    layer_shapes: Dict[Tuple[int, int], str],
    max_rank: int = 8,
    force_binary: bool = False,
) -> Tuple[str, Dict[Tuple[int, int], int]]:
    """Detect whether to use allocation mode or binary inclusion mode.

    Args:
        layer_shapes: Dict mapping (out_dim, in_dim) shape tuples to
            human-readable layer group names.
        max_rank: Maximum rank to consider. Capped per-layer at min(m, n).
            For billion-parameter models where only r=1 is practical,
            set this to 1.
        force_binary: If True, force binary inclusion mode regardless
            of layer dimensions.

    Returns:
        (mode, per_layer_max_rank) where mode is "allocation" or
        "binary_inclusion", and per_layer_max_rank maps each shape
        to the highest usable rank for that layer.
    """
    if force_binary:
        per_layer_max = {shape: 1 for shape in layer_shapes}
        return "binary_inclusion", per_layer_max

    per_layer_max = {}
    for shape in layer_shapes:
        m, n = shape
        layer_max = min(min(m, n), max_rank)
        # Snap down to nearest value in the rank set
        usable = [r for r in FULL_RANK_SET if r <= layer_max and r > 0]
        per_layer_max[shape] = max(usable) if usable else 1

    effective_max = max(per_layer_max.values())
    if effective_max <= 1:
        mode = "binary_inclusion"
    else:
        mode = "allocation"

    return mode, per_layer_max


# ─────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    """Result from a single phase of the pilot."""
    layer_name: str
    shape: Tuple[int, int]
    metric_name: str            # "fitness_variance" or "accuracy_degradation"
    values_per_seed: List[float]
    mean: float
    std: float


@dataclass
class PilotResult:
    """Complete result from the adaptive sensitivity pilot."""
    mode: str                                           # "allocation" or "binary_inclusion"
    phase1_results: List[PhaseResult]                   # fitness variance per layer
    phase2_results: List[PhaseResult]                   # degradation per layer
    sensitivity_ordering: List[str]                     # layer names, most sensitive first
    rank_allocation: Dict[Tuple[int, int], int]         # shape -> assigned rank
    rank_allocation_named: Dict[str, int]               # layer_name -> assigned rank
    phase2_baseline_accuracy: float                     # baseline accuracy for Phase 2
    total_rank_budget: int                              # sum of assigned ranks
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable summary of the pilot result."""
        lines = [
            f"{'='*60}",
            f"ADAPTIVE SENSITIVITY PILOT — {self.mode.upper()} MODE",
            f"{'='*60}",
            "",
            "Phase 1: Isolated Fitness Variance",
            "-" * 40,
        ]
        for r in sorted(self.phase1_results, key=lambda x: x.mean, reverse=True):
            lines.append(f"  {r.layer_name:>10s}  {r.shape}  "
                         f"variance = {r.mean:.6f} ± {r.std:.6f}")

        lines += [
            "",
            f"Phase 2: Causal Ablation (baseline acc = {self.phase2_baseline_accuracy:.4f})",
            "-" * 40,
        ]
        drop_label = "drop-to-0" if self.mode == "binary_inclusion" else "drop-to-1"
        for r in sorted(self.phase2_results, key=lambda x: x.mean, reverse=True):
            lines.append(f"  {r.layer_name:>10s}  {r.shape}  "
                         f"degradation ({drop_label}) = {r.mean:+.4f} ± {r.std:.4f}")

        lines += [
            "",
            "Sensitivity Ordering (most → least):",
            f"  {' >> '.join(self.sensitivity_ordering)}",
            "",
            "Assigned Rank Allocation:",
        ]
        for name in self.sensitivity_ordering:
            rank = self.rank_allocation_named[name]
            lines.append(f"  {name:>10s} → r = {rank}")

        lines += [
            "",
            f"Total rank budget: {self.total_rank_budget}",
            f"{'='*60}",
        ]
        return "\n".join(lines)

    def to_json(self, path: Optional[str] = None) -> dict:
        """Serialise to a JSON-safe dict. Optionally write to file."""
        data = {
            "mode": self.mode,
            "sensitivity_ordering": self.sensitivity_ordering,
            "rank_allocation": {str(k): v for k, v in self.rank_allocation.items()},
            "rank_allocation_named": self.rank_allocation_named,
            "total_rank_budget": self.total_rank_budget,
            "phase2_baseline_accuracy": self.phase2_baseline_accuracy,
            "phase1": [
                {
                    "layer": r.layer_name,
                    "shape": list(r.shape),
                    "metric": r.metric_name,
                    "values": r.values_per_seed,
                    "mean": r.mean,
                    "std": r.std,
                }
                for r in self.phase1_results
            ],
            "phase2": [
                {
                    "layer": r.layer_name,
                    "shape": list(r.shape),
                    "metric": r.metric_name,
                    "values": r.values_per_seed,
                    "mean": r.mean,
                    "std": r.std,
                }
                for r in self.phase2_results
            ],
            "metadata": self.metadata,
        }
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Pilot result saved to {path}")
        return data


# ─────────────────────────────────────────────────────────────────────
# The adaptive pilot itself
# ─────────────────────────────────────────────────────────────────────

class AdaptiveSensitivityPilot:
    """Adaptive sensitivity pilot for LWR-EGGROLL.

    Automatically detects whether to run in allocation mode or binary
    inclusion mode based on layer dimensions and max_rank.

    Args:
        train_fn: A callable with signature:
            train_fn(seed, X_train, y_train, X_test, y_test,
                     rank_spec: dict, label: str)
            -> dict with at least:
                "test_acc": float (final test accuracy),
                "fitness_variance_history": list[float] (optional, for Phase 1)
            If fitness_variance_history is not returned, the pilot will
            use test accuracy as the Phase 1 metric instead.

        layer_shapes: Dict mapping (out_dim, in_dim) shape tuples to
            human-readable layer group names. Example:
            {(256, 784): "input", (256, 256): "hidden", (10, 256): "output"}

        dataset: Tuple of (X_train, y_train, X_test, y_test).

        max_rank: Maximum rank to consider. Set to 1 for large-model
            binary inclusion mode. Default 8.

        force_binary: Force binary inclusion mode regardless of dimensions.

        baseline_rank: The uniform rank used for the Phase 2 baseline
            in allocation mode. Ignored in binary mode (baseline is r=1).
            Default 4.

        output_dir: Directory for saving results. Default "results/pilot".
    """

    def __init__(
        self,
        train_fn: Callable,
        layer_shapes: Dict[Tuple[int, int], str],
        dataset: Tuple,
        max_rank: int = 8,
        force_binary: bool = False,
        baseline_rank: int = 4,
        output_dir: str = "results/pilot",
    ):
        self.train_fn = train_fn
        self.layer_shapes = layer_shapes
        self.shapes_list = list(layer_shapes.keys())
        self.names_list = [layer_shapes[s] for s in self.shapes_list]
        self.dataset = dataset
        self.max_rank = max_rank
        self.force_binary = force_binary
        self.baseline_rank = baseline_rank
        self.output_dir = Path(output_dir)

        # Detect regime
        self.mode, self.per_layer_max_rank = detect_regime(
            layer_shapes, max_rank, force_binary
        )

        print(f"Detected regime: {self.mode.upper()}")
        if self.mode == "binary_inclusion":
            print("  Rank set: {0, 1}")
            print("  Phase 2: drop-to-0 ablation (freeze target layer)")
            self._phase2_drop_rank = 0
            self._phase2_baseline_rank = 1
        else:
            print(f"  Rank set: {{0, 1, 2, 4, 8}} (capped per-layer)")
            print(f"  Phase 2: drop-to-1 ablation from baseline r={baseline_rank}")
            self._phase2_drop_rank = 1
            self._phase2_baseline_rank = baseline_rank

        for shape, name in layer_shapes.items():
            print(f"  {name:>10s}  {shape}  max usable rank = {self.per_layer_max_rank[shape]}")

    # ── helpers ──────────────────────────────────────────────────────

    def _make_rank_spec(self, rank_per_shape: Dict[Tuple[int, int], int]) -> dict:
        """Build rank_spec dict from per-shape assignments."""
        return dict(rank_per_shape)

    def _isolation_spec(self, target_shape: Tuple[int, int], target_rank: int) -> dict:
        """Rank spec that perturbs only the target layer."""
        spec = {s: 0 for s in self.shapes_list}
        spec[target_shape] = target_rank
        return spec

    def _ablation_spec(
        self, target_shape: Tuple[int, int], baseline: int, drop_to: int
    ) -> dict:
        """Rank spec that drops the target layer while others stay at baseline."""
        spec = {s: baseline for s in self.shapes_list}
        spec[target_shape] = drop_to
        return spec

    def _uniform_spec(self, rank: int) -> dict:
        """Uniform rank for all layers."""
        return {s: rank for s in self.shapes_list}

    def _run_condition(
        self, rank_spec: dict, label: str, seeds: List[int]
    ) -> List[dict]:
        """Run training across seeds for a given rank spec."""
        X_tr, y_tr, X_te, y_te = self.dataset
        results = []
        for seed in seeds:
            print(f"    seed={seed} ...", end="", flush=True)
            t0 = time.time()
            result = self.train_fn(
                seed=seed,
                X_train=X_tr, y_train=y_tr,
                X_test=X_te, y_test=y_te,
                rank_spec=rank_spec,
                label=label,
            )
            elapsed = time.time() - t0
            print(f" acc={result['test_acc']:.4f}  ({elapsed:.1f}s)")
            results.append(result)
        return results

    # ── Phase 1 ─────────────────────────────────────────────────────

    def _run_phase1(self, seeds: List[int]) -> List[PhaseResult]:
        """Phase 1: Isolated fitness variance per layer."""
        print("\n" + "=" * 50)
        print("PHASE 1: Isolated Fitness Variance")
        print("=" * 50)

        phase1_results = []
        for shape, name in zip(self.shapes_list, self.names_list):
            # In binary mode, isolate at r=1. In allocation mode, isolate
            # at the per-layer max rank to get the strongest signal.
            if self.mode == "binary_inclusion":
                iso_rank = 1
            else:
                iso_rank = min(self.per_layer_max_rank[shape], self.max_rank)

            spec = self._isolation_spec(shape, iso_rank)
            label = f"phase1_isolate_{name}_r{iso_rank}"
            print(f"\n  Layer: {name} {shape}, isolated at r={iso_rank}")

            run_results = self._run_condition(spec, label, seeds)

            # Extract fitness variance if available, else use test_acc
            # as a proxy for "how much signal this layer generates"
            if "fitness_variance_history" in run_results[0]:
                # Use mean fitness variance across generations per seed
                values = [
                    float(np.mean(r["fitness_variance_history"]))
                    for r in run_results
                ]
                metric = "fitness_variance"
            else:
                # Fallback: use test accuracy as a proxy.
                # Higher accuracy when isolated = layer generates more
                # useful signal on its own.
                values = [r["best_test_acc"] for r in run_results]
                metric = "test_accuracy_isolated"

            phase1_results.append(PhaseResult(
                layer_name=name,
                shape=shape,
                metric_name=metric,
                values_per_seed=values,
                mean=float(np.mean(values)),
                std=float(np.std(values)),
            ))

        return phase1_results

    # ── Phase 2 ─────────────────────────────────────────────────────

    def _run_phase2(self, seeds: List[int]) -> Tuple[float, List[PhaseResult]]:
        """Phase 2: Causal ablation."""
        print("\n" + "=" * 50)
        bl_rank = self._phase2_baseline_rank
        drop_rank = self._phase2_drop_rank
        print(f"PHASE 2: Causal Ablation (baseline r={bl_rank}, drop to r={drop_rank})")
        print("=" * 50)

        # Run baseline
        print(f"\n  Baseline: uniform r={bl_rank}")
        baseline_spec = self._uniform_spec(bl_rank)
        baseline_results = self._run_condition(baseline_spec, f"phase2_baseline_r{bl_rank}", seeds)
        baseline_accs = [r["best_test_acc"] for r in baseline_results]
        baseline_mean = float(np.mean(baseline_accs))
        print(f"  Baseline accuracy: {baseline_mean:.4f} ± {np.std(baseline_accs):.4f}")

        # Ablate each layer
        phase2_results = []
        for shape, name in zip(self.shapes_list, self.names_list):
            spec = self._ablation_spec(shape, bl_rank, drop_rank)
            label = f"phase2_ablate_{name}_to_r{drop_rank}"
            print(f"\n  Ablating: {name} {shape} → r={drop_rank}")

            run_results = self._run_condition(spec, label, seeds)
            ablated_accs = [r["best_test_acc"] for r in run_results]

            # Degradation: positive = layer is important (removing rank hurts)
            #              negative = layer is noisy (removing rank helps)
            degradations = [baseline_mean - acc for acc in ablated_accs]

            phase2_results.append(PhaseResult(
                layer_name=name,
                shape=shape,
                metric_name="accuracy_degradation",
                values_per_seed=degradations,
                mean=float(np.mean(degradations)),
                std=float(np.std(degradations)),
            ))

        return baseline_mean, phase2_results

    # ── Rank assignment ─────────────────────────────────────────────

    def _assign_ranks(
        self,
        phase1_results: List[PhaseResult],
        phase2_results: List[PhaseResult],
    ) -> Dict[Tuple[int, int], int]:
        """Derive rank allocation from pilot results.

        In ALLOCATION mode:
            - Rank layers by Phase 1 sensitivity (descending).
            - Assign ranks from {0, 1, 2, 4, 8} proportional to position.
            - For the least sensitive layer: check Phase 2 causal ablation.
              If degradation <= 0 (negative causal effect), assign r=0.

        In BINARY INCLUSION mode:
            - For each layer: check Phase 2 degradation.
              If degradation > 0 (freezing hurts), assign r=1 (include).
              If degradation <= 0 (freezing helps or is neutral), assign r=0 (exclude).
        """
        # Build Phase 2 lookup
        p2_by_shape = {r.shape: r for r in phase2_results}

        if self.mode == "binary_inclusion":
            # Binary decision: include (r=1) or exclude (r=0)
            allocation = {}
            for r1 in phase1_results:
                p2 = p2_by_shape[r1.shape]
                if p2.mean > 0:
                    # Freezing this layer hurts accuracy → keep it
                    allocation[r1.shape] = 1
                else:
                    # Freezing this layer is neutral or helps → exclude it
                    allocation[r1.shape] = 0
            return allocation

        else:
            # Allocation mode: rank layers by Phase 1 sensitivity
            sorted_by_sensitivity = sorted(
                phase1_results, key=lambda x: x.mean, reverse=True
            )
            n_layers = len(sorted_by_sensitivity)

            # Available non-zero ranks, descending
            available_ranks = sorted(
                [r for r in FULL_RANK_SET if r > 0], reverse=True
            )

            allocation = {}
            for i, layer_result in enumerate(sorted_by_sensitivity):
                shape = layer_result.shape
                max_for_layer = self.per_layer_max_rank[shape]
                p2 = p2_by_shape[shape]

                if i == n_layers - 1:
                    # Least sensitive layer: check Phase 2
                    if p2.mean <= 0:
                        allocation[shape] = 0
                    else:
                        allocation[shape] = 1
                else:
                    # Proportional assignment based on sensitivity position
                    # Most sensitive → highest available, etc.
                    if i < len(available_ranks):
                        candidate = available_ranks[i]
                    else:
                        candidate = 1
                    # Cap at layer's max usable rank
                    allocation[shape] = min(candidate, max_for_layer)

            return allocation

    # ── Main entry point ────────────────────────────────────────────

    def run(
        self,
        n_seeds: int = 5,
        save_results: bool = True,
    ) -> PilotResult:
        """Run the full adaptive sensitivity pilot.

        Args:
            n_seeds: Number of random seeds per condition.
            save_results: Whether to save the JSON result to output_dir.

        Returns:
            PilotResult with the rank allocation and all measurements.
        """
        seeds = list(range(n_seeds))

        print(f"\nAdaptive Sensitivity Pilot")
        print(f"Mode: {self.mode}")
        print(f"Seeds: {seeds}")
        print(f"Layers: {len(self.layer_shapes)}")

        t_start = time.time()

        # Phase 1
        phase1_results = self._run_phase1(seeds)

        # Phase 2
        baseline_acc, phase2_results = self._run_phase2(seeds)

        # Derive allocation
        allocation = self._assign_ranks(phase1_results, phase2_results)

        # Build named allocation and ordering
        shape_to_name = dict(zip(self.shapes_list, self.names_list))
        allocation_named = {shape_to_name[s]: r for s, r in allocation.items()}

        # Sensitivity ordering from Phase 1 (descending)
        ordering = [
            r.layer_name
            for r in sorted(phase1_results, key=lambda x: x.mean, reverse=True)
        ]

        total_budget = sum(allocation.values())
        elapsed = time.time() - t_start

        result = PilotResult(
            mode=self.mode,
            phase1_results=phase1_results,
            phase2_results=phase2_results,
            sensitivity_ordering=ordering,
            rank_allocation=allocation,
            rank_allocation_named=allocation_named,
            phase2_baseline_accuracy=baseline_acc,
            total_rank_budget=total_budget,
            metadata={
                "n_seeds": n_seeds,
                "max_rank": self.max_rank,
                "force_binary": self.force_binary,
                "baseline_rank": self.baseline_rank,
                "wall_clock_seconds": round(elapsed, 1),
            },
        )

        print("\n" + result.summary())

        if save_results:
            out_path = self.output_dir / f"pilot_{self.mode}.json"
            result.to_json(str(out_path))

        return result


# ─────────────────────────────────────────────────────────────────────
# Convenience wrapper for the existing codebase
# ─────────────────────────────────────────────────────────────────────

def run_pilot_with_hyperscalees(
    layer_shapes: Dict[Tuple[int, int], str],
    dataset: Tuple,
    max_rank: int = 8,
    force_binary: bool = False,
    baseline_rank: int = 4,
    n_seeds: int = 5,
    output_dir: str = "results/pilot",
):
    """Convenience wrapper that plugs into the HyperscaleES training functions.

    Expects `train_lwr_eggroll` from experiments.compare_4_methods_mnist
    to be importable. This function adapts its signature to match the
    pilot's expected interface.

    Example:
        from adaptive_sensitivity_pilot import run_pilot_with_hyperscalees

        result = run_pilot_with_hyperscalees(
            layer_shapes={(256, 784): "input", (256, 256): "hidden", (10, 256): "output"},
            dataset=(X_train, y_train, X_test, y_test),
            max_rank=8,       # normal allocation mode
            # max_rank=1,     # or: binary inclusion mode for large models
            n_seeds=5,
        )
        print(result.rank_allocation)
    """
    # Lazy import to avoid circular deps / missing module errors
    # when this file is used as a reference in other contexts.
    from experiments.compare_4_methods_mnist import train_lwr_eggroll

    def adapted_train_fn(seed, X_train, y_train, X_test, y_test, rank_spec, label):
        result = train_lwr_eggroll(
            seed=seed,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            rank_spec=rank_spec,
            label=label,
        )
        return result

    pilot = AdaptiveSensitivityPilot(
        train_fn=adapted_train_fn,
        layer_shapes=layer_shapes,
        dataset=dataset,
        max_rank=max_rank,
        force_binary=force_binary,
        baseline_rank=baseline_rank,
        output_dir=output_dir,
    )

    return pilot.run(n_seeds=n_seeds)


# ─────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Adaptive Sensitivity Pilot for LWR-EGGROLL"
    )
    parser.add_argument(
        "--max-rank", type=int, default=8,
        help="Maximum rank to consider. Set to 1 for binary inclusion mode. (default: 8)"
    )
    parser.add_argument(
        "--force-binary", action="store_true",
        help="Force binary inclusion mode regardless of layer dimensions."
    )
    parser.add_argument(
        "--baseline-rank", type=int, default=4,
        help="Uniform rank for Phase 2 baseline in allocation mode. (default: 4)"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of random seeds per condition. (default: 5)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/pilot",
        help="Directory for saving results. (default: results/pilot)"
    )
    args = parser.parse_args()

    # Default: standard 3-hidden MLP on MNIST
    # Override layer_shapes here for different architectures.
    import jax.numpy as jnp
    import numpy as np_cpu
    from torchvision import datasets as tv_datasets
    from pathlib import Path as P

    DATA_DIR = P("data")
    DATA_DIR.mkdir(exist_ok=True)

    print("Loading MNIST...")
    train_ds = tv_datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    test_ds = tv_datasets.MNIST(root=str(DATA_DIR), train=False, download=True)
    X_train = jnp.asarray(np_cpu.array(train_ds.data, dtype=np_cpu.float32).reshape(-1, 784) / 255.0)
    y_train = jnp.asarray(np_cpu.array(train_ds.targets, dtype=np_cpu.int32))
    X_test = jnp.asarray(np_cpu.array(test_ds.data, dtype=np_cpu.float32).reshape(-1, 784) / 255.0)
    y_test = jnp.asarray(np_cpu.array(test_ds.targets, dtype=np_cpu.int32))

    layer_shapes = {
        (256, 784): "input",
        (256, 256): "hidden",
        (10, 256): "output",
    }

    result = run_pilot_with_hyperscalees(
        layer_shapes=layer_shapes,
        dataset=(X_train, y_train, X_test, y_test),
        max_rank=args.max_rank,
        force_binary=args.force_binary,
        baseline_rank=args.baseline_rank,
        n_seeds=args.n_seeds,
        output_dir=args.output_dir,
    )

    print("\nDone.")
    print(f"Rank allocation: {result.rank_allocation_named}")
