"""Post-pilot strategy selector.

Takes Phase 2 degradation scores (and optionally a PilotResult)
and returns:
  1. Whether LWR or uniform EGGROLL is recommended
  2. The specific rank allocation to use
  3. A summative finding explaining why

Thresholds derived empirically:
  MNIST family:   CV ~ 1.5-3.0  → LWR (high confidence)
  LunarLander:    CV ~ 0.3-0.5  → uniform
  Threshold:      CV > 0.8      → LWR recommended
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List


# ── Thresholds ─────────────────────────────────────────────────────
CV_THRESHOLD = 0.8
CV_HIGH_CONFIDENCE = 1.5
SPREAD_MIN = 0.005

# Default rank set (constrained by JAX JIT recompilation)
RANK_SET = [0, 1, 2, 4, 8]


@dataclass
class StrategyRecommendation:
    """Complete output of the strategy selector."""
    strategy: str                              # "lwr" or "uniform"
    confidence: str                            # "high", "moderate", "low"
    rank_allocation: Dict[str, int]            # layer_name -> assigned rank
    rank_allocation_shapes: Dict[Tuple[int, int], int]  # shape -> rank (for HyperscaleES)
    total_budget: int                          # sum of ranks
    cv: float                                  # coefficient of variation
    spread: float                              # max - min degradation
    degradation_scores: Dict[str, float]       # layer_name -> mean degradation
    finding: str                               # 1-2 line summative finding

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"STRATEGY RECOMMENDATION: {self.strategy.upper()}",
            f"{'='*60}",
            "",
            "Rank allocation:",
        ]
        for name, rank in self.rank_allocation.items():
            deg = self.degradation_scores.get(name, 0.0)
            lines.append(f"  {name:>10s} → r = {rank}  (degradation = {deg:+.4f})")
        lines += [
            f"  {'budget':>10s} = {self.total_budget}",
            "",
            f"Finding: {self.finding}",
            f"{'='*60}",
        ]
        return "\n".join(lines)


def _assign_lwr_ranks(
    ordering: List[str],
    degradation_scores: Dict[str, float],
    max_rank: int = 8,
) -> Dict[str, int]:
    """Assign graded ranks based on sensitivity ordering.

    Rules:
      - Layers with negative degradation get rank 0
        (removing rank helped — actively harmful signal).
      - Remaining layers get descending ranks from RANK_SET.
      - Most sensitive gets max_rank.
    """
    available = sorted([r for r in RANK_SET if 0 < r <= max_rank], reverse=True)
    allocation = {}
    rank_idx = 0
    for name in ordering:
        if degradation_scores.get(name, 0.0) < 0:
            # Negative degradation: removing rank improved accuracy
            allocation[name] = 0
        elif rank_idx < len(available):
            allocation[name] = available[rank_idx]
            rank_idx += 1
        else:
            allocation[name] = 1  # fallback minimum
    return allocation


def _assign_uniform_ranks(
    layer_names: List[str],
    rank: int = 4,
) -> Dict[str, int]:
    """Assign uniform rank to all layers."""
    return {name: rank for name in layer_names}


def select_strategy(
    pilot_result,
    baseline_rank: int = 4,
    max_rank: int = 8,
    layer_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    verbose: bool = True,
) -> StrategyRecommendation:
    """Classify a PilotResult into LWR or uniform, with allocation.

    Parameters
    ----------
    pilot_result : PilotResult
        Output from the adaptive sensitivity pilot.
    baseline_rank : int
        Rank to use if uniform is recommended.
    max_rank : int
        Maximum rank for LWR allocation.
    layer_shapes : dict, optional
        {layer_name: (out_dim, in_dim)} for shape-keyed output.
        If None, uses pilot_result metadata or empty.
    verbose : bool
        Print summary.

    Returns
    -------
    StrategyRecommendation
    """
    degradation_scores = {
        r.layer_name: r.mean for r in pilot_result.phase2_results
    }
    ordering = pilot_result.sensitivity_ordering

    # Resolve layer shapes
    if layer_shapes is None:
        layer_shapes = {
            r.layer_name: r.shape for r in pilot_result.phase2_results
        }

    return _decide(
        degradation_scores=degradation_scores,
        ordering=ordering,
        layer_shapes=layer_shapes,
        baseline_rank=baseline_rank,
        max_rank=max_rank,
        verbose=verbose,
    )


def select_strategy_from_dict(
    degradation_scores: Dict[str, float],
    layer_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    baseline_rank: int = 4,
    max_rank: int = 8,
    verbose: bool = True,
) -> StrategyRecommendation:
    """Classify from raw degradation scores.

    Parameters
    ----------
    degradation_scores : dict
        {layer_name: mean_degradation}
        Positive = removing rank hurts. Negative = removing rank helps.
    layer_shapes : dict, optional
        {layer_name: (out_dim, in_dim)} for shape-keyed output.
    baseline_rank : int
        Rank for uniform fallback.
    max_rank : int
        Maximum rank for LWR graded allocation.
    verbose : bool
        Print summary.
    """
    # Derive ordering from degradation (most sensitive = highest degradation)
    ordering = sorted(degradation_scores.keys(),
                      key=lambda k: degradation_scores[k], reverse=True)

    if layer_shapes is None:
        layer_shapes = {}

    return _decide(
        degradation_scores=degradation_scores,
        ordering=ordering,
        layer_shapes=layer_shapes,
        baseline_rank=baseline_rank,
        max_rank=max_rank,
        verbose=verbose,
    )


def _decide(
    degradation_scores: Dict[str, float],
    ordering: List[str],
    layer_shapes: Dict[str, Tuple[int, int]],
    baseline_rank: int,
    max_rank: int,
    verbose: bool,
) -> StrategyRecommendation:
    """Core decision logic."""
    values = np.array(list(degradation_scores.values()))
    spread = float(np.max(values) - np.min(values))
    mean_abs = float(np.abs(np.mean(values)))
    std = float(np.std(values))
    cv = 0.0 if mean_abs < 1e-8 else std / mean_abs

    # ── Decision ───────────────────────────────────────────────
    if spread < SPREAD_MIN:
        strategy = "uniform"
        confidence = "high"
        rank_alloc = _assign_uniform_ranks(list(degradation_scores.keys()),
                                           rank=baseline_rank)
        finding = (
            f"Phase 2 degradation spread is negligible ({spread:.4f}). "
            f"All layers respond similarly to rank reduction — "
            f"uniform r={baseline_rank} is sufficient, no benefit from "
            f"layer-wise differentiation."
        )

    elif cv > CV_HIGH_CONFIDENCE:
        strategy = "lwr"
        confidence = "high"
        rank_alloc = _assign_lwr_ranks(ordering, degradation_scores, max_rank=max_rank)
        most = ordering[0]
        least = ordering[-1]
        finding = (
            f"Large sensitivity gap (CV={cv:.2f}): {most} layer degrades "
            f"{degradation_scores[most]:+.4f} when reduced vs {least} at "
            f"{degradation_scores[least]:+.4f}. LWR allocation concentrates "
            f"rank on high-sensitivity layers for better efficiency."
        )

    elif cv > CV_THRESHOLD:
        strategy = "lwr"
        confidence = "moderate"
        rank_alloc = _assign_lwr_ranks(ordering, degradation_scores, max_rank=max_rank)
        most = ordering[0]
        least = ordering[-1]
        finding = (
            f"Moderate sensitivity gap (CV={cv:.2f}): {most} layer is more "
            f"sensitive ({degradation_scores[most]:+.4f}) than {least} "
            f"({degradation_scores[least]:+.4f}). LWR allocation likely "
            f"beneficial over uniform rank."
        )

    else:
        strategy = "uniform"
        confidence = "moderate" if cv > CV_THRESHOLD * 0.5 else "high"
        rank_alloc = _assign_uniform_ranks(list(degradation_scores.keys()),
                                           rank=baseline_rank)
        finding = (
            f"Low sensitivity gap (CV={cv:.2f}, spread={spread:.4f}). "
            f"Layers have similar sensitivity to rank reduction — "
            f"uniform r={baseline_rank} is recommended."
        )

    # Build shape-keyed dict for HyperscaleES
    rank_alloc_shapes = {}
    for name, rank in rank_alloc.items():
        if name in layer_shapes:
            rank_alloc_shapes[layer_shapes[name]] = rank

    total_budget = sum(rank_alloc.values())

    rec = StrategyRecommendation(
        strategy=strategy,
        confidence=confidence,
        rank_allocation=rank_alloc,
        rank_allocation_shapes=rank_alloc_shapes,
        total_budget=total_budget,
        cv=cv,
        spread=spread,
        degradation_scores=degradation_scores,
        finding=finding,
    )

    if verbose:
        print(rec.summary())

    return rec


# ── Validation ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Validating strategy selector against known results\n")

    # ── MNIST: input >> hidden >> output, strong gap ──
    print("--- MNIST ---")
    rec = select_strategy_from_dict(
        {"input": 0.0412, "hidden": 0.0089, "output": -0.0156},
        layer_shapes={
            "input": (256, 784), "hidden": (256, 256), "output": (10, 256)
        },
    )
    assert rec.strategy == "lwr"
    assert rec.rank_allocation["input"] == 8
    assert rec.rank_allocation["output"] == 0
    print(f"  Shape-keyed: {rec.rank_allocation_shapes}\n")

    # ── Fashion-MNIST: same pattern ──
    print("--- Fashion-MNIST ---")
    rec = select_strategy_from_dict(
        {"input": 0.0380, "hidden": 0.0102, "output": -0.0201},
    )
    assert rec.strategy == "lwr"
    assert rec.rank_allocation["input"] == 8
    print()

    # ── LunarLander symmetric: small spread ──
    print("--- LunarLander (symmetric) ---")
    rec = select_strategy_from_dict(
        {"input": 0.008, "hidden": 0.005, "output": 0.003},
    )
    assert rec.strategy == "uniform"
    assert rec.rank_allocation["input"] == 4  # uniform r=4
    assert rec.rank_allocation["output"] == 4
    print()

    # ── Edge: all zeros ──
    print("--- All zeros ---")
    rec = select_strategy_from_dict(
        {"input": 0.0, "hidden": 0.0, "output": 0.0},
    )
    assert rec.strategy == "uniform"
    print()

    # ── Edge: one layer massively dominant ──
    print("--- One dominant layer ---")
    rec = select_strategy_from_dict(
        {"input": 0.15, "hidden": 0.001, "output": -0.03},
    )
    assert rec.strategy == "lwr"
    assert rec.rank_allocation["input"] == 8
    print()

    print("All validations passed.")
