"""Post-pilot strategy selector and Phase 3 decision logic.

Takes Phase 2 degradation scores (and optionally a PilotResult)
and returns:
  1. Whether LWR or uniform EGGROLL is recommended
  2. The specific rank allocation to use
  3. A summative finding explaining why

Also provides phase3_freeze_decision() — the shared confirmation gate
used by all pilot implementations (supervised and RL).

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
    strategy: str                              # "lwr", "heterogeneous", or "uniform"
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


# ── Phase 3 confirmation gate ──────────────────────────────────────

def phase3_freeze_decision(
    r0_mean_fitnesses: List[float],
    r1_mean_fitnesses: List[float],
    candidate_shape: Tuple[int, int],
    all_layer_shapes: Dict[str, Tuple[int, int]],
    r0_best_fitnesses: Optional[List[float]] = None,
    r1_best_fitnesses: Optional[List[float]] = None,
) -> int:
    """Shared Phase 3 freeze decision with confirmation gate.

    When the candidate layer holds >50% of total parameters, requires
    rank 0 to win on BOTH mean and best fitness across seeds.
    Otherwise, uses mean fitness only.

    Args:
        r0_mean_fitnesses: Fitness metric per seed with candidate at rank 0.
        r1_mean_fitnesses: Fitness metric per seed with candidate at rank 1.
        candidate_shape: (out_dim, in_dim) of the candidate layer.
        all_layer_shapes: {name: (out_dim, in_dim)} for all layers.
        r0_best_fitnesses: Best individual fitness per seed at rank 0.
        r1_best_fitnesses: Best individual fitness per seed at rank 1.

    Returns:
        0 (freeze) or 1 (minimal perturbation).
    """
    r0_mean = float(np.mean(r0_mean_fitnesses))
    r1_mean = float(np.mean(r1_mean_fitnesses))

    # Compute frozen parameter fraction
    candidate_params = candidate_shape[0] * candidate_shape[1]
    total_params = sum(s[0] * s[1] for s in all_layer_shapes.values())
    frozen_fraction = candidate_params / total_params

    if frozen_fraction > 0.5:
        print(f"  Phase 3 gate: frozen fraction = {frozen_fraction:.1%} > 50%")
        mean_wins = r0_mean >= r1_mean

        if r0_best_fitnesses is not None and r1_best_fitnesses is not None:
            r0_best = float(np.mean(r0_best_fitnesses))
            r1_best = float(np.mean(r1_best_fitnesses))
            best_wins = r0_best >= r1_best
        else:
            best_wins = mean_wins

        decision = 0 if (mean_wins and best_wins) else 1
        print(f"  Mean fitness: r0={r0_mean:.4f}  r1={r1_mean:.4f}"
              f"  ({'r0' if mean_wins else 'r1'})")
        if r0_best_fitnesses is not None:
            print(f"  Best fitness: r0={r0_best:.4f}  r1={r1_best:.4f}"
                  f"  ({'r0' if best_wins else 'r1'})")
        print(f"  Gate: {'BOTH agree -> rank 0' if decision == 0 else 'disagreement -> rank 1'}")
    else:
        decision = 0 if r0_mean >= r1_mean else 1

    return decision


# ── Rank assignment ────────────────────────────────────────────────

def _assign_lwr_ranks(
    ordering: List[str],
    degradation_scores: Dict[str, float],
    max_rank: int = 8,
    phase1_magnitudes: Optional[Dict[str, float]] = None,
    rank_set: Optional[List[int]] = None,
) -> Dict[str, int]:
    """Assign graded ranks based on sensitivity ordering.

    All assigned values are drawn from rank_set. Tiers are derived
    from the set automatically:
      - Most sensitive  → tier 0 (= max non-zero rank in set)
      - Moderate        → tier 1 (one step down)
      - Low P1 magnitude → tier 2 (another step down)
      - Least sensitive → floor rank (smallest non-zero) → Phase 3 candidate
    """
    if rank_set is None:
        rank_set = RANK_SET

    n = len(ordering)
    if n == 0:
        return {}

    # Build tier list from rank set (descending, non-zero)
    tiers = sorted([r for r in rank_set if 0 < r <= max_rank], reverse=True)
    if not tiers:
        tiers = [1]

    floor_rank = tiers[-1]
    top_rank = tiers[0]
    moderate_rank = tiers[1] if len(tiers) > 1 else top_rank
    low_p1_rank = tiers[2] if len(tiers) > 2 else moderate_rank

    if phase1_magnitudes:
        p1_vals = list(phase1_magnitudes.values())
        p1_median = float(np.median(p1_vals))
    else:
        p1_median = None

    allocation = {}
    for i, name in enumerate(ordering):
        deg = degradation_scores.get(name, 0.0)

        # Phase 3 sentinel: confirmed freeze
        if deg < -900:
            allocation[name] = 0
            continue

        if i == 0:
            allocation[name] = top_rank
        elif i < n - 1:
            if phase1_magnitudes and p1_median is not None:
                p1_mag = phase1_magnitudes.get(name, p1_median)
                allocation[name] = (
                    low_p1_rank if p1_mag < p1_median else moderate_rank
                )
            else:
                allocation[name] = moderate_rank
        else:
            allocation[name] = floor_rank

    return allocation


def _assign_uniform_ranks(
    layer_names: List[str],
    rank: int = 4,
) -> Dict[str, int]:
    return {name: rank for name in layer_names}


# ── Strategy selection ─────────────────────────────────────────────

def select_strategy(
    pilot_result,
    baseline_rank: int = 4,
    max_rank: int = 8,
    layer_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    verbose: bool = True,
) -> StrategyRecommendation:
    """Classify a PilotResult into LWR or uniform, with allocation."""
    degradation_scores = {
        r.layer_name: r.mean for r in pilot_result.phase2_results
    }
    ordering = pilot_result.sensitivity_ordering

    if layer_shapes is None:
        layer_shapes = {
            r.layer_name: r.shape for r in pilot_result.phase2_results
        }

    phase1_magnitudes = None
    if hasattr(pilot_result, 'phase1_results') and pilot_result.phase1_results:
        phase1_magnitudes = {
            r.layer_name: r.mean for r in pilot_result.phase1_results
        }

    return _decide(
        degradation_scores=degradation_scores,
        ordering=ordering,
        layer_shapes=layer_shapes,
        baseline_rank=baseline_rank,
        max_rank=max_rank,
        verbose=verbose,
        phase1_magnitudes=phase1_magnitudes,
    )


def select_strategy_from_dict(
    degradation_scores: Dict[str, float],
    layer_shapes: Optional[Dict[str, Tuple[int, int]]] = None,
    baseline_rank: int = 4,
    max_rank: int = 8,
    verbose: bool = True,
    phase1_magnitudes: Optional[Dict[str, float]] = None,
    rank_set: Optional[List[int]] = None,
) -> StrategyRecommendation:
    """Classify from raw degradation scores."""
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
        phase1_magnitudes=phase1_magnitudes,
        rank_set=rank_set,
    )


def _decide(
    degradation_scores: Dict[str, float],
    ordering: List[str],
    layer_shapes: Dict[str, Tuple[int, int]],
    baseline_rank: int,
    max_rank: int,
    verbose: bool,
    phase1_magnitudes: Optional[Dict[str, float]] = None,
    rank_set: Optional[List[int]] = None,
) -> StrategyRecommendation:
    """Core decision logic."""
    if rank_set is None:
        rank_set = RANK_SET

    values = np.array(list(degradation_scores.values()))
    spread = float(np.max(values) - np.min(values))
    mean_abs = float(np.abs(np.mean(values)))
    std = float(np.std(values))
    cv = 0.0 if mean_abs < 1e-8 else std / mean_abs

    # Derive floor rank from rank set
    nonzero = sorted([r for r in rank_set if r > 0])
    floor_rank = nonzero[0] if nonzero else 1

    if spread < SPREAD_MIN:
        strategy = "heterogeneous"
        confidence = "high"
        rank_alloc = _assign_uniform_ranks(list(degradation_scores.keys()),
                                           rank=baseline_rank)
        finding = (
            f"Phase 2 degradation spread is negligible ({spread:.4f}). "
            f"Uniform r={baseline_rank} is sufficient."
        )
    elif cv > CV_HIGH_CONFIDENCE:
        strategy = "lwr"
        confidence = "high"
        rank_alloc = _assign_lwr_ranks(ordering, degradation_scores,
                                       max_rank=max_rank,
                                       phase1_magnitudes=phase1_magnitudes, rank_set=rank_set)
        most, least = ordering[0], ordering[-1]
        finding = (
            f"Large sensitivity gap (CV={cv:.2f}): {most} degrades "
            f"{degradation_scores[most]:+.4f} vs {least} at "
            f"{degradation_scores[least]:+.4f}. LWR recommended."
        )
    elif cv > CV_THRESHOLD:
        strategy = "lwr"
        confidence = "moderate"
        rank_alloc = _assign_lwr_ranks(ordering, degradation_scores,
                                       max_rank=max_rank,
                                       phase1_magnitudes=phase1_magnitudes, rank_set=rank_set)
        most, least = ordering[0], ordering[-1]
        finding = (
            f"Moderate sensitivity gap (CV={cv:.2f}). LWR likely "
            f"beneficial over uniform rank."
        )
    else:
        strategy = "heterogeneous"
        confidence = "moderate" if cv > CV_THRESHOLD * 0.5 else "high"
        rank_alloc = _assign_lwr_ranks(ordering, degradation_scores,
                                       max_rank=max_rank,
                                       phase1_magnitudes=phase1_magnitudes, rank_set=rank_set)
        finding = (
            f"Low sensitivity gap (CV={cv:.2f}). Ordering is noise-level; "
            f"heterogeneous rank provides diversity benefit over uniform."
        )

    rank_alloc_shapes = {}
    for name, rank in rank_alloc.items():
        if name in layer_shapes:
            rank_alloc_shapes[layer_shapes[name]] = rank

    total_budget = sum(rank_alloc.values())

    rec = StrategyRecommendation(
        strategy=strategy, confidence=confidence,
        rank_allocation=rank_alloc,
        rank_allocation_shapes=rank_alloc_shapes,
        total_budget=total_budget,
        cv=cv, spread=spread,
        degradation_scores=degradation_scores,
        finding=finding,
    )

    if verbose:
        print(rec.summary())

    return rec
