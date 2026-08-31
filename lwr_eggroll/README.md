# lwr_eggroll — Core Package

This directory contains the core modules that implement the LWR-EGGROLL framework. All experiment scripts import from this package. Install it with `pip install -e .` from the repository root.

## Modules

### `adaptive_sensitivity_pilot.py`
The three-phase sensitivity pilot for supervised learning environments. Runs Phase 1 (perturbation magnitude via shared-checkpoint elevation), Phase 2 (causal ablation for layer ordering), and Phase 3 (binary freeze decision via rank 0 vs rank 1 head-to-head). Outputs a rank allocation dictionary mapping each layer to its assigned rank. Also implements the flat-basin metric switch: if best-so-far fitness separates from mean fitness by more than 2.33 standard deviations, the pilot switches from mean to best-fitness as its evaluation metric.

### `rl_sensitivity_pilot.py`
Adaptation of the sensitivity pilot for reinforcement learning environments. Handles the differences from supervised learning: episodic fitness evaluation, stochastic reward signals, and the floor-rank rule (minimum rank 1 for stochastic environments, rank 0 excluded).

### `strategy_selector.py`
Analyses the Phase 2 degradation scores to determine the allocation strategy. Uses the coefficient of variation (CV) of degradation scores as the decision metric: CV > 1.5 triggers high-confidence LWR allocation, CV > 0.8 triggers moderate allocation, and CV < 0.8 indicates a heterogeneous regime where ordering signal is weak. Also implements the all-negative diagnostic branch: if all Phase 2 scores are negative (every layer's removal improves fitness), the environment has low effective dimensionality and the selector recommends uniform rank 1 fallback.

### `__init__.py`
Package initialisation. Exposes the public API for imports.
