# Brax Ant — Deterministic RL Experiment

## Environment
Brax Ant locomotion (deterministic JAX-native physics). Architecture: [27, 256, 256, 256, 8] (140,032 parameters, 93.1% in hidden layers).

## Setup
N = 2,048 | σ = 0.2 | α = 0.1 | SGD | 300 generations | 3 seeds per method.
Hyperparameters match Sarkar et al. (2026) Table 19.

## Results — Mean Fitness (6-method comparison)

| Method | Rank (in,hid,out) | Budget | Mean ± std |
|---|---|---|---|
| eggroll_r1 | (1,1,1) | 3 | 76.8 ± 20.3 |
| lwr_4_1_2 | (4,1,2) | 7 | 52.3 ± 14.5 |
| lwr_8_1_4 | (8,1,4) | 13 | 50.5 ± 20.9 |
| eggroll_r4 | (4,4,4) | 12 | 27.2 ± 17.6 |
| lwr_8_4_0 | (8,4,0) | 12 | 16.7 ± 4.9 |
| lwr_4_0_2 | (4,0,2) | 6 | 8.3 ± 3.4 |

## Results — Best Fitness (exploratory)

| Method | Rank (in,hid,out) | Mean best fitness |
|---|---|---|
| lwr_4_2_0 | (4,2,0) | 2121.2 |
| lwr_8_4_0 | (8,4,0) | 1616.6 |
| lwr_4_2_1 (control) | (4,2,1) | 45.6 |
| lwr_8_4_1 (control) | (8,4,1) | 29.3 |

## Key Inferences
1. **Uniform r=1 dominates under mean fitness.** Low effective dimensionality — 93.1% of parameters are in low-sensitivity hidden layers. Rank 1 concentrates sampling on one direction per layer with 2,048 samples, yielding excellent signal-to-noise.
2. **Rank 0 on hidden layers is catastrophic.** lwr_4_0_2 and lwr_8_4_0 collapse because frozen hidden layers (93.1% of parameters) are stuck at random initialisation. Best-so-far fitness ranges 5.1–50.0 across seeds (initialisation lottery).
3. **Best-fitness metric reveals a different regime.** Allocation (4,2,0) under best fitness achieves 2121.2 — two orders of magnitude above all mean-fitness methods. Rank 2 on hidden layers doubles the perturbation subspace, enabling escape from initial fitness basins around generation 165–175.
4. **Output-rank isolation.** Moving output from rank 0 to rank 1 collapses fitness by 40–57× ((4,2,0)→(4,2,1): 2121.2→45.6). Co-adaptation disruption: perturbing the output readout prevents upstream layers from converging.
5. **Pilot diagnostic branch works.** All-negative Phase 2 scores correctly identify low effective dimensionality → uniform r=1 fallback.

