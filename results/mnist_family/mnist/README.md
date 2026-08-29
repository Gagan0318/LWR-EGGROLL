# MNIST — Supervised Learning Experiment

## Dataset
MNIST handwritten digits. 60,000 train / 10,000 test. 28×28 greyscale.

## Architecture
Standard MLP: [784, 256, 256, 256, 10] (~335K parameters).

## Setup
N = 2,048 | σ = 0.05 | α = 0.01 | SGD | 5,000 gens or 300s wall-clock cap | 3 seeds.

## Sensitivity Pilot
Phase 2 ordering: input >> hidden > output. Derived allocation: (8, 4, 0). Budget: 12.
Phase 3: rank 0 wins 5/5 seeds on output layer (+10.5pp mean advantage).

## Results

| Method | Test accuracy ± std |
|---|---|
| LWR aligned (8,4,0) | 84.08 ± 0.76 |
| Vanilla EGGROLL r=4 | 82.72 ± 0.10 |
| LWR reversed (0,4,8) | 77.67 ± 1.17 |

LWR advantage: +1.4pp over vanilla at matched budget.
Half-budget LWR (4,2,0) at budget 6 also outperforms vanilla at budget 12.

## Four-Method Baseline
| Method | Accuracy |
|---|---|
| Backprop (Adam) | 98.07 ± 0.02 |
| OpenAI-ES | 90.95 ± 0.09 |
| EGGROLL r=4 | 82.72 ± 0.10 |
| Sep-CMA-ES | 75.51 ± 1.19 |

## Key Inferences
1. Phase 2 ordering is consistent across all four MNIST-family datasets — architecture-dependent, not dataset-dependent.
2. Reversed allocation (0,4,8) is worst → the *specific* ordering matters, not just heterogeneity.
3. Output layer freezing (rank 0) actively helps — the "loudly wrong" pattern: high Phase 1 variance but negative Phase 2 causal effect.
4. Wall-clock advantage compounds: LWR (8,4,0) completes more generations due to frozen output layer requiring no sampling.

