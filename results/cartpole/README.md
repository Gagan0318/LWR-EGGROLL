# CartPole — Stochastic RL Experiment

## Environment
CartPole-v1 (Gymnasium). Architecture: [4, 256, 256, 2].

## Setup
N = 256 | σ = 0.05 | α = 0.01 | Adam | 300 generations | 5 seeds per method.

## Sensitivity Pilot
Ordering: output > input > hidden. Allocation: (4, 0, 8).

## Results

| Method | Best Fitness |
|--------|-------------|
| OpenAI-ES | 500.0 ± 0.0 |
| EGGROLL r=4 | 500.0 ± 0.0 |
| LWR (4,0,8) | 500.0 ± 0.0 |
| REINFORCE | 500.0 ± 0.0 |

## Inference
CartPole saturates at the maximum reward of 500 for all methods — a ceiling effect that prevents any differentiation between approaches. This environment is too simple to reveal differences in rank allocation strategy.
