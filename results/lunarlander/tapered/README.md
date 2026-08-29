# LunarLander — Tapered Architecture

## Architecture
[8, 256, 64, 4] (~18.7K parameters).

## Setup
N = 256 | σ = 0.05 | α = 0.01 | Adam | 300 generations | 5 seeds.

## Sensitivity Pilot
Ordering: hidden > input > output. Rank set: {1, 2, 4} (rank 0 excluded for stochastic RL).

## Results
| Method | Rank (in,hid,out) | Budget | Mean reward ± std |
|---|---|---|---|
| lwr_capped (4,2,1) | (4,2,1) | 7 | 281.3 ± 5.2 |
| lwr_pilot_uncapped (8,1,4) | (8,1,4) | 13 | 275.2 ± 8.9 |
| eggroll_r4 | (4,4,4) | 12 | 272.1 ± 6.6 |
| eggroll_r1 | (1,1,1) | 3 | 175.8 ± 159.0 |

## Key Inferences
1. Capped LWR (4,2,1) at budget 7 beats vanilla r=4 at budget 12 by +9.2 points with tighter variance. Best efficiency result on stochastic RL.
2. Floor-rank rule (minimum rank 1) is essential — eggroll_r1 collapses, showing too little total rank is also harmful.
3. Tapered architecture provides enough structural asymmetry to overcome environmental noise floor, unlike the symmetric architecture.

