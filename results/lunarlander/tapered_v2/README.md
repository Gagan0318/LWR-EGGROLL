# LunarLander — Tapered v2

## Architecture
[8, 256, 64, 4] — same as tapered, extended investigation.

## Setup
N = 256 | σ = 0.05 | α = 0.01 | Adam | 300 generations.

## Inference
Confirmed that less rank is consistently better on stochastic RL when the floor-rank rule is applied. eggroll_r1 at budget 3 outperformed lwr_4_8_0 at budget 12 — rank allocation must respect the noise floor.

