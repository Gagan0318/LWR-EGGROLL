# LunarLander — Symmetric Tuned

## Architecture
[8, 256, 256, 4] with tuned hyperparameters.

## Setup
N = 256 | σ = 0.05 | α = 0.01 | Adam | 300 generations.

## Results
Forward and reversed capped allocations both score ~282 reward, outperforming uniform r=4 at 272. CV on Phase 2 scores: 0.75 (below 0.8 threshold).

## Inference
Ordering signal is buried in environmental noise (CV < 0.8), but rank diversity still outperforms uniformity — different ranks introduce structural diversity that acts as implicit regularisation. The specific ordering matters less than having heterogeneous ranks.

