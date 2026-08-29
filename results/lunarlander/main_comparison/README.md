# LunarLander — Main Comparison

## Overview
Initial comparison of ES methods vs REINFORCE on the symmetric architecture [8, 256, 256, 4].

## Setup
N = 256 | σ = 0.05 | α = 0.01 | Adam | 300 generations | 5 eval episodes per candidate.

## Results
ES methods (vanilla EGGROLL and LWR) failed: mean fitness ≈ 10, variance ≈ 240.
REINFORCE achieved 309.7 ± 7.2.

## Inference
The fitness signal from 5 stochastic episodes at N=256 is too noisy for ES gradient estimation on the symmetric architecture. This negative result motivated the tapered architecture experiments.

