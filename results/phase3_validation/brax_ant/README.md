# Phase 3 Validation — Brax Ant Hidden Layers

## Experiment
Head-to-head: rank 0 (freeze) vs rank 1 (minimum perturbation) on the hidden layers.
Other layers at their Phase-2-derived ranks. 3 seeds.

## Results
| Seed | Rank 0 fitness | Rank 1 fitness | Winner |
|---|---|---|---|
| 1 | 13.8 | 31.1 | Rank 1 |
| 2 | 15.8 | 38.4 | Rank 1 |
| 3 | 14.2 | 33.4 | Rank 1 |

Mean: 14.6 vs 34.3. Rank 1 wins 3/3 seeds (+19.7 points).

## Inference
Freezing the Brax Ant hidden layers (93.1% of parameters) is catastrophic. With the majority of the network frozen at random initialisation, the small adaptive layers (input 6.9%) cannot compensate. Phase 3 correctly refuses the freeze — opposite outcome from MNIST, confirming Phase 3 is an unbiased diagnostic.

