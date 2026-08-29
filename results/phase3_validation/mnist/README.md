# Phase 3 Validation — MNIST Output Layer

## Experiment
Head-to-head: rank 0 (freeze) vs rank 1 (minimum perturbation) on the output layer.
Other layers at their Phase-2-derived ranks. 5 seeds.

## Results
| Seed | Rank 0 acc. | Rank 1 acc. | Winner |
|---|---|---|---|
| 1 | 64.1% | 53.1% | Rank 0 |
| 2 | 63.5% | 52.7% | Rank 0 |
| 3 | 62.8% | 52.3% | Rank 0 |
| 4 | 62.1% | 51.9% | Rank 0 |
| 5 | 62.0% | 51.9% | Rank 0 |

Mean: 62.9% vs 52.4%. Rank 0 wins 5/5 seeds (+10.5pp).

## Inference
Freezing the MNIST output layer actively improves performance. The output layer (1.3% of parameters, shape 10×256) contributes directional noise ("loudly wrong" pattern) — its perturbations hurt the ES gradient estimate more than they help. Rank 0 removes this noise source.

