# Fashion-MNIST — Supervised Learning Experiment

## Dataset
Fashion-MNIST. 60,000 train / 10,000 test. 28×28 greyscale, 10 clothing categories.

## Architecture & Setup
Same as MNIST: [784, 256, 256, 256, 10], N=2,048, σ=0.05, α=0.01, 3 seeds.

## Sensitivity Pilot
Ordering identical to MNIST: input >> hidden > output. Allocation: (8, 4, 0).

## Results
| Method | Test accuracy ± std |
|---|---|
| LWR aligned (8,4,0) | 72.64 ± 0.38 |
| Vanilla EGGROLL r=4 | 71.36 ± 0.70 |
| LWR reversed (0,4,8) | 69.87 ± 0.78 |

LWR advantage: +1.3pp.

## Inference
Cross-dataset transfer confirmed — same architecture produces same ordering. Pilot cost amortisable across datasets.

