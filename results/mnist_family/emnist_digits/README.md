# EMNIST-Digits — Supervised Learning Experiment

## Dataset
EMNIST-Digits. 240,000 train / 40,000 test. 28×28 greyscale, 10 digit classes.

## Architecture & Setup
Same as MNIST: [784, 256, 256, 256, 10], N=2,048, σ=0.05, α=0.01, 3 seeds.

## Sensitivity Pilot
Ordering identical to MNIST: input >> hidden > output. Allocation: (8, 4, 0).

## Results
| Method | Test accuracy ± std |
|---|---|
| LWR aligned (8,4,0) | 87.38 ± 0.72 |
| Vanilla EGGROLL r=4 | 85.76 ± 1.11 |
| LWR reversed (0,4,8) | 81.77 ± 1.41 |

LWR advantage: +1.6pp.

## Inference
Largest dataset in the MNIST family (240K train). Consistent ordering and consistent LWR advantage confirms scalability to larger training sets.

