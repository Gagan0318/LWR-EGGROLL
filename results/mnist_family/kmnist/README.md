# KMNIST — Supervised Learning Experiment

## Dataset
Kuzushiji-MNIST. 60,000 train / 10,000 test. 28×28 greyscale, 10 Japanese character classes.

## Architecture & Setup
Same as MNIST: [784, 256, 256, 256, 10], N=2,048, σ=0.05, α=0.01, 3 seeds.

## Sensitivity Pilot
Ordering identical to MNIST: input >> hidden > output. Allocation: (8, 4, 0).

## Results
| Method | Test accuracy ± std |
|---|---|
| LWR aligned (8,4,0) | 52.1 ± 1.10 |
| Vanilla EGGROLL r=4 | 45.79 ± 1.27 |
| LWR reversed (0,4,8) | 43.10 ± 0.37 |

LWR advantage: +6.3pp — the largest in the MNIST family.

## Inference
KMNIST has the most complex class boundaries among MNIST-family datasets, making it most sensitive to perturbation quality. LWR's advantage is largest here because misallocated rank causes the most damage on difficult tasks.

