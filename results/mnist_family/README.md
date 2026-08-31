# MNIST Family — Supervised Learning Experiments

Four 28x28 greyscale image classification datasets sharing the same MLP architecture [784, 256, 256, 256, 10] (~335K parameters). The three-phase sensitivity pilot produces identical layer ordering (input > hidden > output) and allocation (8, 4, 0) across all four datasets, confirming that sensitivity is architecture-dependent rather than dataset-dependent.

## Datasets

- **MNIST** — handwritten digits, 60K train / 10K test. Primary benchmark with the most extensive experiment coverage (rank sweeps, sigma interaction, wall-clock budget, architecture variations, wide-hidden ablation).
- **Fashion-MNIST** — clothing categories, 60K / 10K. Cross-dataset transfer validation.
- **KMNIST** — Japanese Kuzushiji characters, 60K / 10K. Largest LWR advantage (+6.3pp).
- **EMNIST-Digits** — digits, 240K / 40K. Largest training set; includes EMNIST-Letters generalisation test.

## Experiment Types per Dataset

Each dataset folder contains some or all of: `pilot/` (three-phase sensitivity pilot), `headline_comparison/` (four-method baseline + LWR vs vanilla), `reversed_allocation/` (allocation (0,4,8) as negative control), `transfer/` (cross-dataset allocation transfer), `wall_clock_budget/` (fixed-time budget comparisons), and dataset-specific investigations. See individual dataset READMEs for details.
