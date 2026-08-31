# Results

All experimental results are stored as per-seed JSON files. Each JSON contains the full training history (fitness per generation), final metrics, and experiment configuration. Results are organised by environment family.

## Structure

```
results/
├── mnist_family/           Supervised learning (4 datasets)
│   ├── mnist/              MNIST — primary benchmark
│   ├── fashion_mnist/      Fashion-MNIST
│   ├── kmnist/             KMNIST (Kuzushiji-MNIST)
│   └── emnist_digits/      EMNIST-Digits
├── lunarlander/            Stochastic RL (4 sub-experiments)
│   ├── main_comparison/    Symmetric arch, initial comparison
│   ├── symmetric_tuned/    Symmetric arch, tuned hyperparameters
│   ├── tapered/            Tapered arch [8,256,64,4], 5 seeds
│   └── tapered_v2/         Tapered arch, extended allocations
├── brax_ant/               Deterministic RL — Brax Ant locomotion
├── phase3_validation/      Phase 3 freeze decision validation
│   ├── mnist/              Output layer: rank 0 vs rank 1 (5 seeds)
│   └── brax_ant/           Hidden layers: rank 0 vs rank 1 (3 seeds)
└── figures/                Cross-environment summary figures
```

## Naming Convention

Result files follow the pattern `{method}_seed{N}.json` or `seed{N}.json` within a method subfolder. Allocation labels use the format `lwr_{input}_{hidden}_{output}` — for example, `lwr_8_4_0` means rank 8 on the input layer, rank 4 on hidden, and rank 0 (frozen) on the output layer.

Each environment's subfolder has its own README with setup details, result tables, and key inferences.
