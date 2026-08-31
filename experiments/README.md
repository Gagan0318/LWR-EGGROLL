# Experiments

Each script corresponds to one or more sections of Chapter 4 in the dissertation. All scripts write results to `results/` and skip seeds that already have a JSON file, making them safely resumable. Hyperparameters are defined within each script.

## Supervised Learning — MNIST Family

| Script | Purpose |
|--------|---------|
| `compare_4_methods_mnist.py` | Core MNIST experiment: four-method baseline (Backprop, OpenAI-ES, Sep-CMA-ES, EGGROLL), LWR allocation comparisons, rank sweeps, budget-matched runs, and sensitivity pilot phases. The main workhorse script — most MNIST results come from here. |
| `four_method_cross_dataset.py` | Runs the four-method baseline comparison across Fashion-MNIST, KMNIST, and EMNIST-Digits. |
| `emnist_full_suite.py` | Full LWR suite on EMNIST-Digits: pilot, headline LWR vs vanilla, reversed allocation, wall-clock budget, and EMNIST-Letters generalisation test. |
| `run_3phase_pilot_all_datasets.py` | Runs the three-phase sensitivity pilot across all four MNIST-family datasets in sequence. Produces the pilot allocation JSONs. |
| `variance_rank_sweep.py` | σ × rank interaction grid: sweeps sigma ∈ {0.01, 0.03, 0.05, 0.1, 0.3} against rank ∈ {1, 2, 4, 8, 12, 16, 24, 32}. |
| `wall_budget_sweep.py` | Wall-clock budget experiment: compares LWR and vanilla EGGROLL under fixed time budgets (60s, 120s, 300s). |
| `tight_budget_sweep.py` | Tight-budget variant of the wall-clock experiment with additional LWR configurations. |
| `wide_hidden_ablation_v2.py` | Wide-hidden architecture [784, 512, 1024, 10] ablation. Tests whether sensitivity ordering holds when the hidden layer has more parameters than the input layer. |
| `moving_hidden_layer.py` | Architecture variation: tests sensitivity ordering across standard, narrow, deep, and tapered MLP architectures on MNIST. |
| `correctness_test.py` | Verifies that LWR-EGGROLL with uniform rank_spec=4 is bit-identical to vanilla EGGROLL r=4. |
| `phase3_multiseed_validation.py` | Phase 3 validation on MNIST: rank 0 vs rank 1 head-to-head on the output layer, 5 seeds. |
| `lwr_eggroll.py` | Standalone LWR-EGGROLL training script used during early development. |

## Stochastic RL — Gymnasium

| Script | Purpose |
|--------|---------|
| `cartpole_lwr.py` | CartPole-v1 experiment: four-method comparison. CartPole saturates at 500 ± 0 for all methods (ceiling effect). |
| `lunarlander_lwr.py` | LunarLander-v3 on symmetric architecture [8, 64, 64, 4]. Initial ES comparison — produced near-zero ES fitness, motivating the tuned follow-up. |
| `lunarlander_symmetric_tuned.py` | Re-runs the symmetric LunarLander with 8 eval episodes (up from 5) and additional LWR configurations to separate fitness noise from ES limitations. |
| `lunarlander_tapered.py` | LunarLander on tapered architecture [8, 256, 64, 4] with live sensitivity pilot, 5 seeds, and REINFORCE baseline. Source of the capped LWR (4,2,1) headline result. |
| `lunarlander_tapered_v2.py` | Extended capped-budget allocations on the tapered architecture. Confirms that lower total rank is consistently better on stochastic RL with the floor-rank rule. |

## Deterministic RL — Brax

| Script | Purpose |
|--------|---------|
| `brax_ant_experiment.ipynb` | Brax Ant locomotion (Colab notebook). Runs the full 10-method × 3-seed comparison plus the dual-metric pilot. Designed for GPU execution on Google Colab — see `notebooks/` for the runnable version. |
| `brax_phase3_multiseed.py` | Phase 3 validation on Brax Ant: rank 0 vs rank 1 head-to-head on the hidden layers, 3 seeds. |
