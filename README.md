# LWR-EGGROLL

**Layer-Wise Rank EGGROLL** — a novel extension of the EGGROLL low-rank evolution strategy that replaces uniform perturbation rank with per-layer rank allocation derived from a three-phase sensitivity pilot.

## Background: Evolution Strategies and the Rank Problem

Evolution Strategies (ES) optimise neural networks without computing gradients. Instead of backpropagation, ES evaluates a population of randomly perturbed copies of the network and updates the weights in the direction that improves fitness. OpenAI-ES (Salimans et al., 2017) scaled this idea to modern networks by using full-rank Gaussian noise to perturb entire parameter vectors — but the cost scales linearly with the number of parameters, making it expensive for large networks.

**EGGROLL** ([Sarkar et al., 2026](https://github.com/ESHyperscale/HyperscaleES)) addressed this by replacing full-rank noise with structured low-rank perturbations. Instead of perturbing every dimension of a weight matrix, EGGROLL decomposes each perturbation as W̃ = W + (σ/√r) · U · Vᵀ, where U and V are thin random matrices and r is the perturbation rank. This dramatically reduces the sampling cost while preserving the ES gradient signal — a rank-4 perturbation of a 256×784 weight matrix explores a 4-dimensional subspace instead of the full 200,704-dimensional space, cutting the per-generation cost proportionally.

**The limitation:** EGGROLL applies the same rank r to every layer in the network. But not all layers benefit equally from exploration. Some layers (like the input layer in supervised learning) are highly sensitive to perturbation and need higher rank to capture useful gradient directions. Others (like the output layer) may actively hurt the search by injecting noise into the gradient estimate — these layers would be better off with minimal rank or frozen entirely. A uniform rank wastes budget on insensitive layers and starves sensitive ones.

## LWR-EGGROLL: The Solution

LWR-EGGROLL replaces EGGROLL's uniform rank with a per-layer rank allocation determined by a three-phase sensitivity pilot that runs automatically before training begins:

- **Phase 1 (Magnitude):** Measures how much each layer's perturbation affects fitness variance using a shared checkpoint, providing a refinement signal for middle-tier layers.
- **Phase 2 (Causal Ablation):** The primary ordering mechanism. Drops each layer's rank from the baseline to rank 1 while holding all other layers constant, measuring the causal effect on performance. This reveals which layers genuinely help vs. hurt when perturbed.
- **Phase 3 (Freeze Decision):** For the least-sensitive layer, runs a head-to-head between rank 0 (frozen) and rank 1 (minimal perturbation) to determine whether freezing is safe.

The pilot outputs a rank allocation like (8, 4, 0) — rank 8 on the input layer, rank 4 on hidden, rank 0 (frozen) on output — tailored to the specific network and task with no manual tuning.

## Key Results

- **Supervised (MNIST family):** LWR-EGGROLL (8,4,0) outperforms vanilla EGGROLL r=4 by +1.4 to +6.3 percentage points across four datasets, with consistent input > hidden > output sensitivity ordering.
- **Stochastic RL (LunarLander):** Capped LWR (2,4,1) at budget 7 beats vanilla r=4 at budget 12 by +9.2 reward points.
- **Deterministic RL (Brax Ant):** LWR (4,2,0) under the best-fitness metric achieves mean best fitness of 2121, two orders of magnitude above uniform-rank baselines.

## Repository Structure

```
lwr_eggroll/          Core package — sensitivity pilot, strategy selector
experiments/          Experiment scripts (one per Chapter 4 section)
results/              Per-seed JSON results and figures, organised by environment
notebooks/            Colab notebook for Brax Ant (GPU-dependent experiments)
Setup/                Environment setup guide and automated setup script
references/           Links to upstream dependencies (HyperscaleES)
```

Utility files:
- `results/dump_results.py` — prints a per-seed summary of all results (also saved as `results/all_results.txt`)
- `environment.yml` / `requirements.txt` — conda and pip dependency specs
- `pyproject.toml` — package metadata for `pip install -e .`

## Setup

Full environment setup instructions are in [Setup/SETTING_UP.md](Setup/SETTING_UP.md). In brief:

1. Create the conda environment: `conda env create -f environment.yml`
2. Activate it: `conda activate eggroll`
3. Install the LWR-EGGROLL package in editable mode: `pip install -e .`
4. Verify: `python -c "from lwr_eggroll import adaptive_sensitivity_pilot; print('OK')"`

After setup, you will have JAX, evosax, HyperscaleES, and all dependencies installed. You can then run any experiment script directly (e.g. `python experiments/compare_4_methods_mnist.py`). Existing per-seed result JSONs are skipped on re-run, so experiments are safely resumable.

For Brax Ant experiments that require GPU, see `notebooks/brax_ant_colab.ipynb` which is designed to run on Google Colab.

## Reproducing Results

All result JSONs are committed to the repository under `results/`. Each experiment script writes to its corresponding results subfolder and skips seeds that already have a JSON file. To verify results from scratch, delete the relevant JSONs and re-run the script. See individual README files in each results subfolder for experiment details and key findings.

## Built On

This project extends [HyperscaleES (EGGROLL)](https://github.com/ESHyperscale/HyperscaleES) by Sarkar et al. (2026). The base EGGROLL implementation is installed as a dependency — see [Setup/SETTING_UP.md](Setup/SETTING_UP.md) for details.

## Citation

This repository accompanies the MSc dissertation *"LWR-EGGROLL: Layer-Wise Rank Allocation for Low-Rank Evolution Strategies"* by Gagan Deep Singh, University of Birmingham, 2026.
