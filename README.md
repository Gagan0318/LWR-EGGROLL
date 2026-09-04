# LWR-EGGROLL

**Layer-Wise Rank EGGROLL** — a novel extension of the EGGROLL low-rank evolution strategy that replaces uniform perturbation rank with per-layer rank allocation derived from a three-phase sensitivity pilot.

Standard EGGROLL applies the same rank `r` to every layer. LWR-EGGROLL runs a short diagnostic pilot to measure each layer's sensitivity, then assigns higher rank to sensitive layers and lower rank (or rank 0, freezing the layer entirely) to insensitive ones. The result is better accuracy and reward at equal or reduced computational budget across supervised learning and reinforcement learning environments.

## Key Results

- **Supervised (MNIST family):** LWR-EGGROLL (8,4,0) outperforms vanilla EGGROLL r=4 by +1.4 to +6.3 percentage points across four datasets, with consistent input > hidden > output sensitivity ordering.
- **Stochastic RL (LunarLander):** Capped LWR (4,2,1) at budget 7 beats vanilla r=4 at budget 12 by +9.2 reward points.
- **Deterministic RL (Brax Ant):** LWR (4,2,0) under the best-fitness metric achieves mean best fitness of 2121, two orders of magnitude above uniform-rank baselines.

## Repository Structure

```
lwr_eggroll/          Core package — sensitivity pilot, strategy selector
experiments/          Experiment scripts (one per Chapter 4 section)
results/              Per-seed JSON results and figures, organised by environment
notebooks/            Colab notebook for Brax Ant (GPU-dependent experiments)
```

Utility files at root level:

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

This repository accompanies the MSc dissertation *"LWR-EGGROLL: Layer-Wise Rank Allocation for Low-Rank Evolution Strategies"* by Gagan Deep Singh, supervised by Prof. Per Kristian Lehre, University of Birmingham, September 2026.
