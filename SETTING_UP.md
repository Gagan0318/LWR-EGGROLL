# Setting Up LWR-EGGROLL

Copy-paste setup for running the experiments. Two paths are given:
**CPU-only** (sufficient for MNIST family, CartPole, and LunarLander)
and **GPU (CUDA 12)** (required for the Brax Ant experiments).

Assumes a fresh machine with `git`, `python>=3.10`, and either `conda`
or `venv`.

## 1. Clone the repository

```bash
git clone https://git.cs.bham.ac.uk/projects-2025-26/gxs523.git
cd gxs523
```

(Or the public mirror: `git clone https://github.com/Gagan0318/LWR-EGGROLL.git`)

## 2. Create an environment

Conda:

```bash
conda create -n eggroll python=3.11 -y
conda activate eggroll
```

Or venv:

```bash
python -m venv .venv
source .venv/bin/activate
```

## 3. Install JAX

CPU-only:

```bash
pip install --upgrade "jax==0.11.0"
```

GPU (CUDA 12):

```bash
pip install --upgrade "jax[cuda12]==0.11.0"
```

> If the GPU install misbehaves, the exact package versions used in this
> project (including `jax-cuda12-plugin` and `jax-cuda12-pjrt`) are pinned
> in `requirements.txt` — run `pip install -r requirements.txt` for a
> byte-for-byte match.

## 4. Install HyperscaleES (base EGGROLL library)

Installed directly from the public repo, pinned to the exact commit used
in this project — nothing is vendored into this repo:

```bash
pip install "git+https://github.com/ESHyperscale/HyperscaleES.git@b77f7d6f91238fd575313e946b9cad21e0a74b32"
```

## 5. Install this package

From the repository root:

```bash
pip install -e .
```

For the RL experiments (Brax, gymnax, evosax):

```bash
pip install -e ".[rl]"
```

## 6. Verify

```bash
python -c "from lwr_eggroll.adaptive_sensitivity_pilot import AdaptiveSensitivityPilot; print('LWR-EGGROLL ready')"
```

## What runs where

| Experiment family | Hardware |
|-------------------|----------|
| MNIST family | CPU or GPU |
| CartPole | CPU or GPU |
| LunarLander | CPU or GPU |
| Brax Ant | GPU (CUDA 12) required |

Results are written to `results/<experiment_name>/` relative to the repo root.
