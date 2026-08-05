# LWR-EGGROLL: Layer-Wise Rank EGGROLL

Sensitivity-informed per-layer rank allocation for low-rank evolution strategies.

LWR-EGGROLL extends EGGROLL (Sarkar et al., 2025) by replacing its uniform
perturbation rank with a per-layer rank vector derived from a two-phase
sensitivity pilot. The most sensitive layers receive higher rank; the least
sensitive can be frozen at rank zero. This improves gradient estimate quality
while reducing computational cost.

**Author:** Gagan Deep Singh
**Supervisor:** Prof. Per Kristian Lehre
**Programme:** MSc Data Science, University of Birmingham
**Submission:** 1 September 2026

## Key results

- **~11 pp accuracy gap** between sensitivity-aligned (8, 2, 0) and reversed (0, 2, 8) allocations on MNIST, with identical total rank budget
- **Cross-dataset transfer:** MNIST-derived allocation generalises to Fashion-MNIST, KMNIST, and EMNIST-Digits without re-piloting
- **Dual advantage:** better rank allocation + lower total rank budget = higher accuracy in less wall-clock time
- **Population × rank interaction** validated: LWR matters most at moderate population sizes where aggregate search subspace Nr is limited

## Directory structure

- `experiments/` — self-contained Python scripts, one per experiment. Each
  produces a JSON result file in `results/`.
- `scripts/` — reusable utilities: plotting, sweep launchers, data prep.
- `notebooks/` — exploratory analysis and prototyping.
- `results/` — raw JSON outputs from experiment scripts (gitignored).
- `figures/` — final PNG/PDF plots for the dissertation (gitignored).
- `logs/` — text notes, findings summaries, and debug logs.
- `HyperscaleES/` — git submodule linking the original EGGROLL library.

### Core files

| File | Purpose |
|---|---|
| `HyperscaleES/src/hyperscalees/noiser/lwr_eggroll.py` | LWR-EGGROLL noiser class |
| `experiments/compare_4_methods_mnist.py` | Four-method baseline + LWR training |
| `experiments/adaptive_sensitivity_pilot.py` | Adaptive pilot (allocation & binary inclusion modes) |
| `experiments/moving_hidden_layer.py` | Tapered architecture experiment |
| `experiments/position_rank_test.py` | Per-shape vs per-position rank resolution |

## Environment

```bash
conda create -n eggroll python=3.13 -y
conda activate eggroll
pip install -r requirements.txt
```

## Installation

HyperscaleES is included as a git submodule. After cloning this repository:

```bash
git clone --recurse-submodules https://github.com/Gagan0318/Dissertation---LWR-Eggroll.git
cd Dissertation---LWR-Eggroll
cd HyperscaleES
pip install "jax[cuda12]"
pip install -e .
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
cd HyperscaleES
pip install -e .
```

## Workflow

1. Prototype in `notebooks/`.
2. Move working code to a script in `experiments/`.
3. Run from project root: `python experiments/<name>.py`.
4. Results save to `results/<name>.json`.
5. Plot from `scripts/` or `notebooks/`.
6. Commit script + findings to git; results and figures stay local.

## Findings log

See `logs/findings.md` for a running log of all experimental results with
full reproducibility parameters.

---

## EGGROLL — Original Work

This project builds upon and extends **EGGROLL** (Evolution Guided GeneRal
Optimisation via Low-rank Learning), a low-rank evolution strategy for scaling
backpropagation-free optimisation to billion-parameter neural networks.

LWR-EGGROLL is an independent extension. The original EGGROLL algorithm,
codebase, and theoretical analysis are the work of the following authors:

> **Bidipta Sarkar**, Mattie Fellows, Juan Agustin Duque, Alistair Letcher,
> Antonio León Villares, Anya Sims, Clarisse Wibault, Dmitry Samsonov,
> Dylan Cope, Jarek Liesen, Kang Li, Lukas Seier, Theo Wolf, Uljad Berdica,
> Valentin Mohl, Alexander David Goldie, Aaron Courville, Karin Sevegnani,
> Shimon Whiteson, **Jakob Nicolaus Foerster**
>
> University of Oxford (FLAIR & WhiRL) · MILA — Québec AI Institute · NVIDIA AI Technology Center

| Resource | Link |
|---|---|
| Paper | [arXiv:2511.16652](https://arxiv.org/abs/2511.16652) |
| Project page | [eshyperscale.github.io](https://eshyperscale.github.io/) |
| HyperscaleES (JAX library) | [github.com/ESHyperscale/HyperscaleES](https://github.com/ESHyperscale/HyperscaleES) |
| nano-egg (int8 training) | [github.com/ESHyperscale/nano-egg](https://github.com/ESHyperscale/nano-egg) |
