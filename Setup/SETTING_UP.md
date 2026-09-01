# Setting Up LWR-EGGROLL

Copy-paste setup for running the experiments. Two paths are given:
**CPU-only** (sufficient for MNIST family, CartPole, and LunarLander)
and **GPU (CUDA 12)** (required for the Brax Ant experiments).

Assumes a fresh machine with `git`, `python>=3.10`, and either `conda`
or `venv`.

## Quick setup (optional)

If you'd rather not run the steps below one at a time, the repo ships a
`setup.sh` that wraps them. Clone first (Section 1), then from the repo root:

```bash
bash setup.sh              # CPU install, JAX 0.11.0
bash setup.sh --gpu        # GPU (CUDA 12), JAX 0.11.0
bash setup.sh --gpu --brax # GPU, JAX 0.9.0.1 (to reproduce the Brax Ant results)
bash setup.sh --gpu --brax --zip   # also build gxs523.zip for the Colab route
```

The sections below explain each step the script runs; read them if a step
fails or you want to adapt it. The script is a convenience, not a replacement
for understanding the setup.

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

> **Brax Ant reproducibility note:** the committed Brax Ant results in
> `results/brax_ant/` were produced with **JAX 0.9.0.1**, not 0.11.0. The
> reproducible notebook `experiments/brax_ant_experiment.ipynb` pins
> 0.9.0.1 automatically in its own setup cell, so run that notebook as-is
> to reproduce those numbers — you do not need to change the JAX version
> above for any of the other experiments.

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

## 7. Running the Brax Ant notebook

`experiments/brax_ant_experiment.ipynb` runs in two environments, auto-detected.

### Local / university Jupyter (or VS Code)

Open the notebook from inside the cloned repo. Paths resolve relative to the
repo root and the setup cell installs everything — no restart, no extra steps.
Skip the rest of this section.

### Google Colab (browser-only Drive)

Colab cannot reach your local machine, so the repo has to arrive on your
Google Drive first. You do this once, via the browser.

**Step 1 — build the upload zip (on your machine).** From the folder that
*contains* your clone, make a `.git`-free copy named `gxs523` and zip it. This
uses only Python's standard library, so nothing extra to install:

```bash
cd /path/to/parent-of-clone        # the folder containing your cloned repo
cp -r <your-clone-folder> gxs523   # e.g. cp -r uni-clean gxs523
rm -rf gxs523/.git                 # drop git history (keeps the zip small)
python -c "import shutil; shutil.make_archive('gxs523','zip','.','gxs523')"
rm -rf gxs523                      # remove the temp copy; keep gxs523.zip
```

**Where the zip ends up.** These commands `cd` to the folder *containing* your
clone, so `gxs523.zip` is created **beside** the repo folder, not inside it:

```
some-parent-folder/
├── gxs523/         <- your cloned repo (working copy, untouched)
└── gxs523.zip      <- the upload archive, OUTSIDE the repo
```

This is intentional: keeping the zip outside the repo means it never gets
committed into git, and the archive contains exactly one top-level `gxs523/`
folder (no nested `gxs523/gxs523/`). The zip is a throwaway transport file —
once it is on Drive and unpacked (Step 3), you can delete the local copy.

Optional check that the archive is structured correctly (expect `True`,
`False`, `31`):

```bash
python -c "import zipfile; z=zipfile.ZipFile('gxs523.zip'); n=z.namelist(); print(any(x=='gxs523/pyproject.toml' for x in n)); print(any(x.startswith('gxs523/gxs523/') for x in n)); print(sum('gxs523/results/brax_ant/' in x and x.endswith('.json') for x in n))"
```

**Step 2 — upload (browser).** Go to drive.google.com and upload `gxs523.zip`
to the **root of My Drive** (so it sits at `MyDrive/gxs523.zip`).

**Step 3 — unpack (in the notebook).** Run the notebook's first cell (Cell 0).
It mounts Drive and, if `MyDrive/gxs523` doesn't exist yet, extracts the zip
there. Re-running is safe. After that, run Cell 1 onward as normal — the
notebook's `DRIVE_REPO_ROOT` already points at `MyDrive/gxs523`, so there is
no path to edit and no access token stored anywhere.

### Run modes (both environments)

Set `RUN_MODE` in the notebook's config cell:

- `"reuse"` *(default)* — loads the committed result JSONs (methods skip, the
  pilot short-circuits). Fast check that the notebook runs end to end.
- `"fresh"` — regenerates everything from zero into a separate
  `results/brax_ant_rerun/` folder, leaving the committed results untouched.
  Runs the full pilot and trains every method (hours, GPU required).
