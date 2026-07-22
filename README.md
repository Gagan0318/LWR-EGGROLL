# EGGROLL Dissertation Codebase

Comparative empirical benchmarking of EGGROLL (Sarkar et al., 2026) against
classical Evolution Strategies (OpenAI-ES, CMA-ES / Sep-CMA-ES, PGPE) on
reinforcement learning environments, with an empirical rank-sensitivity study.

**Author:** Gagan Deep Singh
**Supervisor:** Prof. Per Kristian Lehre
**Programme:** MSc Data Science, University of Birmingham
**Submission:** 1 September 2026

## Directory structure

- `experiments/` — self-contained Python scripts, one per experiment. Each
  produces a JSON result file in `results/`.
- `scripts/` — reusable utilities: plotting, sweep launchers, data prep.
- `notebooks/` — exploratory analysis and prototyping. Not experiments proper.
- `configs/` — YAML/JSON hyperparameter configs (when experiments get complex).
- `results/` — raw JSON outputs from experiment scripts. Regeneratable, not in git.
- `figures/` — final PNG/PDF plots for the dissertation. Regeneratable, not in git.
- `logs/` — text notes, findings summaries, and debug logs.

## Environment

Reproducible via `requirements.txt` and `environment.yml`:

```bash
conda create -n eggroll python=3.13 -y
conda activate eggroll
pip install -r requirements.txt
```

## Workflow

1. Prototype in `notebooks/`.
2. Move working code to a script in `experiments/`.
3. Run from project root: `python experiments/<name>.py`.
4. Results save to `results/<name>.json`.
5. Load and plot from a notebook in `notebooks/` or a script in `scripts/`.
6. Commit script + findings to git; results and figures stay local.

## Findings log

See `logs/findings.md` for a running log of experimental results and
interpretations.
