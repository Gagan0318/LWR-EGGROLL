# Notebooks

### `brax_ant_colab.ipynb`

Google Colab notebook for running the Brax Ant deterministic RL experiment. Brax Ant requires GPU acceleration (JAX + Brax) and was run on Colab Pro (L4/A100/T4 GPUs).

The notebook has two run modes controlled by the `RUN_MODE` variable:
- **"reuse"** (default) — loads the committed result JSONs from the repository. The pilot short-circuits on the existing `pilot_results.json` and method cells skip if their result file exists. This mode proves the notebook runs end-to-end and reproduces the reported results.
- **"fresh"** — regenerates everything from scratch into a separate `results/brax_ant_rerun/` folder, never touching the committed results. This mode allows independent verification of result credibility.

The notebook bootstraps the environment by unpacking a local zip of the repository uploaded to Google Drive. See the dissertation appendix for full reproducibility infrastructure documentation.
