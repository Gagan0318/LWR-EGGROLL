# Setup

This folder contains the environment setup guide and an automated setup script for the LWR-EGGROLL project.

### `SETTING_UP.md`

Step-by-step instructions for setting up the development environment from scratch. Covers two installation paths (CPU-only and GPU with CUDA 12), JAX installation, HyperscaleES installation pinned to the exact commit used in this project, the LWR-EGGROLL package install, and the Google Colab bootstrap process for Brax Ant experiments. Read this if any step in the automated script fails or if you want to understand what each step does.

### `setup.sh`

Automated script that wraps the steps in SETTING_UP.md. Run from the repository root:

bash Setup/setup.sh # CPU install, JAX 0.11.0
bash Setup/setup.sh --gpu # GPU (CUDA 12), JAX 0.11.0
bash Setup/setup.sh --gpu --brax # GPU, JAX 0.9.0.1 (Brax Ant reproducibility)


After either method, verify the install with:

conda activate eggroll
python -c "from lwr_eggroll.adaptive_sensitivity_pilot import AdaptiveSensitivityPilot; print('LWR-EGGROLL ready')"

