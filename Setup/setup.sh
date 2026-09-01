#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────
# setup.sh — one-command setup for LWR-EGGROLL.
#
# This is a convenience wrapper around the steps in SETTING_UP.md. The doc
# remains the source of truth; read it if any step here fails or if you want
# to understand what each step does.
#
# Usage:
#   bash setup.sh              # CPU install, JAX 0.11.0 (matches SETTING_UP.md)
#   bash setup.sh --gpu        # GPU (CUDA 12) install, JAX 0.11.0
#   bash setup.sh --gpu --brax # GPU install pinned to JAX 0.9.0.1 (Brax Ant repro)
#   bash setup.sh --zip        # also build gxs523.zip for the Colab/Brax route
#
# Flags combine, e.g.:  bash setup.sh --gpu --brax --zip
# Run from the repository root (the folder containing pyproject.toml).
# ─────────────────────────────────────────────────────────────────────
set -euo pipefail

GPU=0
BRAX=0
MAKE_ZIP=0
for arg in "$@"; do
  case "$arg" in
    --gpu)  GPU=1 ;;
    --brax) BRAX=1 ;;
    --zip)  MAKE_ZIP=1 ;;
    -h|--help)
      sed -n '2,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown flag: $arg (see --help)"; exit 1 ;;
  esac
done

# ── Sanity: must be at repo root ──────────────────────────────────────
if [[ ! -f pyproject.toml || ! -d lwr_eggroll ]]; then
  echo "ERROR: run this from the repository root (needs pyproject.toml and lwr_eggroll/)." >&2
  exit 1
fi

# Pick JAX version: Brax Ant results were produced on 0.9.0.1; everything
# else uses 0.11.0 (see SETTING_UP.md).
if [[ "$BRAX" -eq 1 ]]; then
  JAX_VER="0.9.0.1"
else
  JAX_VER="0.11.0"
fi

echo "=============================================="
echo " LWR-EGGROLL setup"
echo "   GPU:        $([[ $GPU -eq 1 ]] && echo yes || echo no)"
echo "   JAX:        $JAX_VER $([[ $BRAX -eq 1 ]] && echo '(Brax Ant repro)' || echo '(default)')"
echo "   build zip:  $([[ $MAKE_ZIP -eq 1 ]] && echo yes || echo no)"
echo "=============================================="

# ── 1. JAX ────────────────────────────────────────────────────────────
echo ">> Installing JAX $JAX_VER ..."
if [[ "$GPU" -eq 1 ]]; then
  pip install --upgrade "jax[cuda12]==${JAX_VER}"
else
  pip install --upgrade "jax==${JAX_VER}"
fi

# ── 2. HyperscaleES (base EGGROLL library, pinned commit) ─────────────
echo ">> Installing HyperscaleES (base EGGROLL library) ..."
pip install "git+https://github.com/ESHyperscale/HyperscaleES.git@b77f7d6f91238fd575313e946b9cad21e0a74b32"

# ── 3. This package ───────────────────────────────────────────────────
echo ">> Installing this package (lwr_eggroll) ..."
if [[ "$GPU" -eq 1 ]]; then
  pip install -e ".[rl]"
else
  pip install -e .
fi

# ── 4. Verify ─────────────────────────────────────────────────────────
echo ">> Verifying ..."
python -c "from lwr_eggroll.adaptive_sensitivity_pilot import AdaptiveSensitivityPilot; print('LWR-EGGROLL ready')"

# ── 5. (Optional) Build the Colab/Brax upload zip ─────────────────────
if [[ "$MAKE_ZIP" -eq 1 ]]; then
  echo ">> Building gxs523.zip for the Colab/Brax route ..."
  REPO_DIR="$(pwd)"
  REPO_NAME="$(basename "$REPO_DIR")"
  PARENT_DIR="$(dirname "$REPO_DIR")"
  (
    cd "$PARENT_DIR"
    rm -rf gxs523 gxs523.zip
    cp -r "$REPO_NAME" gxs523
    rm -rf gxs523/.git
    python -c "import shutil; shutil.make_archive('gxs523','zip','.','gxs523')"
    rm -rf gxs523
    COUNT=$(python -c "import zipfile; z=zipfile.ZipFile('gxs523.zip'); print(sum('results/brax_ant/' in n and n.endswith('.json') for n in z.namelist()))")
    echo ""
    echo "   Built: $PARENT_DIR/gxs523.zip"
    echo "   Brax result JSONs inside: $COUNT (expected 31)"
  )
  echo ""
  echo "   NEXT (manual, browser + Colab — a script cannot do these):"
  echo "     1. Upload gxs523.zip to the ROOT of My Drive at drive.google.com"
  echo "     2. Open experiments/brax_ant_experiment.ipynb in Colab"
  echo "     3. Run Cell 0 (unzips to MyDrive/gxs523), then Cell 1 onward"
fi

echo ""
echo "Setup complete."
