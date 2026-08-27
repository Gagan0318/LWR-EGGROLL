#!/bin/bash
# Usage: cd ~/dissertation/eggroll-diss && bash experiments/run_all_plots.sh
set -e
mkdir -p figures

echo "════════════════════════════════════════════"
echo "  Generating all figures for meeting"
echo "════════════════════════════════════════════"

SCRIPTS=(
    "experiments/plot_wall_clock_budget.py"
    "experiments/plot_phase2_degradation.py"
    "experiments/plot_sigma_rank_heatmap.py"
    "experiments/plot_lwr_vs_vanilla_reversed.py"
    "experiments/plot_pop_rank_interaction.py"
    "experiments/plot_lunarlander.py"
    "experiments/plot_convergence_curves.py"
    "experiments/plot_brax_ant.py"
)

PASSED=0; FAILED=0; SKIPPED=0
for script in "${SCRIPTS[@]}"; do
    name=$(basename "$script" .py)
    echo ""
    echo "── $name ──"
    if [ ! -f "$script" ]; then
        echo "  [SKIP] $script not found"
        ((SKIPPED++)); continue
    fi
    if python "$script" 2>&1; then
        ((PASSED++))
    else
        echo "  [FAIL] $name — check paths above"
        ((FAILED++))
    fi
done

echo ""
echo "════════════════════════════════════════════"
echo "  Done: $PASSED passed, $FAILED failed, $SKIPPED skipped"
echo "════════════════════════════════════════════"
ls -la figures/*.png 2>/dev/null || echo "  (no PNGs generated)"
