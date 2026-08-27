"""Sensitivity Pilot Visualisation — Phase 1, 2, 3 bar charts.

Generates three figures from the pilot JSON:
  1. Phase 1 — Fitness variance per layer (elevation r=4 → r=8)
  2. Phase 2 — Fitness degradation per layer (ablation r=4 → r=1)
  3. Phase 3 — Binary inclusion (rank 0 vs rank 1 for least sensitive layer)

Usage:
    python plot_sensitivity_pilot.py

Reads from:  results/pilot_3phase/mnist/pilot_allocation.json
Saves to:    figures/pilot_phase1.png, figures/pilot_phase2.png, figures/pilot_phase3.png
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ── Config ──────────────────────────────────────────────────────────────
PILOT_PATH = "results/pilot_3phase/mnist/pilot_allocation.json"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

# Fallback: try alternative paths
ALT_PATHS = [
    "results/pilot_3phase/mnist/pilot_allocation.json",
    "results/pilot_3phase/MNIST/pilot_allocation.json",
    "results/sensitivity_pilot/mnist/pilot_allocation.json",
]

pilot = None
for p in [PILOT_PATH] + ALT_PATHS:
    if os.path.exists(p):
        with open(p) as f:
            pilot = json.load(f)
        print(f"Loaded pilot from: {p}")
        break

if pilot is None:
    print("ERROR: Could not find pilot_allocation.json. Tried:")
    for p in [PILOT_PATH] + ALT_PATHS:
        print(f"  {p}")
    print("\nPlease update PILOT_PATH in this script.")
    exit(1)

# ── Extract data from list-of-dicts structure ───────────────────────────
phase1 = pilot.get("phase1", pilot.get("phase_1", []))
phase2 = pilot.get("phase2", pilot.get("phase_2", []))
phase3 = pilot.get("phase3", pilot.get("phase_3", {}))
baseline_acc = pilot.get("phase2_baseline_accuracy", None)

# ── Friendly layer names ────────────────────────────────────────────────
LAYER_LABELS = {
    "(256, 784)": "Input\n(784→256)",
    "(10, 256)": "Output\n(256→10)",
    "(256, 256)": "Hidden\n(256→256)",
    "input": "Input",
    "hidden": "Hidden",
    "output": "Output",
    "layer_0": "Input\n(784→256)",
    "layer_1": "Hidden\n(256→256)",
    "layer_2": "Output\n(256→10)",
}

def friendly_name(raw):
    return LAYER_LABELS.get(raw, raw)

# ── Colour scheme ───────────────────────────────────────────────────────
COLORS = {
    "phase1": "#5DCAA5",  # teal
    "phase2": "#7F77DD",  # purple
    "phase3_r0": "#D85A30",  # coral
    "phase3_r1": "#378ADD",  # blue
}

# ── PHASE 1: Fitness variance bar chart ─────────────────────────────────
if phase1 and isinstance(phase1, list):
    labels = [friendly_name(entry["layer"]) for entry in phase1]
    values = [entry["mean"] for entry in phase1]
    stds = [entry.get("std", 0) for entry in phase1]

    # Sort by value descending
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, yerr=stds, capsize=5,
                  color=COLORS["phase1"], edgecolor="white", width=0.55)
    ax.set_ylabel("Fitness variance (higher = more sensitive)")
    ax.set_title("Phase 1 — Elevation test (r=1 → r=8 from shared checkpoint)",
                 fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.04,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pilot_phase1.png"), dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR}/pilot_phase1.png")

    with open(os.path.join(FIG_DIR, "pilot_phase1_explanation.txt"), "w") as f:
        f.write("Phase 1 — Refinement (elevation r=1 → r=8 from shared checkpoint)\n\n")
        f.write("Starting from a shared checkpoint trained at uniform r=4 for 100 "
                "generations, each layer is elevated to r=8 one at a time while all other "
                "layers remain at r=1. A single generation is executed from the checkpoint "
                "(no continued training) and the fitness variance across the population is "
                "recorded. Higher variance indicates the layer responds more strongly to "
                "increased perturbation rank — it benefits more from directional coverage. "
                "The shared checkpoint ensures all measurements start from identical network "
                "weights, eliminating divergence bias. Phase 1 is confirmatory: it validates "
                "the ordering from Phase 2 rather than establishing it.\n")
else:
    print("WARNING: Could not extract Phase 1 scores. Check pilot JSON structure.")

# ── PHASE 2: Degradation bar chart ──────────────────────────────────────
if phase2 and isinstance(phase2, list):
    labels = [friendly_name(entry["layer"]) for entry in phase2]
    values = [entry["mean"] for entry in phase2]
    stds = [entry.get("std", 0) for entry in phase2]

    # Sort by absolute value descending (largest degradation = most sensitive)
    order = np.argsort([abs(v) for v in values])[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [COLORS["phase2"] if v >= 0 else "#D85A30" for v in values]
    bars = ax.bar(labels, values, yerr=stds, capsize=5,
                  color=colors, edgecolor="white", width=0.55)
    ax.set_ylabel("Accuracy degradation (higher = more sensitive)")
    ax.set_title("Phase 2 — Causal ablation (drop layer r=4 → r=1)",
                 fontsize=14, fontweight="bold")
    ax.axhline(y=0, color="gray", linewidth=0.5, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)

    if baseline_acc is not None:
        ax.annotate(f"Baseline accuracy: {baseline_acc:.4f}",
                    xy=(0.02, 0.95), xycoords="axes fraction",
                    ha="left", va="top", fontsize=9, color="gray")

    for bar, val in zip(bars, values):
        y_pos = (bar.get_height() + max(abs(v) for v in values)*0.04 if val >= 0
                 else bar.get_height() - max(abs(v) for v in values)*0.06)
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                f"{val:+.4f}", ha="center", va=va, fontsize=11)

    if any(v < 0 for v in values):
        ax.annotate("Negative = rank reduction\nimproves accuracy (\"loudly wrong\")",
                    xy=(0.95, 0.05), xycoords="axes fraction",
                    ha="right", va="bottom", fontsize=9,
                    color="#D85A30", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pilot_phase2.png"), dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR}/pilot_phase2.png")

    with open(os.path.join(FIG_DIR, "pilot_phase2_explanation.txt"), "w") as f:
        f.write("Phase 2 — Primary ordering (causal ablation, drop to r=1)\n\n")
        f.write(f"Baseline accuracy at uniform r=4: {baseline_acc:.4f}\n\n" if baseline_acc else "")
        f.write("This is the workhorse of the sensitivity pilot. Each layer is individually "
                "dropped from r=4 to r=1 while all other layers remain at r=4. The accuracy "
                "degradation measures how much performance suffers when that layer receives a "
                "lower-quality gradient estimate. Larger degradation means the layer is more "
                "sensitive — it needs a higher rank to maintain performance. The ordering from "
                "Phase 2 directly determines the rank allocation: the most sensitive layer gets "
                "the highest rank, and the least sensitive gets the lowest.\n\n"
                "Negative degradation (coral bars) means reducing the layer's rank actually "
                "improved accuracy. This is the 'loudly wrong' finding: some layers generate "
                "high-variance perturbation signal that is actively harmful. Freezing them "
                "(rank 0) is not just cost-saving — it improves model quality.\n")
else:
    print("WARNING: Could not extract Phase 2 scores. Check pilot JSON structure.")

# ── PHASE 3: Binary inclusion comparison ────────────────────────────────
# Structure: {"layer_name": "output", "rank0_mean": X, "rank1_mean": Y, ...}
r0_score = None
r1_score = None
p3_layer = None

if isinstance(phase3, dict):
    r0_score = phase3.get("rank0_mean")
    r1_score = phase3.get("rank1_mean")
    r0_std = phase3.get("rank0_std", 0)
    r1_std = phase3.get("rank1_std", 0)
    r0_accs = phase3.get("rank0_accs", [])
    r1_accs = phase3.get("rank1_accs", [])
    p3_layer = phase3.get("layer_name", "least sensitive")
    assigned = phase3.get("assigned_rank", None)

if r0_score is not None and r1_score is not None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar([f"Rank 0\n(frozen)", f"Rank 1\n(minimal)"],
                  [r0_score, r1_score],
                  yerr=[r0_std, r1_std], capsize=6,
                  color=[COLORS["phase3_r0"], COLORS["phase3_r1"]],
                  edgecolor="white", width=0.45)

    for bar, val in zip(bars, [r0_score, r1_score]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(r0_score, r1_score)*0.03,
                f"{val:.4f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Plot individual seed points
    if r0_accs:
        ax.scatter([0]*len(r0_accs), r0_accs, color="white", edgecolor=COLORS["phase3_r0"],
                   s=40, zorder=5, linewidth=1.5)
    if r1_accs:
        ax.scatter([1]*len(r1_accs), r1_accs, color="white", edgecolor=COLORS["phase3_r1"],
                   s=40, zorder=5, linewidth=1.5)

    ax.set_ylabel("Test Accuracy")
    ax.set_title(f"Phase 3 — Binary inclusion ({p3_layer} layer)",
                 fontsize=14, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

    winner = "Rank 0" if r0_score > r1_score else "Rank 1"
    diff = abs(r0_score - r1_score)
    ax.annotate(f"{winner} wins by {diff:.4f} ({diff*100:.2f}pp) → assigned rank {assigned}",
                xy=(0.5, 0.92), xycoords="axes fraction",
                ha="center", fontsize=10, fontstyle="italic",
                color=COLORS["phase3_r0"] if r0_score > r1_score else COLORS["phase3_r1"])

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "pilot_phase3.png"), dpi=200)
    plt.close()
    print(f"Saved {FIG_DIR}/pilot_phase3.png")

    with open(os.path.join(FIG_DIR, "pilot_phase3_explanation.txt"), "w") as f:
        f.write(f"Phase 3 — Binary inclusion (rank 0 vs rank 1 for {p3_layer} layer)\n\n")
        f.write(f"Rank 0 mean accuracy: {r0_score:.4f} ± {r0_std:.4f} "
                f"(seeds: {[f'{a:.4f}' for a in r0_accs]})\n")
        f.write(f"Rank 1 mean accuracy: {r1_score:.4f} ± {r1_std:.4f} "
                f"(seeds: {[f'{a:.4f}' for a in r1_accs]})\n\n")
        f.write(f"Result: {winner} wins by {diff:.4f} ({diff*100:.2f} percentage points). "
                f"Assigned rank: {assigned}.\n\n")
        if r0_score > r1_score:
            f.write("Freezing the least sensitive layer is actively beneficial — the "
                    "perturbation signal from this layer was harmful, and removing it "
                    "improves overall model quality. This is the 'loudly wrong' effect: "
                    "even minimal perturbation (rank 1) introduces noise that degrades "
                    "accuracy. Rank 0 eliminates this harmful signal entirely.\n")
        else:
            f.write("Minimal perturbation outperforms freezing. The layer still contributes "
                    "useful signal, even at rank 1. This suggests a floor constraint: "
                    "the layer cannot be safely frozen in this environment.\n")
else:
    print("WARNING: Could not extract Phase 3 scores.")
    print(f"Phase 3 data: {phase3}")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n── Summary ──")
p1_layers = [e["layer"] for e in phase1] if isinstance(phase1, list) else "NOT FOUND"
p2_layers = [e["layer"] for e in phase2] if isinstance(phase2, list) else "NOT FOUND"
print(f"Phase 1 layers: {p1_layers}")
print(f"Phase 2 layers: {p2_layers}")
print(f"Phase 3: {p3_layer} layer — r0={r0_score}, r1={r1_score}")
print(f"Final allocation: {pilot.get('rank_allocation_named', 'N/A')}")
print(f"\nAll figures saved to {FIG_DIR}/")
print("Each figure has a matching _explanation.txt file.")
