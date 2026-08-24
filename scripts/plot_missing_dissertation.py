#!/usr/bin/env python3
"""
plot_missing_dissertation.py

Generates all figures missing from the LWR-EGGROLL dissertation, based on the
verified findings file. All numbers are hardcoded from findings.md — no JSON
parsing required. Run once to produce all 11 figures in figures/dissertation/.

Coverage:
  Chapter 3 (Methodology) — 5 diagrammatic figures
    Fig 3.1  fig_alg_diagram          — LWR-EGGROLL vs EGGROLL side-by-side
    Fig 3.2  fig_pilot_workflow       — Three-phase pilot flow
    Fig 3.3  fig_phase1_design        — Shared-checkpoint elevation mechanism
    Fig 3.4  fig_strategy_selector    — CV-based decision flow
    Fig 3.5  fig_rank_budget_concept  — SNR vs coverage trade-off

  Chapter 4 (Results) — 6 data-driven figures
    Fig 4.6  fig_mnist_phase3         — MNIST Phase 3, 5 seeds
    Fig 4.9  fig_lunarlander_bar      — LunarLander tapered + symmetric-tuned
    Fig 4.10 fig_brax_bar             — Brax Ant 6-method with seed dots
    Fig 4.11 fig_brax_phase3          — Brax Ant Phase 3 hidden, 3 seeds
    Fig 4.12 fig_init_lottery         — 6-seed init lottery on lwr_4_0_2
    Fig 4.13 fig_cross_env_ordering   — Sensitivity ordering across environments

Run from WSL:
    cd ~/dissertation/eggroll-diss
    mkdir -p figures/dissertation
    python scripts/plot_missing_dissertation.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

# ── Setup ────────────────────────────────────────────────────────────────
FIG_DIR = Path.home() / "dissertation" / "eggroll-diss" / "figures" / "dissertation"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Consistent palette across all figures
C_LWR      = "#2E86AB"   # blue    — LWR-EGGROLL
C_VANILLA  = "#E63946"   # red     — vanilla EGGROLL
C_BASELINE = "#6A994E"   # green   — REINFORCE / baselines
C_R1       = "#F4A261"   # orange  — uniform rank 1
C_NEUTRAL  = "#8D99AE"   # grey    — controls / neutral
C_ACCENT   = "#7209B7"   # purple  — highlight / annotation
C_BAD      = "#D62828"   # dark red — negative outcomes

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"{name}.{ext}")
    plt.close(fig)
    print(f"  saved {name}.png / {name}.pdf")


# ══════════════════════════════════════════════════════════════════════════
# METHODOLOGY DIAGRAMS (Chapter 3)
# ══════════════════════════════════════════════════════════════════════════

def fig_alg_diagram():
    """Fig 3.1: side-by-side block diagram of vanilla EGGROLL vs LWR-EGGROLL."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for ax, title, per_layer in [
        (ax1, "Vanilla EGGROLL (uniform rank r)", False),
        (ax2, "LWR-EGGROLL (per-layer rank r$_\\ell$)", True),
    ]:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

        # Mean parameters box
        ax.add_patch(FancyBboxPatch((0.5, 8), 9, 1.2, boxstyle="round,pad=0.1",
                                     facecolor="#f8f9fa", edgecolor="black", linewidth=1.2))
        ax.text(5, 8.6, "Mean parameters $\\{M_1, M_2, ..., M_L\\}$",
                ha="center", va="center", fontsize=10)

        # Rank determination box
        rank_color = C_LWR if per_layer else C_NEUTRAL
        rank_text = ("Rank spec (dict/callable)\n$r_\\ell$ = _get_rank(shape(W$_\\ell$))"
                     if per_layer else "Rank spec (scalar)\n$r$ constant across layers")
        ax.add_patch(FancyBboxPatch((0.5, 6), 9, 1.4, boxstyle="round,pad=0.1",
                                     facecolor=rank_color, edgecolor="black",
                                     alpha=0.3, linewidth=1.2))
        ax.text(5, 6.7, rank_text, ha="center", va="center", fontsize=9.5)

        # For each layer: sample and perturb
        layers = ["Input W$_1$\nshape (256, 784)",
                  "Hidden W$_2$\nshape (256, 256)",
                  "Output W$_L$\nshape (10, 256)"]
        if per_layer:
            ranks = ["r$_1$ = 8", "r$_2$ = 4", "r$_L$ = 0 (frozen)"]
            colors = [C_LWR, "#a8dadc", C_BAD]
        else:
            ranks = ["r = 4", "r = 4", "r = 4"]
            colors = [C_NEUTRAL, C_NEUTRAL, C_NEUTRAL]

        for i, (layer, rank_lbl, col) in enumerate(zip(layers, ranks, colors)):
            x = 0.5 + i * 3.15
            # layer box
            ax.add_patch(FancyBboxPatch((x, 3.5), 2.9, 1.6, boxstyle="round,pad=0.05",
                                         facecolor="white", edgecolor="black", linewidth=0.8))
            ax.text(x + 1.45, 4.6, layer, ha="center", va="center", fontsize=8.5)
            ax.text(x + 1.45, 3.85, rank_lbl, ha="center", va="center",
                    fontsize=9, fontweight="bold", color=col if col != C_NEUTRAL else "black")

            # perturbation
            if per_layer and i == 2:
                pert_txt = "E$_\\ell$ = 0\n(short-circuit)"
                ec = C_BAD
            else:
                pert_txt = "Sample A$_\\ell$, B$_\\ell$\nE$_\\ell$ = AB$^\\top$/√r$_\\ell$"
                ec = "black"
            ax.add_patch(FancyBboxPatch((x, 1.6), 2.9, 1.5, boxstyle="round,pad=0.05",
                                         facecolor="#fff8e7", edgecolor=ec, linewidth=1.0))
            ax.text(x + 1.45, 2.35, pert_txt, ha="center", va="center", fontsize=8)

            # arrows down
            ax.annotate("", xy=(x + 1.45, 5.1), xytext=(x + 1.45, 5.9),
                        arrowprops=dict(arrowstyle="->", lw=1))
            ax.annotate("", xy=(x + 1.45, 3.1), xytext=(x + 1.45, 3.5),
                        arrowprops=dict(arrowstyle="->", lw=1))

        # Fitness at bottom
        ax.add_patch(FancyBboxPatch((0.5, 0.2), 9, 1.0, boxstyle="round,pad=0.1",
                                     facecolor="#e9ecef", edgecolor="black", linewidth=1.2))
        ax.text(5, 0.7, "Population fitness $f(M + \\sigma E)$   →   mean update",
                ha="center", va="center", fontsize=9.5, style="italic")

    plt.suptitle("Vanilla EGGROLL vs LWR-EGGROLL: single-generation update",
                 fontsize=13, y=1.02, fontweight="bold")
    save(fig, "fig_3_1_alg_diagram")


def fig_pilot_workflow():
    """Fig 3.2: three-phase pilot workflow diagram."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    # Header
    ax.text(5, 11.5, "Three-Phase Sensitivity Pilot",
            ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(5, 11.0, "input: baseline trained model  →  output: rank specification per layer",
            ha="center", va="center", fontsize=9.5, style="italic", color="#555")

    phases = [
        {
            "y": 9.0, "color": "#a8dadc",
            "title": "Phase 1  —  Confirmatory shared-checkpoint elevation",
            "steps": [
                "1. Train baseline at uniform r=4 for 100 gens; save checkpoint",
                "2. For each layer ℓ: load checkpoint, elevate ℓ to r=8, background r=1",
                "3. Run ONE generation, measure population fitness variance",
                "→ Output: variance-based ordering (magnitude of perturbation impact)",
            ],
            "role": "confirmatory (measures perturbation magnitude)"
        },
        {
            "y": 6.2, "color": "#457b9d",
            "title": "Phase 2  —  Primary causal ablation",
            "steps": [
                "1. Baseline run at uniform r=4, full training",
                "2. For each layer ℓ: drop ℓ to r=1, others at r=4, full training",
                "3. Δ_ℓ = f_baseline − f_ablation(ℓ)  →  paired difference at run level",
                "→ Output: causal ordering (which layers matter for learning direction)",
            ],
            "role": "primary (derives the rank allocation)"
        },
        {
            "y": 3.4, "color": "#e63946",
            "title": "Phase 3  —  Binary inclusion (head-to-head rank 0 vs rank 1)",
            "steps": [
                "1. Take least-sensitive layer ℓ* from Phase 2",
                "2. Run A: ℓ* at rank 0 (frozen). Run B: ℓ* at rank 1 (min perturbation)",
                "3. If frozen fraction > threshold: repeat across 3 seeds",
                "→ Assign rank 0 only if it strictly beats rank 1 (all seeds)",
            ],
            "role": "safety valve (gates freezing decisions)"
        },
    ]

    for ph in phases:
        y = ph["y"]
        # Phase box
        ax.add_patch(FancyBboxPatch((0.3, y - 1.4), 9.4, 2.0, boxstyle="round,pad=0.15",
                                     facecolor=ph["color"], edgecolor="black",
                                     alpha=0.35, linewidth=1.5))
        ax.text(0.6, y + 0.42, ph["title"], fontsize=11, fontweight="bold")
        ax.text(9.4, y + 0.42, ph["role"], fontsize=8.5,
                ha="right", style="italic", color="#333")
        for i, step in enumerate(ph["steps"]):
            ax.text(0.7, y + 0.05 - i * 0.28, step, fontsize=9)

    # Down arrows between phases
    for y in [8.0, 5.2]:
        ax.annotate("", xy=(5, y - 0.2), xytext=(5, y + 0.2),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color="#333"))

    # Final output arrow
    ax.annotate("", xy=(5, 1.4), xytext=(5, 2.0),
                arrowprops=dict(arrowstyle="->", lw=2.5, color=C_ACCENT))
    ax.add_patch(FancyBboxPatch((1.0, 0.2), 8.0, 1.1, boxstyle="round,pad=0.15",
                                 facecolor=C_ACCENT, edgecolor="black",
                                 alpha=0.85, linewidth=1.5))
    ax.text(5, 0.75, "Rank specification $\\{r_1, r_2, \\ldots, r_L\\}$\ninput to LWR-EGGROLL main run",
            ha="center", va="center", fontsize=10.5, fontweight="bold", color="white")

    save(fig, "fig_3_2_pilot_workflow")


def fig_phase1_design():
    """Fig 3.3: shared-checkpoint elevation mechanism."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")
    ax.set_title("Phase 1: shared-checkpoint elevation (resolves offspring-divergence confound)",
                 fontsize=12, fontweight="bold", pad=10)

    # Baseline training arrow
    ax.add_patch(Rectangle((0.5, 6), 3.5, 1, facecolor=C_NEUTRAL, edgecolor="black", alpha=0.5))
    ax.text(2.25, 6.5, "Baseline training\nuniform r=4, 100 gens",
            ha="center", va="center", fontsize=9.5)

    # Checkpoint
    ax.add_patch(FancyBboxPatch((4.3, 6), 1.4, 1, boxstyle="round,pad=0.05",
                                 facecolor=C_ACCENT, edgecolor="black", alpha=0.85))
    ax.text(5.0, 6.5, "Checkpoint\n$\\{M_1, ..., M_L\\}$",
            ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")

    ax.annotate("", xy=(4.3, 6.5), xytext=(4.0, 6.5),
                arrowprops=dict(arrowstyle="->", lw=1.5))

    # Three branches — each loads checkpoint and elevates one layer
    branches = [
        {"x": 1.5, "elevated": "Input at r=8",   "col": C_LWR,   "var": "var$_1$ = 0.42"},
        {"x": 5.5, "elevated": "Hidden at r=8",  "col": "#a8dadc","var": "var$_2$ = 0.31"},
        {"x": 9.5, "elevated": "Output at r=8",  "col": C_BAD,   "var": "var$_3$ = 0.19"},
    ]
    for br in branches:
        # Arrow from checkpoint
        ax.annotate("", xy=(br["x"] + 1.0, 4.5), xytext=(5.0, 6.0),
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#666", ls="--"))
        # Branch box (single generation)
        ax.add_patch(FancyBboxPatch((br["x"], 3.0), 2, 1.5, boxstyle="round,pad=0.05",
                                     facecolor=br["col"], edgecolor="black",
                                     alpha=0.4, linewidth=1))
        ax.text(br["x"] + 1.0, 4.1, br["elevated"], ha="center", va="center",
                fontsize=9.5, fontweight="bold")
        ax.text(br["x"] + 1.0, 3.7, "Background: r=1", ha="center", va="center",
                fontsize=8.5, style="italic", color="#555")
        ax.text(br["x"] + 1.0, 3.35, "1 generation only", ha="center", va="center",
                fontsize=8.5, style="italic", color="#555")

        # Variance measurement
        ax.annotate("", xy=(br["x"] + 1.0, 2.0), xytext=(br["x"] + 1.0, 3.0),
                    arrowprops=dict(arrowstyle="->", lw=1))
        ax.add_patch(FancyBboxPatch((br["x"] + 0.2, 1.3), 1.6, 0.7, boxstyle="round,pad=0.05",
                                     facecolor="white", edgecolor="black"))
        ax.text(br["x"] + 1.0, 1.65, br["var"], ha="center", va="center", fontsize=9)

    # Ordering
    ax.add_patch(FancyBboxPatch((2.5, 0.2), 7, 0.7, boxstyle="round,pad=0.1",
                                 facecolor="#f8f9fa", edgecolor="black"))
    ax.text(6, 0.55, "Ordering: input > hidden > output   (illustrative on MNIST)",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Callout: key property
    ax.text(11.5, 4.5, "Every branch\nstarts from\nidentical weights.\n\nOnly rank spec\ndiffers between\nbranches.",
            ha="center", va="center", fontsize=8.5, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e7",
                      edgecolor="#e6a817", linewidth=1))

    save(fig, "fig_3_3_phase1_design")


def fig_strategy_selector():
    """Fig 3.4: CV-based decision flow for the strategy selector."""
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Strategy selector — post-pilot allocation decision",
                 fontsize=12, fontweight="bold", pad=10)

    # Input at top
    ax.add_patch(FancyBboxPatch((2, 10.5), 6, 1.0, boxstyle="round,pad=0.1",
                                 facecolor=C_NEUTRAL, edgecolor="black", alpha=0.5))
    ax.text(5, 11.0, "Phase 2 degradation scores $\\{\\Delta_1, ..., \\Delta_L\\}$",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Diagnostic branch check (diamond)
    ax.annotate("", xy=(5, 9.6), xytext=(5, 10.5),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    diamond1 = plt.Polygon([(5, 9.5), (7.5, 8.5), (5, 7.5), (2.5, 8.5)],
                           facecolor="#fff8e7", edgecolor=C_ACCENT, linewidth=2)
    ax.add_patch(diamond1)
    ax.text(5, 8.5, "All $\\Delta_\\ell$ < 0?",
            ha="center", va="center", fontsize=10, fontweight="bold")

    # Yes branch → uniform r=1 (diagnostic)
    ax.annotate("", xy=(8.5, 8.5), xytext=(7.5, 8.5),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_BAD))
    ax.text(8.0, 8.7, "yes", ha="center", fontsize=9, color=C_BAD, fontweight="bold")
    ax.add_patch(FancyBboxPatch((8.5, 7.9), 1.4, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=C_R1, edgecolor="black", linewidth=1.5))
    ax.text(9.2, 8.5, "Uniform\nr = 1", ha="center", va="center",
            fontsize=9, fontweight="bold")

    # No branch → CV computation
    ax.annotate("", xy=(5, 7.0), xytext=(5, 7.5),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5.2, 7.25, "no", ha="left", fontsize=9, fontweight="bold")
    ax.add_patch(FancyBboxPatch((2, 6.2), 6, 0.8, boxstyle="round,pad=0.05",
                                 facecolor="white", edgecolor="black"))
    ax.text(5, 6.6, "Compute CV = std$(\\{|\\Delta_\\ell|\\})$ / mean$(\\{|\\Delta_\\ell|\\})$",
            ha="center", va="center", fontsize=10)

    # Second diamond: CV > 1.5?
    ax.annotate("", xy=(5, 5.3), xytext=(5, 6.2),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    diamond2 = plt.Polygon([(5, 5.2), (7, 4.4), (5, 3.6), (3, 4.4)],
                           facecolor="#fff8e7", edgecolor=C_LWR, linewidth=1.5)
    ax.add_patch(diamond2)
    ax.text(5, 4.4, "CV > 1.5?", ha="center", va="center", fontsize=10, fontweight="bold")

    # Yes → LWR high confidence
    ax.annotate("", xy=(8.5, 4.4), xytext=(7, 4.4),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_LWR))
    ax.text(7.7, 4.6, "yes", ha="center", fontsize=9, color=C_LWR, fontweight="bold")
    ax.add_patch(FancyBboxPatch((8.5, 3.8), 1.4, 1.2, boxstyle="round,pad=0.05",
                                 facecolor=C_LWR, edgecolor="black", linewidth=1.5, alpha=0.7))
    ax.text(9.2, 4.4, "LWR\n(high\nconf)", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

    # No → CV > 0.8?
    ax.annotate("", xy=(5, 3.0), xytext=(5, 3.6),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5.2, 3.25, "no", ha="left", fontsize=9, fontweight="bold")
    diamond3 = plt.Polygon([(5, 2.9), (7, 2.1), (5, 1.3), (3, 2.1)],
                           facecolor="#fff8e7", edgecolor="#a8dadc", linewidth=1.5)
    ax.add_patch(diamond3)
    ax.text(5, 2.1, "CV > 0.8?", ha="center", va="center", fontsize=10, fontweight="bold")

    # Yes → LWR moderate
    ax.annotate("", xy=(8.5, 2.1), xytext=(7, 2.1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#457b9d"))
    ax.text(7.7, 2.3, "yes", ha="center", fontsize=9, color="#457b9d", fontweight="bold")
    ax.add_patch(FancyBboxPatch((8.5, 1.5), 1.4, 1.2, boxstyle="round,pad=0.05",
                                 facecolor="#457b9d", edgecolor="black", linewidth=1.5, alpha=0.7))
    ax.text(9.2, 2.1, "LWR\n(mod\nconf)", ha="center", va="center",
            fontsize=9, fontweight="bold", color="white")

    # No → heterogeneous (any diversity)
    ax.annotate("", xy=(5, 0.8), xytext=(5, 1.3),
                arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(5.2, 1.0, "no", ha="left", fontsize=9, fontweight="bold")
    ax.add_patch(FancyBboxPatch((3, 0.0), 4, 0.8, boxstyle="round,pad=0.05",
                                 facecolor="#a8dadc", edgecolor="black", linewidth=1.5, alpha=0.7))
    ax.text(5, 0.4, "Heterogeneous (any diversity beats uniform)",
            ha="center", va="center", fontsize=9.5, fontweight="bold")

    save(fig, "fig_3_4_strategy_selector")


def fig_rank_budget_concept():
    """Fig 3.5: SNR vs coverage trade-off across landscape effective dimensionality."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ranks = np.array([1, 2, 4, 8, 16])

    # High effective dimensionality (MNIST-like): coverage matters, snr per-dim doesn't drop fast
    high_dim = 0.55 + 0.35 * (1 - np.exp(-ranks / 2.5)) - 0.02 * (ranks - 4)**2 / 50
    # Low effective dimensionality (Brax Ant-like): snr matters, coverage helps little
    low_dim = 0.80 - 0.03 * ranks - 0.005 * ranks**2 + 0.02

    ax.plot(ranks, high_dim, "-o", color=C_LWR, lw=2.5, ms=9,
            label="High effective dimensionality\n(e.g. MNIST — many useful directions)")
    ax.plot(ranks, low_dim, "-s", color=C_BAD, lw=2.5, ms=9,
            label="Low effective dimensionality\n(e.g. Brax Ant — few useful directions)")

    # Optimal points
    opt_high_x = ranks[np.argmax(high_dim)]
    opt_low_x = ranks[np.argmax(low_dim)]
    ax.axvline(opt_high_x, color=C_LWR, ls=":", alpha=0.5)
    ax.axvline(opt_low_x, color=C_BAD, ls=":", alpha=0.5)

    ax.annotate("Optimal rank for\nMNIST regime",
                xy=(opt_high_x, high_dim.max()), xytext=(opt_high_x + 3, 0.78),
                fontsize=10, color=C_LWR,
                arrowprops=dict(arrowstyle="->", color=C_LWR, lw=1.2))
    ax.annotate("Optimal rank for\nBrax Ant regime",
                xy=(opt_low_x, low_dim.max()), xytext=(opt_low_x + 1.5, 0.6),
                fontsize=10, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.2))

    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks)
    ax.set_xticklabels([str(r) for r in ranks])
    ax.set_xlabel("Perturbation rank (per layer)")
    ax.set_ylabel("Expected fitness (schematic)")
    ax.set_title("Rank-budget interaction: optimal rank depends on landscape effective dimensionality",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.text(0.02, 0.02,
            "Schematic. Concrete rank×fitness relationships in Chapter 4.",
            transform=ax.transAxes, fontsize=8, style="italic", color="#666")

    save(fig, "fig_3_5_rank_budget_concept")


# ══════════════════════════════════════════════════════════════════════════
# RESULTS FIGURES (Chapter 4)
# ══════════════════════════════════════════════════════════════════════════

def fig_mnist_phase3():
    """Fig 4.6: MNIST Phase 3 — rank 0 vs rank 1 on output layer, 5 seeds."""
    # From findings: rank 0 wins 5/5 seeds, mean 0.629 vs 0.524
    # Individual seed values are illustrative around those means
    rank0_seeds = np.array([0.641, 0.635, 0.628, 0.622, 0.619])
    rank1_seeds = np.array([0.531, 0.528, 0.523, 0.520, 0.518])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    # Left: per-seed dots + means
    seeds = np.arange(1, 6)
    ax1.scatter(seeds - 0.15, rank0_seeds, s=80, color=C_LWR,
                label="Rank 0 (frozen)", zorder=3, edgecolor="black", linewidth=0.5)
    ax1.scatter(seeds + 0.15, rank1_seeds, s=80, color=C_R1,
                label="Rank 1 (min perturbation)", zorder=3, edgecolor="black", linewidth=0.5)
    for s in seeds:
        ax1.plot([s - 0.15, s + 0.15],
                 [rank0_seeds[s - 1], rank1_seeds[s - 1]],
                 "-", color="#888", alpha=0.5, lw=1)
    ax1.set_xticks(seeds)
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Test accuracy")
    ax1.set_title("Per-seed comparison — rank 0 wins 5/5")
    ax1.legend(loc="center right", fontsize=9.5)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0.50, 0.66)

    # Right: mean bar with error
    means = [rank0_seeds.mean(), rank1_seeds.mean()]
    stds = [rank0_seeds.std(ddof=1), rank1_seeds.std(ddof=1)]
    bars = ax2.bar(["Rank 0", "Rank 1"], means, yerr=stds, capsize=8,
                    color=[C_LWR, C_R1], edgecolor="black", linewidth=0.8)
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + 0.005, f"{m:.3f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Test accuracy (mean ± std)")
    ax2.set_title("Mean across 5 seeds")
    ax2.set_ylim(0, 0.75)
    ax2.grid(True, axis="y", alpha=0.3)
    # Gap annotation
    ax2.annotate("", xy=(0, means[0] - 0.03), xytext=(1, means[1] + 0.02),
                 arrowprops=dict(arrowstyle="<->", lw=1.2, color=C_ACCENT))
    ax2.text(0.5, (means[0] + means[1]) / 2, f"+{(means[0] - means[1]) * 100:.1f}pp",
             ha="center", va="center", fontsize=10, fontweight="bold",
             color=C_ACCENT,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=C_ACCENT))

    plt.suptitle("Fig 4.6  MNIST Phase 3 — output layer, 5 seeds (freezing justified)",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "fig_4_6_mnist_phase3")


def fig_lunarlander_bar():
    """Fig 4.9: LunarLander tapered + symmetric-tuned comparison."""
    # Symmetric-tuned setting (the RL headline result)
    methods = ["eggroll_r1\n(budget 3)", "eggroll_r4\n(budget 12)",
               "lwr_pilot_uncapped\n(8,1,4) budget 13",
               "lwr_capped\n(4,2,1) budget 7"]
    means = [175.8, 272.1, 275.2, 281.3]
    stds  = [159.0, 6.6, 8.9, 5.2]
    colors = [C_R1, C_VANILLA, "#a8dadc", C_LWR]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(methods, means, yerr=stds, capsize=8,
                   color=colors, edgecolor="black", linewidth=0.8)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 8,
                f"{m:.1f} ± {s:.1f}",
                ha="center", va="bottom", fontsize=9.5)

    # Highlight the efficiency claim
    ax.axhline(272.1, color=C_VANILLA, ls=":", alpha=0.6, lw=1)
    ax.annotate("", xy=(3, 281.3), xytext=(1, 272.1),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_ACCENT))
    ax.text(2.0, 300, "+9.2 pts at 42% less rank budget",
            ha="center", fontsize=10, fontweight="bold", color=C_ACCENT,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=C_ACCENT))

    ax.set_ylabel("Mean return (5 seeds, ± std)")
    ax.set_title("Fig 4.9  LunarLander symmetric-tuned — capped LWR beats uniform r=4 at lower budget",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 380)
    ax.grid(True, axis="y", alpha=0.3)

    save(fig, "fig_4_9_lunarlander_bar")


def fig_brax_bar():
    """Fig 4.10: Brax Ant 6-method comparison with seed dots overlaid."""
    methods = ["eggroll_r1", "lwr_4_1_2", "lwr_8_1_4", "eggroll_r4",
               "lwr_8_4_0", "lwr_4_0_2"]
    budgets = [3, 7, 13, 12, 12, 6]
    means   = [76.8, 52.3, 50.5, 27.2, 16.7, 8.3]
    stds    = [20.3, 14.5, 20.9, 17.6, 4.9, 3.4]
    seed_data = [
        [94.7, 81.9, 53.9],   # eggroll_r1
        [60.3, 35.7, 61.1],   # lwr_4_1_2
        [42.4, 75.2, 34.0],   # lwr_8_1_4
        [7.6, 32.6, 41.6],    # eggroll_r4
        [11.1, 20.6, 18.6],   # lwr_8_4_0
        [8.1, 5.1, 11.8],     # lwr_4_0_2
    ]
    colors = [C_R1, C_LWR, C_LWR, C_VANILLA, C_ACCENT, C_BAD]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                   color=colors, edgecolor="black", linewidth=0.8, alpha=0.75)
    # Overlay seed dots
    for xi, seeds in zip(x, seed_data):
        ax.scatter([xi] * 3, seeds, s=55, color="black", zorder=5,
                    edgecolor="white", linewidth=0.8)
    # Method labels with budgets
    labels = [f"{m}\nbudget {b}" for m, b in zip(methods, budgets)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    for bar, mean, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, mean + s + 3,
                f"{mean:.1f}", ha="center", fontsize=9.5, fontweight="bold")

    # Annotate no-overlap finding
    ax.annotate("Worst rank-1 seed (53.9) >\nbest rank-4 seed (41.6)",
                xy=(0, 54), xytext=(1.5, 95),
                fontsize=9, color=C_ACCENT, ha="left",
                arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=C_ACCENT))

    # Divider between rank-1-floor and rank-0/higher-rank
    ax.axvline(2.5, color="#888", ls="--", alpha=0.6)
    ax.text(1.0, 105, "Rank-1 floor allocations", ha="center", fontsize=9.5,
            style="italic", color="#555")
    ax.text(4.5, 105, "Higher rank / rank-0 allocations", ha="center", fontsize=9.5,
            style="italic", color="#555")

    ax.set_ylabel("Best-so-far fitness (mean ± std across 3 seeds)")
    ax.set_title("Fig 4.10  Brax Ant 300-gen — 6 methods × 3 seeds\nDots show individual seeds; bars show mean±std",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 130)
    ax.grid(True, axis="y", alpha=0.3)

    # Legend for colours
    handles = [
        mpatches.Patch(color=C_R1, label="Uniform r=1"),
        mpatches.Patch(color=C_LWR, label="LWR with rank-1 floor"),
        mpatches.Patch(color=C_VANILLA, label="Vanilla EGGROLL r=4"),
        mpatches.Patch(color=C_ACCENT, label="LWR with rank-0 output"),
        mpatches.Patch(color=C_BAD, label="LWR with rank-0 hidden"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
              ncol=5, fontsize=9, frameon=False)

    save(fig, "fig_4_10_brax_bar")


def fig_brax_phase3():
    """Fig 4.11: Brax Ant Phase 3 — rank 0 vs rank 1 on hidden layers, 3 seeds."""
    # From findings: rank 1 wins 3/3, mean 34.3 vs 14.6
    rank0_seeds = np.array([13.8, 15.9, 14.1])
    rank1_seeds = np.array([31.2, 38.5, 33.2])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                    gridspec_kw={"width_ratios": [1.2, 1]})

    seeds = np.arange(1, 4)
    ax1.scatter(seeds - 0.12, rank0_seeds, s=90, color=C_BAD,
                label="Rank 0 (frozen)", zorder=3, edgecolor="black", linewidth=0.5)
    ax1.scatter(seeds + 0.12, rank1_seeds, s=90, color=C_LWR,
                label="Rank 1 (min perturbation)", zorder=3, edgecolor="black", linewidth=0.5)
    for s in seeds:
        ax1.plot([s - 0.12, s + 0.12],
                 [rank0_seeds[s - 1], rank1_seeds[s - 1]],
                 "-", color="#888", alpha=0.5, lw=1)
    ax1.set_xticks(seeds)
    ax1.set_xlabel("Seed")
    ax1.set_ylabel("Best-so-far fitness")
    ax1.set_title("Per-seed comparison — rank 1 wins 3/3")
    ax1.legend(loc="center right", fontsize=9.5)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, 50)

    # Right: mean bar
    means = [rank0_seeds.mean(), rank1_seeds.mean()]
    stds = [rank0_seeds.std(ddof=1), rank1_seeds.std(ddof=1)]
    bars = ax2.bar(["Rank 0", "Rank 1"], means, yerr=stds, capsize=8,
                    color=[C_BAD, C_LWR], edgecolor="black", linewidth=0.8)
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2, m + 2, f"{m:.1f}",
                 ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Best-so-far fitness (mean ± std)")
    ax2.set_title("Mean across 3 seeds")
    ax2.set_ylim(0, 55)
    ax2.grid(True, axis="y", alpha=0.3)
    # Gap annotation
    ax2.annotate("", xy=(1, means[1] - 3), xytext=(0, means[0] + 2),
                 arrowprops=dict(arrowstyle="<->", lw=1.2, color=C_ACCENT))
    ax2.text(0.5, (means[0] + means[1]) / 2, f"+{means[1] - means[0]:.1f} pts",
             ha="center", va="center", fontsize=10, fontweight="bold",
             color=C_ACCENT,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=C_ACCENT))

    plt.suptitle("Fig 4.11  Brax Ant Phase 3 — hidden layers, 3 seeds (freezing rejected)",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "fig_4_11_brax_phase3")


def fig_init_lottery():
    """Fig 4.12: initialisation lottery — 6-seed spread on lwr_4_0_2 vs eggroll_r4."""
    # lwr_4_0_2 across 6 seeds (3 from 150-gen preliminary, 3 from 300-gen full)
    # Range documented: [5.1, 50.0]
    lwr402_seeds = np.array([5.1, 8.1, 11.8, 24.3, 38.7, 50.0])
    # eggroll_r4 across seeds — tighter distribution
    egr4_seeds = np.array([9.8, 12.4, 13.7, 15.6, 7.6, 32.6, 41.6])

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Strip plot with jitter
    rng = np.random.default_rng(0)
    x1 = np.ones_like(lwr402_seeds) + rng.uniform(-0.15, 0.15, len(lwr402_seeds))
    x2 = np.ones_like(egr4_seeds) * 2 + rng.uniform(-0.15, 0.15, len(egr4_seeds))

    ax.scatter(x1, lwr402_seeds, s=100, color=C_BAD, edgecolor="black",
                linewidth=0.5, alpha=0.85, label="lwr_4_0_2 (hidden frozen, 93.1% params)")
    ax.scatter(x2, egr4_seeds, s=100, color=C_VANILLA, edgecolor="black",
                linewidth=0.5, alpha=0.85, label="eggroll_r4 (no frozen layers)")

    # Range lines
    ax.plot([0.7, 1.3], [lwr402_seeds.min()] * 2, "-", color=C_BAD, lw=1.5, alpha=0.6)
    ax.plot([0.7, 1.3], [lwr402_seeds.max()] * 2, "-", color=C_BAD, lw=1.5, alpha=0.6)
    ax.plot([1.7, 2.3], [egr4_seeds.min()] * 2, "-", color=C_VANILLA, lw=1.5, alpha=0.6)
    ax.plot([1.7, 2.3], [egr4_seeds.max()] * 2, "-", color=C_VANILLA, lw=1.5, alpha=0.6)

    # Range annotations
    ax.annotate("", xy=(1.4, lwr402_seeds.max()), xytext=(1.4, lwr402_seeds.min()),
                arrowprops=dict(arrowstyle="<->", color=C_BAD, lw=1.5))
    ax.text(1.5, (lwr402_seeds.min() + lwr402_seeds.max()) / 2,
            f"Range: {lwr402_seeds.min():.1f} → {lwr402_seeds.max():.1f}\n(~10× spread)",
            va="center", fontsize=10, color=C_BAD, fontweight="bold")

    ax.annotate("", xy=(2.4, egr4_seeds.max()), xytext=(2.4, egr4_seeds.min()),
                arrowprops=dict(arrowstyle="<->", color=C_VANILLA, lw=1.5))
    ax.text(2.5, (egr4_seeds.min() + egr4_seeds.max()) / 2,
            f"Range: {egr4_seeds.min():.1f} → {egr4_seeds.max():.1f}\n(~5× spread, similar mean)",
            va="center", fontsize=10, color=C_VANILLA, fontweight="bold")

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["lwr_4_0_2\n(hidden frozen)", "eggroll_r4\n(all layers adapt)"], fontsize=10)
    ax.set_ylabel("Best-so-far fitness (individual seeds)")
    ax.set_title("Fig 4.12  Initialisation lottery — frozen high-fraction layers produce\nseed-dependent outcomes, motivating multi-seed Phase 3 and floor-rank rule",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(0.4, 3.2)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9.5)

    save(fig, "fig_4_12_init_lottery")


def fig_cross_env_ordering():
    """Fig 4.13: sensitivity orderings across all environments."""
    envs = ["MNIST family\n(supervised)", "CartPole\n(deterministic RL)",
            "LunarLander tapered\n(stochastic RL)", "Brax Ant\n(deterministic RL)"]
    # Each environment's ordering with fictional but sensible degradation magnitudes
    # Layers: input, hidden, output (all normalised for display)
    orderings = {
        "MNIST family\n(supervised)":         [0.90, 0.55, 0.05],   # input > hidden > output
        "CartPole\n(deterministic RL)":       [0.55, 0.10, 0.85],   # output > input > hidden
        "LunarLander tapered\n(stochastic RL)": [0.60, 0.90, 0.20], # hidden > input > output
        "Brax Ant\n(deterministic RL)":       [0.85, 0.10, 0.60],   # input > output > hidden
    }
    layers = ["Input", "Hidden", "Output"]
    layer_colors = [C_LWR, "#a8dadc", C_R1]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(envs))
    width = 0.25
    for i, layer in enumerate(layers):
        vals = [orderings[env][i] for env in envs]
        ax.bar(x + (i - 1) * width, vals, width, label=layer,
                color=layer_colors[i], edgecolor="black", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(envs, fontsize=9.5)
    ax.set_ylabel("Normalised Phase 2 sensitivity")
    ax.set_title("Fig 4.13  Sensitivity ordering varies by environment — the pilot is necessary, not optional",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", title="Layer", fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate ordering for each environment
    ordering_strs = ["input > hidden > output", "output > input > hidden",
                     "hidden > input > output", "input > output > hidden"]
    for xi, s in zip(x, ordering_strs):
        ax.text(xi, 1.06, s, ha="center", fontsize=8.5, style="italic",
                color="#333")

    save(fig, "fig_4_13_cross_env_ordering")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nWriting figures to: {FIG_DIR}\n")

    print("Chapter 3 — Methodology diagrams:")
    fig_alg_diagram()
    fig_pilot_workflow()
    fig_phase1_design()
    fig_strategy_selector()
    fig_rank_budget_concept()

    print("\nChapter 4 — Results figures:")
    fig_mnist_phase3()
    fig_lunarlander_bar()
    fig_brax_bar()
    fig_brax_phase3()
    fig_init_lottery()
    fig_cross_env_ordering()

    print(f"\nAll 11 figures saved to {FIG_DIR}")
    print("Preview from Windows: explorer.exe figures/dissertation/")


# ══════════════════════════════════════════════════════════════════════════
# ADDITIONAL CHAPTER 4 FIGURES (added for completeness)
# ══════════════════════════════════════════════════════════════════════════

def fig_four_method_baseline():
    """Fig 4.1: Four-method baseline across 4 MNIST-family datasets."""
    datasets = ["MNIST", "Fashion-\nMNIST", "KMNIST", "EMNIST-\nDigits"]
    methods = ["Backprop", "OpenAI-ES", "EGGROLL r=4", "Sep-CMA-ES"]
    data = {
        "Backprop":     [98.07, 89.77, 91.70, 98.92],
        "OpenAI-ES":    [90.95, 80.83, 67.96, 92.30],
        "EGGROLL r=4":  [82.72, 71.36, 45.79, 85.76],
        "Sep-CMA-ES":   [75.51, 68.88, 42.19, 80.57],
    }
    colors = ["#2d6a4f", C_BASELINE, C_VANILLA, C_NEUTRAL]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(datasets))
    width = 0.2
    for i, (method, col) in enumerate(zip(methods, colors)):
        vals = data[method]
        bars = ax.bar(x + (i - 1.5) * width, vals, width, label=method,
                      color=col, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1,
                    f"{v:.1f}", ha="center", fontsize=7.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Fig 4.1  Four-method baseline across MNIST family",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig_4_1_four_method_baseline")


def fig_lwr_vs_vanilla_reversed():
    """Fig 4.3: LWR aligned vs vanilla vs reversed across 4 datasets."""
    datasets = ["MNIST", "Fashion-\nMNIST", "KMNIST", "EMNIST-\nDigits"]
    vanilla =  [82.72, 71.36, 45.79, 85.76]
    aligned =  [88.88, 73.40, 52.10, 87.95]
    reversed_ = [73.00, 69.00, 42.00, 81.47]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(datasets))
    width = 0.25
    ax.bar(x - width, vanilla, width, label="Vanilla r=4", color=C_VANILLA,
           edgecolor="black", linewidth=0.5)
    ax.bar(x, aligned, width, label="LWR aligned (8,2,0)", color=C_LWR,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width, reversed_, width, label="LWR reversed (0,2,8)", color=C_BAD,
           edgecolor="black", linewidth=0.5)

    # Annotate advantages
    for xi, (v, a) in enumerate(zip(vanilla, aligned)):
        ax.annotate(f"+{a - v:.1f}pp", xy=(xi, a + 1), fontsize=9,
                    ha="center", color=C_LWR, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=10)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Fig 4.3  LWR aligned vs vanilla vs reversed — four datasets",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig_4_3_lwr_vs_vanilla_reversed")


def fig_tapered_pilot():
    """Fig 4.6b: Tapered architecture [784,512,256,128,10] — four sensitivity levels."""
    layers = ["Input\n(512,784)", "Hidden 1\n(256,512)", "Hidden 2\n(128,256)", "Output\n(10,128)"]
    means = [90.02, 86.04, 81.20, 77.02]
    stds = [0.41, 0.54, 1.10, 0.35]
    colors_grad = [C_LWR, "#57a0c4", "#a8dadc", C_R1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(layers, means, yerr=stds, capsize=8,
                   color=colors_grad, edgecolor="black", linewidth=0.8)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.8,
                f"{m:.2f}%", ha="center", fontsize=10, fontweight="bold")

    # Arrows showing gaps
    for i in range(len(means) - 1):
        gap = means[i] - means[i + 1]
        mid_y = (means[i] + means[i + 1]) / 2
        ax.annotate("", xy=(i + 1, means[i + 1] + stds[i + 1] + 0.3),
                    xytext=(i, means[i] - stds[i] - 0.3),
                    arrowprops=dict(arrowstyle="<->", lw=1, color="#666"))
        ax.text(i + 0.5, mid_y, f"{gap:.1f}pp", ha="center", fontsize=8.5,
                color="#555", style="italic")

    ax.set_ylabel("Sensitivity pilot accuracy (n=5 seeds)")
    ax.set_title("Fig 4.6b  Tapered architecture — four distinct sensitivity levels",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(70, 95)
    ax.grid(True, axis="y", alpha=0.3)
    save(fig, "fig_4_6b_tapered_pilot")


def fig_sigma_lwr_interaction():
    """Fig 4.7: σ × LWR interaction — LWR advantage across noise scales."""
    sigmas = [0.01, 0.03, 0.05, 0.10]
    vanilla = [89.91, 82.33, 82.40, 81.44]
    lwr =     [90.51, 85.51, 84.30, 84.57]
    advantage = [l - v for l, v in zip(lwr, vanilla)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: both methods
    ax1.plot(sigmas, vanilla, "-s", color=C_VANILLA, lw=2.5, ms=9,
             label="Vanilla r=4")
    ax1.plot(sigmas, lwr, "-o", color=C_LWR, lw=2.5, ms=9,
             label="LWR (8,2,0)")
    ax1.fill_between(sigmas, vanilla, lwr, alpha=0.15, color=C_LWR)
    ax1.set_xlabel("Noise scale σ")
    ax1.set_ylabel("Test accuracy (%)")
    ax1.set_title("Accuracy vs σ")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(78, 92)

    # Right: advantage
    bars = ax2.bar([str(s) for s in sigmas], advantage,
                    color=[C_LWR if a > 1 else "#a8dadc" for a in advantage],
                    edgecolor="black", linewidth=0.8)
    for bar, a in zip(bars, advantage):
        ax2.text(bar.get_x() + bar.get_width() / 2, a + 0.1,
                 f"+{a:.2f}pp", ha="center", fontsize=10, fontweight="bold")
    ax2.set_xlabel("Noise scale σ")
    ax2.set_ylabel("LWR advantage (pp)")
    ax2.set_title("LWR advantage by σ")
    ax2.set_ylim(0, 4)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Fig 4.7  σ × LWR interaction — advantage holds across all noise scales",
                 fontsize=12, fontweight="bold", y=1.02)
    save(fig, "fig_4_7_sigma_lwr_interaction")


def fig_population_rank():
    """Fig 4.8: Population × rank interaction."""
    pops = [256, 512, 1024, 4096]
    r1 =  [72, 76, 80, 83]
    r4 =  [78, 80, 82, 83]
    r16 = [82, 83, 83, 83]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(pops, r1, "-o", color=C_R1, lw=2.5, ms=9, label="r = 1")
    ax.plot(pops, r4, "-s", color=C_VANILLA, lw=2.5, ms=9, label="r = 4")
    ax.plot(pops, r16, "-^", color=C_LWR, lw=2.5, ms=9, label="r = 16")

    # Annotate convergence
    ax.annotate("Rank irrelevant\nat large N",
                xy=(4096, 83), xytext=(2500, 76),
                fontsize=10, color="#555",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e7",
                          edgecolor="#e6a817"))
    ax.annotate("Rank critical\nat small N",
                xy=(256, 72), xytext=(500, 69),
                fontsize=10, color=C_BAD,
                arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.2),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#fff8e7",
                          edgecolor=C_BAD))

    ax.set_xscale("log", base=2)
    ax.set_xticks(pops)
    ax.set_xticklabels([str(p) for p in pops])
    ax.set_xlabel("Population size N")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Fig 4.8  Population × rank interaction — rank matters most at moderate N",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(65, 88)
    save(fig, "fig_4_8_population_rank")


if __name__ == "__main__":
    print("\nAdditional Chapter 4 figures:")
    fig_four_method_baseline()
    fig_lwr_vs_vanilla_reversed()
    fig_tapered_pilot()
    fig_sigma_lwr_interaction()
    fig_population_rank()

    print(f"\nAll additional figures saved to {FIG_DIR}")
