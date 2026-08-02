"""
Plot the multi-seed MNIST comparison sweep.

Reads all JSONs from results/ and produces three figures:
    1. Learning curves (test acc vs wall-clock, one line per method+rank,
       shaded confidence bands over seeds)
    2. Bar chart of best test accuracy with error bars
    3. Scatter of wall-clock at peak vs test accuracy at peak
"""
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

# --- Setup ---
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Consistent colours + display names across all three figures
METHOD_STYLE = {
    "backprop":    {"colour": "#1f77b4", "label": "Backprop + Adam"},
    "openai_es":   {"colour": "#ff7f0e", "label": "OpenAI-ES"},
    "sep_cma_es":  {"colour": "#2ca02c", "label": "Sep-CMA-ES"},
    "eggroll_r1":  {"colour": "#d62728", "label": "EGGROLL r=1"},
    "eggroll_r4":  {"colour": "#9467bd", "label": "EGGROLL r=4"},
    "eggroll_r16": {"colour": "#8c564b", "label": "EGGROLL r=16"},
}

# Plot order (controls legend order and z-order in overlays)
METHOD_ORDER = ["backprop", "openai_es", "sep_cma_es",
                "eggroll_r1", "eggroll_r4", "eggroll_r16"]


def load_all_results():
    """Load every JSON in results/, group by method."""
    grouped = defaultdict(list)
    for fn in sorted(RESULTS_DIR.glob("*.json")):
        with open(fn) as f:
            r = json.load(f)
        grouped[r["method"]].append(r)
    for method, runs in grouped.items():
        print(f"  {method}: {len(runs)} runs")
    return grouped


def figure_1_learning_curves(grouped):
    """Test accuracy vs wall-clock for every method, mean ± std across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        runs = grouped[method]
        style = METHOD_STYLE[method]

        # Each run has a history of (wall_s, step, test_acc).
        # Different runs have different wall_s grids — interpolate to a common grid.
        all_walls = [np.array([h[0] for h in r["history"]]) for r in runs]
        all_accs = [np.array([h[2] for h in r["history"]]) for r in runs]

        # Common wall-clock grid: from 0 to the shortest run's max time
        t_max = min(w[-1] for w in all_walls)
        t_grid = np.linspace(0, t_max, 200)

        # Interpolate each run's acc onto the grid
        interp_accs = np.stack([
            np.interp(t_grid, w, a) for w, a in zip(all_walls, all_accs)
        ])

        mean_acc = interp_accs.mean(axis=0)
        std_acc = interp_accs.std(axis=0)

        ax.plot(t_grid, mean_acc, color=style["colour"],
                label=style["label"], linewidth=2)
        ax.fill_between(t_grid, mean_acc - std_acc, mean_acc + std_acc,
                        color=style["colour"], alpha=0.15)

    ax.set_xlabel("Wall-clock time (seconds)", fontsize=12)
    ax.set_ylabel("Test accuracy", fontsize=12)
    ax.set_title("MNIST: test accuracy over training (3 seeds, mean ± std)",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    fn = FIGURES_DIR / "01 - Learning curves.png"
    fig.tight_layout()
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"  saved {fn}")
    plt.close(fig)


def figure_2_bar_chart(grouped):
    """Bar chart of best test acc with error bars across seeds."""
    fig, ax = plt.subplots(figsize=(9, 6))

    labels = []
    means = []
    stds = []
    colours = []

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        runs = grouped[method]
        style = METHOD_STYLE[method]
        accs = np.array([r["best_test_acc"] for r in runs])

        labels.append(style["label"])
        means.append(accs.mean())
        stds.append(accs.std())
        colours.append(style["colour"])

    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, color=colours,
                  edgecolor="black", linewidth=0.8)

    # Annotate each bar with the mean value
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Best test accuracy", fontsize=12)
    ax.set_title("MNIST: peak test accuracy by method (3 seeds, mean ± std)",
                 fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_axisbelow(True)

    fn = FIGURES_DIR / "02 - Best acc (bar).png"
    fig.tight_layout()
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"  saved {fn}")
    plt.close(fig)


def figure_3_wallclock_scatter(grouped):
    """Wall-clock at peak vs peak test accuracy. One point per seed, per method."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for method in METHOD_ORDER:
        if method not in grouped:
            continue
        runs = grouped[method]
        style = METHOD_STYLE[method]

        walls = np.array([r.get("converged_at_wall_s") or r.get("total_wall_s", 0)
                          for r in runs])
        accs = np.array([r["best_test_acc"] for r in runs])

        # One dot per seed
        ax.scatter(walls, accs, color=style["colour"], s=100,
                   edgecolor="black", linewidth=0.8, label=style["label"], alpha=0.85)

        # Annotate the mean point with the method name
        ax.annotate(
            style["label"],
            (walls.mean(), accs.mean()),
            xytext=(8, 5), textcoords="offset points",
            fontsize=9, fontweight="bold",
            color=style["colour"])
        
        # Plot the mean as a bigger marker
        ax.scatter(walls.mean(), accs.mean(), color=style["colour"], s=250,
                   marker="*", edgecolor="black", linewidth=1.2)

    ax.set_xlabel("Wall-clock to peak (seconds)", fontsize=12)
    ax.set_ylabel("Best test accuracy", fontsize=12)
    ax.set_title("MNIST: accuracy vs wall-clock tradeoff (dots = seeds, stars = mean)",
                 fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)

    fn = FIGURES_DIR / "03 - Wallclock (scatter).png"
    fig.tight_layout()
    fig.savefig(fn, dpi=150, bbox_inches="tight")
    print(f"  saved {fn}")
    plt.close(fig)


if __name__ == "__main__":
    print("Loading results...")
    grouped = load_all_results()

    print("\nGenerating figure 1: learning curves...")
    figure_1_learning_curves(grouped)

    print("\nGenerating figure 2: bar chart...")
    figure_2_bar_chart(grouped)

    print("\nGenerating figure 3: wall-clock scatter...")
    figure_3_wallclock_scatter(grouped)

    print(f"\nAll figures saved to {FIGURES_DIR}/")