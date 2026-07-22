"""
Template for a plotting utility that reads a JSON result file and produces
a figure. Copy this file, rename it, and edit.

Usage: python scripts/plot_<name>.py
"""

import os
import json
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe in scripts and terminals
import matplotlib.pyplot as plt

RESULTS_PATH = os.path.expanduser("~/dissertation/eggroll-diss/results/<name>.json")
FIGURE_PATH = os.path.expanduser("~/dissertation/eggroll-diss/figures/<name>.png")


def load(path):
    with open(path) as f:
        return json.load(f)


def plot(data, out_path):
    # your plotting code goes here
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Placeholder")
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    data = load(RESULTS_PATH)
    plot(data, FIGURE_PATH)
