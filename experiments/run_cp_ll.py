"""Overnight launcher: three-phase pilot + RL experiments.

Runs sequentially:
  1. Three-phase sensitivity pilot on MNIST, Fashion-MNIST, KMNIST, EMNIST-Digits
  2. CartPole-v1: OpenAI-ES vs vanilla EGGROLL vs LWR-EGGROLL
  3. LunarLander-v3: OpenAI-ES vs vanilla EGGROLL vs LWR-EGGROLL

Expected total runtime: ~4-6 hours.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    {
        "name": "Three-Phase Sensitivity Pilot (4 datasets)",
        "script": "run_3phase_pilot_all_datasets.py",
        "log": "logs/3phase_pilot_all.log",
    },
    {
        "name": "CartPole-v1 ES Comparison",
        "script": "cartpole_lwr.py",
        "log": "logs/cartpole_lwr.log",
    },
    {
        "name": "LunarLander-v3 ES Comparison",
        "script": "lunarlander_lwr.py",
        "log": "logs/lunarlander_lwr.log",
    },
]


def main():
    # Ensure logs directory exists
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("OVERNIGHT LAUNCHER")
    print(f"Experiments: {len(EXPERIMENTS)}")
    print("=" * 60)

    t_total = time.time()

    for i, exp in enumerate(EXPERIMENTS, 1):
        script_path = SCRIPTS_DIR / exp["script"]
        log_path = Path(exp["log"])

        print(f"\n{'#' * 60}")
        print(f"# [{i}/{len(EXPERIMENTS)}] {exp['name']}")
        print(f"# Script: {script_path}")
        print(f"# Log: {log_path}")
        print(f"{'#' * 60}\n")

        if not script_path.exists():
            print(f"  ERROR: {script_path} not found. Skipping.")
            continue

        t0 = time.time()

        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                [sys.executable, str(script_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path.cwd()),
            )

        elapsed = time.time() - t0

        if proc.returncode == 0:
            print(f"  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)")
            # Print last 15 lines of log as summary
            with open(log_path) as f:
                lines = f.readlines()
                tail = lines[-15:] if len(lines) > 15 else lines
                for line in tail:
                    print(f"  | {line.rstrip()}")
        else:
            print(f"  FAILED (exit code {proc.returncode}) after {elapsed:.0f}s")
            print(f"  Check {log_path} for details.")
            with open(log_path) as f:
                lines = f.readlines()
                tail = lines[-10:] if len(lines) > 10 else lines
                for line in tail:
                    print(f"  | {line.rstrip()}")

    total_elapsed = time.time() - t_total
    print(f"\n{'=' * 60}")
    print(f"ALL EXPERIMENTS COMPLETE")
    print(f"Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
