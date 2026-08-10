"""New Phase 1 Experiments — Complete overnight run.

Runs everything sequentially:
  1. Three-phase sensitivity pilot (checkpoint Phase 1) on 4 MNIST datasets
  2. CartPole-v1: OpenAI-ES vs EGGROLL vs LWR vs REINFORCE
  3. LunarLander-v3 (symmetric): same comparison
  4. LunarLander-v3 (tapered): architectural asymmetry hypothesis test

Expected total runtime: ~6-8 hours.
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

EXPERIMENTS = [
    {
        "name": "Three-Phase Pilot — Checkpoint Phase 1 (4 datasets)",
        "script": "run_3phase_pilot_all_datasets.py",
        "log": "logs/pilot_checkpoint.log",
    },
    {
        "name": "CartPole-v1 — ES Method Comparison",
        "script": "cartpole_lwr.py",
        "log": "logs/cartpole_lwr.log",
    },
    {
        "name": "LunarLander-v3 — Symmetric Architecture",
        "script": "lunarlander_lwr.py",
        "log": "logs/lunarlander_symmetric.log",
    },
    {
        "name": "LunarLander-v3 — Tapered Architecture (asymmetry hypothesis)",
        "script": "lunarlander_tapered.py",
        "log": "logs/lunarlander_tapered.log",
    },
]


def main():
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("NEW PHASE 1 EXPERIMENTS — COMPLETE OVERNIGHT RUN")
    print(f"Experiments: {len(EXPERIMENTS)}")
    for i, exp in enumerate(EXPERIMENTS, 1):
        print(f"  {i}. {exp['name']}")
    print("=" * 70, flush=True)

    t_total = time.time()
    results_summary = []

    for i, exp in enumerate(EXPERIMENTS, 1):
        script_path = SCRIPTS_DIR / exp["script"]
        log_path = Path(exp["log"])

        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(EXPERIMENTS)}] {exp['name']}")
        print(f"# Script: {script_path}")
        print(f"# Log: {log_path}")
        print(f"{'#' * 70}\n", flush=True)

        if not script_path.exists():
            print(f"  ERROR: {script_path} not found. Skipping.", flush=True)
            results_summary.append((exp["name"], "SKIPPED", 0))
            continue

        t0 = time.time()

        with open(log_path, "w") as log_file:
            proc = subprocess.run(
                [sys.executable, "-u", str(script_path)],
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(Path.cwd()),
            )

        elapsed = time.time() - t0

        if proc.returncode == 0:
            status = "DONE"
            print(f"  DONE in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)
            with open(log_path) as f:
                lines = f.readlines()
                tail = lines[-20:] if len(lines) > 20 else lines
                for line in tail:
                    print(f"  | {line.rstrip()}")
        else:
            status = "FAILED"
            print(f"  FAILED (exit code {proc.returncode}) after {elapsed:.0f}s", flush=True)
            with open(log_path) as f:
                lines = f.readlines()
                tail = lines[-15:] if len(lines) > 15 else lines
                for line in tail:
                    print(f"  | {line.rstrip()}")

        results_summary.append((exp["name"], status, elapsed))
        print("", flush=True)

    total_elapsed = time.time() - t_total

    print(f"\n{'=' * 70}")
    print("OVERNIGHT RUN COMPLETE")
    print(f"{'=' * 70}")
    print(f"\n{'Experiment':<55} {'Status':<10} {'Time'}")
    print("-" * 70)
    for name, status, elapsed in results_summary:
        print(f"{name:<55} {status:<10} {elapsed/60:.1f} min")
    print(f"\nTotal time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'=' * 70}", flush=True)


if __name__ == "__main__":
    main()
