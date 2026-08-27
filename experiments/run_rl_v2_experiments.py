"""Overnight launcher: RL v2 experiments.
1. Tapered v2 — r=1 baseline + capped LWR (same eval conditions as original)
2. Symmetric tuned — higher eval episodes + larger population

Run from: ~/dissertation/eggroll-diss/
"""
import subprocess, sys, time

EXPERIMENTS = [
    {
        "name": "LunarLander Tapered v2 (r=1 baseline + capped LWR)",
        "script": "experiments/lunarlander_tapered_v2.py",
        "log": "logs/lunarlander_tapered_v2.log",
    },
    {
        "name": "LunarLander Symmetric Tuned (POP=512, EVAL=16)",
        "script": "experiments/lunarlander_symmetric_tuned.py",
        "log": "logs/lunarlander_symmetric_tuned.log",
    },
]

print("=" * 70)
print("RL V2 EXPERIMENTS — OVERNIGHT RUN")
print(f"Experiments: {len(EXPERIMENTS)}")
for i, e in enumerate(EXPERIMENTS, 1):
    print(f"  {i}. {e['name']}")
print("=" * 70, flush=True)

for i, exp in enumerate(EXPERIMENTS, 1):
    print(f"\n{'#'*70}")
    print(f"# [{i}/{len(EXPERIMENTS)}] {exp['name']}")
    print(f"# Script: {exp['script']}")
    print(f"# Log: {exp['log']}")
    print(f"{'#'*70}", flush=True)

    t0 = time.time()
    with open(exp["log"], "w") as log:
        result = subprocess.run(
            [sys.executable, "-u", exp["script"]],
            stdout=log, stderr=subprocess.STDOUT
        )
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (rc={result.returncode})"
    print(f"  → {status} in {elapsed:.0f}s ({elapsed/60:.1f} min)", flush=True)

print(f"\n{'=' * 70}")
print("ALL RL V2 EXPERIMENTS COMPLETE")
print(f"{'=' * 70}", flush=True)
