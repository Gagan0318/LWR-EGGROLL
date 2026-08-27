"""Sequential launcher: Brax Ant mini signal test FIRST, then LunarLander capped LWR.
Run from ~/dissertation/eggroll-diss/
Brax finishes first (~1-2h) so you can start writing. LunarLander continues in background.
"""
import subprocess, sys, time

SCRIPTS = [
    ("experiments/brax_ant_mini_test.py", "Brax Ant mini signal test"),
    ("experiments/lunarlander_symmetric_tuned.py", "LunarLander capped LWR + summary regen"),
]

t0 = time.time()
for script, label in SCRIPTS:
    print(f"\n{'='*60}")
    print(f"LAUNCHING: {label}")
    print(f"Script: {script}")
    print(f"{'='*60}\n", flush=True)

    t_start = time.time()
    result = subprocess.run(
        [sys.executable, "-u", script],
        cwd="/home/gagan/dissertation/eggroll-diss" if __name__ == "__main__" else ".",
    )
    elapsed = time.time() - t_start

    if result.returncode != 0:
        print(f"\n*** {label} FAILED (exit code {result.returncode}) after {elapsed:.0f}s ***", flush=True)
        print(f"Continuing to next script...\n", flush=True)
    else:
        print(f"\n*** {label} DONE in {elapsed:.0f}s ***\n", flush=True)

total = time.time() - t0
print(f"\n{'='*60}")
print(f"ALL DONE — total {total:.0f}s ({total/60:.1f} min)")
print(f"{'='*60}")
