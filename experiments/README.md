# Experiments

Each script maps to a section of Chapter 4. Scripts write results to
`results/<experiment_name>/` (relative to the repo root) and can be
re-run independently; existing per-seed JSONs are skipped on re-run.

## LunarLander family

Four distinct experiments across two architectures:

| Script | Architecture | Purpose |
|--------|-------------|---------|
| `lunarlander_lwr.py` | [8, 64, 64, 4] symmetric | Original symmetric run: ES method comparison with pilot-derived LWR. Produced the near-zero ES fitness that motivated the tuned follow-up. |
| `lunarlander_symmetric_tuned.py` | [8, 64, 64, 4] symmetric | Re-runs the symmetric setup with 8 eval episodes (from 5) to test whether the near-zero result was fitness noise rather than an ES limitation. |
| `lunarlander_tapered.py` | [8, 256, 64, 4] tapered | Tapered (non-uniform width) architecture with a live sensitivity pilot, 5 seeds, and a REINFORCE baseline. |
| `lunarlander_tapered_v2.py` | [8, 256, 64, 4] tapered | Capped-budget allocations on the tapered architecture (r=1 baseline plus LWR variants capped at rank 4). Source of the Chapter 4 capped-allocation results. |

## Wide hidden ablation

| Script | Purpose |
|--------|---------|
| `wide_hidden_ablation_v2.py` | Wide-hidden architecture ablation referenced in Chapter 4. |
