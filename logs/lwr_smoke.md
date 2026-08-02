# 2026-07-26 — LWR-EGGROLL smoke test

## What was built
- lwr_eggroll.py noiser class in HyperscaleES fork
- Refactored compare_4_methods_mnist.py: shared training body for EGGROLL/LWR
- Two smoke test configs: uniform r=4 (sanity) and hetero 8/4/2 (hand-picked)

## Results (n=3 seeds, pooled across two identical runs)
- lwr_uniform_r4:    ~0.827 pooled
- lwr_hetero_8_4_2:  ~0.829 pooled
- Verdict: statistically indistinguishable. Guessed allocation ≈ uniform.

## Known issues
- Small numerical discrepancy vs vanilla EGGROLL r=4 (0.827 vs 0.8272±0.001) 
  — unclear source, possibly JIT scheduling variance
- Run-to-run variance ~0.008 at n=3 seeds — need n=5+ for real claims
- Shape-collapse limitation: two (256,256) hidden layers share rank; per-position 
  identity needs framework refactor (~4-6hr, deferred)

## Monday agenda
1. Follow up with Lehre on the email
2. Design sensitivity pilot (see design_notes/)
3. Debug JIT variance (low-priority)

## Sensitivity pilot: four open decisions
1. What does "layer sensitivity" mean operationally?
2. Pilot protocol: OAT vs full factorial vs random
3. Allocation rule: fixed budget vs threshold vs ladder
4. Validation: MNIST alone or transfer to Brax
