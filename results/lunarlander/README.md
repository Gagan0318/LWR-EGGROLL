# LunarLander — Stochastic RL Experiments

LunarLander-v3 (Gymnasium) with stochastic wind. Four sub-experiments investigating LWR-EGGROLL under noisy fitness signals, progressing from a failed initial attempt to a successful efficiency result.

## Sub-experiments

- **main_comparison/** — Symmetric architecture [8, 256, 256, 4], initial four-method comparison. ES methods failed (mean fitness of approximately 10) due to insufficient fitness signal from 5 stochastic episodes. This negative result motivated the tuned and tapered follow-ups.
- **symmetric_tuned/** — Same symmetric architecture with 8 evaluation episodes and extended LWR configurations. CV on Phase 2 scores fell below 0.8, indicating that ordering signal is buried in environmental noise — but rank diversity still outperforms uniformity.
- **tapered/** — Tapered architecture [8, 256, 64, 4] with live sensitivity pilot, 5 seeds, and a REINFORCE baseline. Source of the headline result: capped LWR (4,2,1) at budget 7 beats vanilla r=4 at budget 12 by +9.2 reward points.
- **tapered_v2/** — Extended capped-budget investigation on the tapered architecture. Confirms that lower total rank is consistently better on stochastic RL when the floor-rank rule (minimum rank 1) is applied.
