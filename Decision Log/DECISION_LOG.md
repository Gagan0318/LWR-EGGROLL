# LWR-EGGROLL Project Decision Log

This document records the key decisions made throughout the LWR-EGGROLL dissertation project, from initial exploration through final submission. It captures what was decided, the rationale, and the outcome.


---


# Phase 1: Domain Exploration and Foundation


## Learning Evolutionary Strategies from Scratch

**Decision:** Pursue an Evolution Strategies (ES) project with ES being a completely new academic domain that had not been covered in the MSc Data Science modules.

**Rationale:** The project topic (extending EGGROLL) was assigned by the supervisor (Prof. Per Kristian Lehre). This required self-directed learning of an entirely new domain — ES theory, the OpenAI-ES paper (Salimans et al., 2017), CMA-ES variants, low-rank perturbation methods, and the EGGROLL framework (Sarkar et al., 2026).

**Outcome:** A considerable portion of the project's early phase was devoted to reading foundational ES literature, understanding the mathematical framework (fitness shaping, natural gradient approximation, rank-r perturbation), and building working implementations from scratch. This independent learning is itself a contribution — the project demonstrates the ability to enter a new research domain and produce original experimental work within it.


## Local Development Environment


**Decision:** Set up WSL2 (Ubuntu) with JAX 0.11.0, HyperscaleES, evosax 0.2.0, and an RTX 5060 GPU for local development.

**Rationale:** Needed a reproducible environment for iterating on code before running expensive experiments on Colab.

**Outcome:** Successful. All code was developed and debugged locally before migration to Colab for GPU-intensive training runs.


## Four-Method Baseline Comparison


**Decision:** Benchmark Backpropagation + Adam, OpenAI-ES, Sep-CMA-ES, and vanilla EGGROLL on MNIST before any LWR work.

**Rationale:** Establishes the performance hierarchy that LWR-EGGROLL is positioned within. Confirms EGGROLL's Pareto position (lower cost 
than OpenAI-ES, lower accuracy) and identifies KMNIST as the dataset most sensitive to perturbation quality.

**Outcome:** Clean hierarchy: Backprop >> OpenAI-ES > EGGROLL > Sep-CMA-ES. Used as Table 4.1 in the dissertation.


## σ × Rank Interaction Sweep


**Decision:** Run a full grid of noise scale (σ) and perturbation rank combinations on MNIST.

**Rationale:** Understand the interaction between these two hyperparameters before introducing per-layer allocation.

**Outcome:** Confirmed that higher σ increases fitness variance, which higher rank can counteract. This theoretical grounding informed the design of LWR-EGGROLL — allocate rank where it reduces variance most.


---


# Phase 2: Sensitivity Pilot Design


## Phase 1 — Shared Checkpoint Design (Final)


**Decision:** Use a shared-checkpoint approach for Phase 1 rather than independent training runs.

**Rationale:** Four iterations were attempted. 
(1) Rank-zero background isolation created an unrealistic model context. 
(2) Independent runs at rank-one background introduced stochastic divergence — differences could not be attributed to the rank change alone (identified by Prof. Lehre). 
(3) Wall-clock burn-in was approached as a potential idea, but dropped due to it not being reproducible across hardware. 
(4) The shared checkpoint eliminates all confounders by construction: every measurement starts from identical weights.

**Outcome:** Phase 1 became a clean, reproducible perturbation-magnitude measurement. However, magnitude alone proved insufficient for ordering — leading to Phase 2.


## Phase 2 as Primary Ordering Mechanism


**Decision:** Designate Phase 2 (causal ablation) as the primary sensitivity ordering, with Phase 1 assigned the refinement role.

**Rationale:** Phase 1 measures how *much* a layer perturbs fitness. Phase 2 measures whether that perturbation helps or hurts learning. The MNIST output layer illustrates the distinction: Phase 1 shows high magnitude (it produces large fitness variance), but Phase 2 shows this is harmful (reducing its rank improves performance) — the "loudly wrong" pattern. Phase 2 orderings were consistent across all four MNIST-family datasets; Phase 1 orderings varied.

**Outcome:** This pivot, catalysed by supervisory feedback, was the single most important methodological decision. It changed the entire allocation logic and directly produced the headline results.


## Phase 1 Cross-Referencing (Refinement Role)


**Decision:** Allow Phase 1 magnitudes to downgrade (but never upgrade) middle-tier ranks assigned by Phase 2.

**Rationale:** Phase 1 was assigned the role to measure magnitude. But with Phase 2 doing all the heavy liifting, and independently allocating ranks by the produced ordering made phase 1 obsolete. So, phase 1 was given a secondary role to adapt moderately effective layer to the most practical rank. Conserves rank budget. If Phase 2 says a layer has moderate sensitivity, but Phase 1 says its perturbation magnitude is low, a lower rank (rank 2 instead of rank 4) suffices. This saves budget without introducing unvalidated upgrades.

**Outcome:** Rank 2 became the budget-saving assignment for layers with moderate directional sensitivity but low magnitude.


## Phase 3 — Binary Freeze Decision


**Decision:** Add a head-to-head rank 0 vs rank 1 test for the least-sensitive layer.

**Rationale:** "Least sensitive" (Phase 2) does not mean "safe to freeze." Even if phase 2 points to the layer not being extremely helpful to the model, freezing it means stripping the model of any potential exploration, no matter how little the layer positively contrbutes to the model. This only fails when the layer actively hurts the model by inducing noise instead of useful signal

**Outcome:** Phase 3 runs both rank 0 and rank 1 generations, and measures the hed-to-head results. The winner decides what rank is to be allocated.


## Confirmation Gate for Large Frozen Fractions


**Decision:** When the candidate layer holds more than 50% of total parameters, cross-check with best fitness before assigning rank 0.

**Rationale:** High-parameter-fraction freezing is high-stakes — if wrong, the majority of the network is untrainable. Best fitness provides 
a second opinion that may disagree with mean fitness in RL environments.

**Outcome:** This gate surfaced the Brax Ant best-fitness finding. The gate itself produces allocation (4, 1, 2) as a safer fallback for Brax Ant; the full best-fitness pilot produces (4, 2, 0) which achieves dramatically higher fitness.


## Strategy Selector


**Decision:** Build a selector using the coefficient of variation (CV) of Phase 2 degradation scores to classify confidence levels and detect 
degenerate landscapes.

**Rationale:** Not all environments benefit equally from heterogeneous allocation. The selector uses CV > 1.5 (high confidence), CV > 0.8 (moderate), CV < 0.8 (low — ordering is unreliable), and an all-negative diagnostic branch (all Phase 2 scores negative → low effective 
dimensionality → uniform rank 1 fallback).

**Outcome:** The diagnostic branch correctly identified Brax Ant's low-effective-dimensionality regime under mean fitness, prescribing uniform rank 1 — which the six-method comparison confirmed as optimal.


## Rank Set: {0, 1, 2, 4, 8}


**Decision:** Constrain the rank set to five values (powers of two plus zero).

**Rationale:** Each unique rank triggers a JAX JIT compilation. More values multiply compilation overhead. An extended sweep confirmed rank 32 underperforms rank 2 on EMNIST-Digits, bounding the useful range. For stochastic RL, rank 0 is excluded (fitness noise overwhelms the freeze signal), giving {1, 2, 4}.

**Outcome:** Five compilations total, amortised across all pilot runs. Approximately 45 seconds of compilation overhead.


## Shape-Based Rank Resolution


**Decision:** Resolve rank by weight matrix shape rather than layer position.

**Rationale:** In MLPs, each functional position (input, hidden, output) has a unique shape, so shape serves as a proxy for position. This allows the _get_rank function to operate on parameter shapes without knowing the architecture's topology.

**Outcome:** Works correctly for all tested architectures. Known limitation: breaks for architectures with repeated shapes (e.g., transformer blocks). Position-based resolution is flagged as future work.


## Mean Fitness as Default Pilot Metric


**Decision:** Use mean population fitness throughout the pilot (Phase 1 magnitude, Phase 2 ablation, Phase 3 head-to-head).

**Rationale:** Mean fitness measures the population's collective learning — a reasonable proxy for the deployed model's quality in supervised learning, where the final model is derived from the population mean.

**Outcome:** Correct for supervised learning. Incorrect (or at least suboptimal) for deterministic RL with skewed population distributions, where best fitness is more appropriate. This limitation led to the best-fitness experiment as a discovered finding.


---


# Phase 3: Experimental Programme


## Canonical Allocation (8, 4, 0) on MNIST Family


**Decision:** Standardise on the pilot-derived allocation (input rank 8, hidden rank 4, output rank 0) as the headline result.

**Rationale:** The higher limits of the rank sets were used, with the highest being 8, and the step down from it 4. It was empirically found that results steadily improved until rank 8 ebing the highest. The pilot consistently produces this allocation across all four MNIST-family datasets.

**Outcome:** +1–6pp improvement over vanilla r = 4 across four datasets. Half-budget (4, 2, 0) also outperforms vanilla at half the rank budget.


## Reversed Allocation (0, 4, 8) as Control


**Decision:** Invert the pilot ordering as a principled control.

**Rationale:** Places maximum rank on the least-sensitive layer and zero on the most-sensitive. Demonstrates that the benefit is from the 
*specific* ordering, not from using different ranks per se.

**Outcome:** Worst performance on all datasets, confirming the ordering matters.


## LunarLander: Capped Allocation (4, 2, 1)


**Decision:** Use rank set {1, 2, 4} for stochastic RL (no rank 0).

**Rationale:** Environmental stochasticity prevents reliable identification of insensitive layers. Rank 1 is the safe floor.

**Outcome:** +9.2 reward points over vanilla r = 4 at 42% less rank budget.


## Brax Ant: Six-Method Comparison


**Decision:** Run all six allocations (eggroll_r1, eggroll_r4, lwr_4_0_2, lwr_8_4_0, lwr_4_1_2, lwr_8_1_4) at 300 generations, 3 seeds each.

**Rationale:** Comprehensive comparison including both the pilot-derived allocations and the uniform baselines.

**Outcome:** Uniform r = 1 dominated all methods (76.8 mean fitness). Diagnostic branch correctly identified this as the optimal choice.


## Brax Ant Best-Fitness Experiment


**Decision:** Run the best-fitness pilot ordering (4, 2, 0) and (8, 4, 0) with output-rank controls (4, 2, 1) and (8, 4, 1).

**Rationale:** The confirmation gate surfaced a disagreement between mean and best fitness. The best-fitness pilot produced a different ordering. Three seeds of (4, 2, 0) from the initial exploration showed dramatic results (>2000 fitness vs ~60–76 for all mean-fitness methods).

**Outcome:** Shared-codebase re-runs confirmed: (4, 2, 0) mean best = 2121.2, (8, 4, 0) mean best = 1616.6. Output-rank controls collapse 40–57× when output moves from rank 0 to rank 1. This is the project's most dramatic finding.


---


# Key Technical Insights


- **"Loudly wrong" pattern:** A layer can have high perturbation magnitude (Phase 1) but harmful perturbation direction (Phase 2). The MNIST and Brax Ant output layers both exhibit this — rank reduction or freezing improves performance by removing noise from the gradient estimate.

- **Effective dimensionality determines optimal rank:** When most parameters are in low-sensitivity layers, the useful gradient directions are few. Lower rank concentrates sampling budget on fewer directions with better per-direction estimates. This is why uniform r = 1 beats r = 4 on Brax Ant under mean fitness.

- **Parameter fraction determines freeze safety:** Freezing 1.3% of parameters (MNIST output) is safe; freezing 93.1% (Brax Ant hidden) is catastrophic. The mechanism is the initialisation lottery — frozen weights stay random, and the adaptive layers must compensate.

- **Pilot metric should match deployment objective:** Mean fitness tracks population learning (appropriate for supervised). Best fitness tracks the deployed champion (appropriate for RL). On Brax Ant, this mismatch is the difference between a flatlined search and a 2121.2 best-fitness allocation.

- **The pilot is self-diagnosing:** It either produces a graded allocation that improves performance, or it detects that graded allocation is inappropriate (all-negative Phase 2 → low effective dimensionality → uniform rank 1 fallback). It never makes things worse than vanilla EGGROLL.