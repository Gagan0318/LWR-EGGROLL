# ============================================================
# Comparative sweep: Backprop / OpenAI-ES / Sep-CMA-ES / EGGROLL
# ============================================================

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.90"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "cuda_async"
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import jax
import jax.numpy as jnp
from jax import random
import optax
import flax.linen as nn

# --- Directory setup ---
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
DATA_DIR = Path("/home/gagan/dissertation/data/mnist")   # existing cache
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

print(f"[env] JAX {jax.__version__} on {jax.default_backend()}")
print(f"[env] Devices: {jax.devices()}")

# ============================================================
# Config
# ============================================================
@dataclass
class ExperimentConfig:
    # Data
    batch_size: int = 256

    # MLP architecture 
    hidden_dims: tuple = (256, 256, 256)
    n_classes: int = 10
    input_dim: int = 784

    # Convergence-based stopping
    patience: int = 20
    tol: float = 1e-3
    max_wall_seconds: float = 1800
    eval_every_gens: int = 20
    eval_every_steps: int = 100

    # Backprop
    bp_lr: float = 1e-3

    # OpenAI-ES (pop reduced 4096 -> 1024 for VRAM)
    # The paper's Brax/IDP config (σ=0.5, lr=0.1) causes parameter divergence on MNIST — fitness falls to -1.5e11 by gen 420, test acc stuck at chance
    # Fixed by aligning openes_sigma_init to 0.05 and openes_lr_init to 0.01
    openes_pop: int = 1024
    openes_sigma_init: float = 0.05
    openes_sigma_decay: float = 0.9995
    openes_lr_init: float = 0.01
    openes_lr_decay: float = 0.9995

    # Sep-CMA-ES (not in paper; evosax default pop for d ≈ 333k)
    sepcma_pop: int = 40

    # EGGROLL (paper Brax/IDP)
    eggroll_rank: int = 1
    eggroll_pop: int = 2048
    eggroll_sigma_init: float = 0.05
    eggroll_sigma_decay: float = 0.9995
    eggroll_lr_init: float = 0.01
    eggroll_lr_decay: float = 0.9995

    seeds: tuple = (0, 1, 2)

CFG = ExperimentConfig()

# ============================================================
# MNIST loading (via torchvision, cached locally)
# ============================================================
def load_mnist():
    """Load MNIST from the local torchvision cache as JAX arrays.
    Returns: (X_train, y_train, X_test, y_test) all on device.
    """
    from torchvision import datasets

    # torchvision expects root/ to contain the MNIST/ folder
    train = datasets.MNIST(root=str(DATA_DIR), train=True, download=False)
    test = datasets.MNIST(root=str(DATA_DIR), train=False, download=False)

    X_train = np.array(train.data, dtype=np.float32).reshape(-1, 784) / 255.0
    y_train = np.array(train.targets, dtype=np.int32)
    X_test = np.array(test.data, dtype=np.float32).reshape(-1, 784) / 255.0
    y_test = np.array(test.targets, dtype=np.int32)

    # Push to device once, keep there
    X_train = jnp.asarray(X_train)
    y_train = jnp.asarray(y_train)
    X_test = jnp.asarray(X_test)
    y_test = jnp.asarray(y_test)

    print(f"[data] X_train {X_train.shape} {X_train.dtype}")
    print(f"[data] y_train {y_train.shape} {y_train.dtype}")
    print(f"[data] X_test  {X_test.shape} {X_test.dtype}")
    print(f"[data] y_test  {y_test.shape} {y_test.dtype}")
    print(f"[data] pixel range [{X_train.min():.3f}, {X_train.max():.3f}]")

    return X_train, y_train, X_test, y_test

# ============================================================
# MLP: 784 -> 256 -> 256 -> 256 -> 10, ReLU
# ============================================================
class MLP(nn.Module):
    hidden_dims: tuple = (256, 256, 256)
    n_classes: int = 10

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_dims:
            x = nn.Dense(h)(x)
            x = nn.relu(x)
        x = nn.Dense(self.n_classes)(x)
        return x   # logits

def init_params(seed: int):
    """Initialise MLP params. Returns the pytree used as `solution=` template."""
    model = MLP(hidden_dims=CFG.hidden_dims, n_classes=CFG.n_classes)
    key = random.PRNGKey(seed)
    dummy_input = jnp.zeros((1, CFG.input_dim))
    params = model.init(key, dummy_input)["params"]
    return model, params

def count_params(params) -> int:
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
# ============================================================
# Convergence tracker: shared stopping logic for all methods
# ============================================================
class ConvergenceTracker:
    """Tracks test accuracy over training and signals when to stop.

    Stop conditions (either triggers):
      1. `patience` consecutive evaluations with no improvement > `tol`
      2. wall-clock exceeds `max_wall_seconds`

    Usage:
        tracker = ConvergenceTracker(patience=20, tol=1e-3, max_wall_seconds=1800)
        tracker.start()
        for step in range(...):
            # ... training step ...
            if step % eval_every == 0:
                test_acc = evaluate(...)
                converged = tracker.update(step, test_acc)
                if converged:
                    break
        result = tracker.finalise()
    """

    def __init__(self, patience: int, tol: float, max_wall_seconds: float):
        self.patience = patience
        self.tol = tol
        self.max_wall_seconds = max_wall_seconds

        self.history = []          # list of (wall_s, step, test_acc)
        self.best_acc = -float("inf")
        self.best_wall_s = None
        self.best_step = None
        self.stalls = 0            # consecutive evals with no improvement
        self.t0 = None
        self.stop_reason = None    

    def start(self):
        self.t0 = time.perf_counter()

    def update(self, step: int, test_acc: float) -> bool:
        """Log an evaluation and return True if we should stop."""
        assert self.t0 is not None, "Call start() before update()"
        wall_s = time.perf_counter() - self.t0
        self.history.append((wall_s, int(step), float(test_acc)))

        # Improvement check
        if test_acc > self.best_acc + self.tol:
            self.best_acc = float(test_acc)
            self.best_wall_s = wall_s
            self.best_step = int(step)
            self.stalls = 0
        else:
            self.stalls += 1

        # Stop conditions
        if self.stalls >= self.patience:
            self.stop_reason = "patience"
            return True
        if wall_s >= self.max_wall_seconds:
            self.stop_reason = "wall_cap"
            return True
        return False

    def finalise(self) -> dict:
        """Return a serialisable summary of the run."""
        return {
            "history": self.history,
            "best_test_acc": self.best_acc,
            "converged_at_wall_s": self.best_wall_s,
            "converged_at_step": self.best_step,
            "total_wall_s": self.history[-1][0] if self.history else 0.0,
            "n_evals": len(self.history),
            "stop_reason": self.stop_reason or "manual",
            "converged": self.stop_reason == "patience",
        }# ============================================================
# Convergence tracker: shared stopping logic for all methods
# ============================================================
class ConvergenceTracker:
    """Tracks test accuracy over training and signals when to stop.

    Stop conditions (either triggers):
      1. `patience` consecutive evaluations with no improvement > `tol`
      2. wall-clock exceeds `max_wall_seconds`

    Usage:
        tracker = ConvergenceTracker(patience=20, tol=1e-3, max_wall_seconds=1800)
        tracker.start()
        for step in range(...):
            # ... training step ...
            if step % eval_every == 0:
                test_acc = evaluate(...)
                converged = tracker.update(step, test_acc)
                if converged:
                    break
        result = tracker.finalise()
    """

    def __init__(self, patience: int, tol: float, max_wall_seconds: float):
        self.patience = patience
        self.tol = tol
        self.max_wall_seconds = max_wall_seconds

        self.history = []          # list of (wall_s, step, test_acc)
        self.best_acc = -float("inf")
        self.best_wall_s = None
        self.best_step = None
        self.stalls = 0            # consecutive evals with no improvement
        self.t0 = None
        self.stop_reason = None    # "patience" | "wall_cap" | "manual"

    def start(self):
        self.t0 = time.perf_counter()

    def update(self, step: int, test_acc: float) -> bool:
        """Log an evaluation and return True if we should stop."""
        assert self.t0 is not None, "Call start() before update()"
        wall_s = time.perf_counter() - self.t0
        self.history.append((wall_s, int(step), float(test_acc)))

        # Improvement check
        if test_acc > self.best_acc + self.tol:
            self.best_acc = float(test_acc)
            self.best_wall_s = wall_s
            self.best_step = int(step)
            self.stalls = 0
        else:
            self.stalls += 1

        # Stop conditions
        if self.stalls >= self.patience:
            self.stop_reason = "patience"
            return True
        if wall_s >= self.max_wall_seconds:
            self.stop_reason = "wall_cap"
            return True
        return False

    def finalise(self) -> dict:
        """Return a serialisable summary of the run."""
        return {
            "history": self.history,
            "best_test_acc": self.best_acc,
            "converged_at_wall_s": self.best_wall_s,
            "converged_at_step": self.best_step,
            "total_wall_s": self.history[-1][0] if self.history else 0.0,
            "n_evals": len(self.history),
            "stop_reason": self.stop_reason or "manual",
            "converged": self.stop_reason == "patience",
        }
# ============================================================
# Method 1: Backprop with Adam
# ============================================================
def evaluate_test_acc(model, params, X_test, y_test) -> float:
    """Full test-set accuracy. Called every eval_every_steps."""
    logits = model.apply({"params": params}, X_test)
    preds = jnp.argmax(logits, axis=-1)
    return float(jnp.mean(preds == y_test))


def train_backprop(seed: int, X_train, y_train, X_test, y_test) -> dict:
    """Standard supervised training: Adam + cross-entropy on mini-batches.
    Uses ConvergenceTracker for stopping."""
    print(f"\n[backprop seed={seed}] initialising...")

    model, params = init_params(seed=seed)
    optimizer = optax.adam(learning_rate=CFG.bp_lr)
    opt_state = optimizer.init(params)

    def loss_fn(params, X, y):
        logits = model.apply({"params": params}, X)
        # softmax cross-entropy (mean over batch)
        one_hot = jax.nn.one_hot(y, CFG.n_classes)
        log_probs = jax.nn.log_softmax(logits)
        return -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

    @jax.jit
    def train_step(params, opt_state, X_batch, y_batch):
        loss, grads = jax.value_and_grad(loss_fn)(params, X_batch, y_batch)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    tracker = ConvergenceTracker(
        patience=CFG.patience,
        tol=CFG.tol,
        max_wall_seconds=CFG.max_wall_seconds,
    )
    tracker.start()

    key = random.PRNGKey(seed)
    n_train = X_train.shape[0]
    step = 0

    while True:
        # sample a random mini-batch 
        key, subkey = random.split(key)
        idx = random.randint(subkey, (CFG.batch_size,), 0, n_train)
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        params, opt_state, loss = train_step(params, opt_state, X_batch, y_batch)
        step += 1

        if step % CFG.eval_every_steps == 0:
            test_acc = evaluate_test_acc(model, params, X_test, y_test)
            stop = tracker.update(step, test_acc)
            if step % (CFG.eval_every_steps * 5) == 0:
                print(f"[backprop seed={seed}] step={step:6d}  "
                      f"loss={loss:.4f}  test_acc={test_acc:.4f}  "
                      f"wall_s={tracker.history[-1][0]:.1f}")
            if stop:
                print(f"[backprop seed={seed}] STOPPED at step={step} "
                      f"(reason={tracker.stop_reason})")
                break

    result = tracker.finalise()
    result["method"] = "backprop"
    result["seed"] = seed
    result["n_params"] = count_params(params)
    result["hp"] = {
        "lr": CFG.bp_lr,
        "batch_size": CFG.batch_size,
        "optimizer": "adam",
    }
    return result


def save_result(result: dict):
    """Write per-run JSON to results/."""
    fn = RESULTS_DIR / f"{result['method']}_seed{result['seed']}.json"
    with open(fn, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[save] wrote {fn}")
# ============================================================
# Method 2: OpenAI-ES (evosax 0.2.0 API)
# ============================================================
from evosax.algorithms import Open_ES

def _make_fitness_fn(model, X_train, y_train):
    """Returns a jitted fn: (params_batch, X_batch, y_batch) -> fitness_batch.
    Fitness = negative mean cross-entropy (ES maximises)."""

    def per_member_fitness(params, X_batch, y_batch):
        logits = model.apply({"params": params}, X_batch)
        one_hot = jax.nn.one_hot(y_batch, CFG.n_classes)
        log_probs = jax.nn.log_softmax(logits)
        ce = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
        return ce   # evosax 0.2.0 minimises; return CE as is

    # vmap over the population axis
    return jax.jit(jax.vmap(per_member_fitness, in_axes=(0, None, None)))


def train_openai_es(seed: int, X_train, y_train, X_test, y_test) -> dict:
    """OpenAI-ES on MLP with paper's IDP hyperparameters (pop reduced 4096 -> 1024)."""
    print(f"\n[openai_es seed={seed}] initialising...")

    model, params_template = init_params(seed=seed)

    # optax exponential decay for learning rate 
    lr_schedule = optax.exponential_decay(
        init_value=CFG.openes_lr_init,
        transition_steps=1,
        decay_rate=CFG.openes_lr_decay,
    )
    optimizer = optax.adam(learning_rate=lr_schedule)

    # sigma schedule: matches paper's sigma_decay=0.9995 per gen
    sigma_init = CFG.openes_sigma_init
    sigma_decay = CFG.openes_sigma_decay
    std_schedule = lambda gen: sigma_init * (sigma_decay ** gen)

    strategy = Open_ES(
        population_size=CFG.openes_pop,
        solution=params_template,
        optimizer=optimizer,
        std_schedule=std_schedule,
    )

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    default_params = strategy.default_params
    state = strategy.init(subkey, params_template, default_params)
    fitness_fn = _make_fitness_fn(model, X_train, y_train)

    tracker = ConvergenceTracker(
        patience=CFG.patience,
        tol=CFG.tol,
        max_wall_seconds=CFG.max_wall_seconds,
    )
    tracker.start()

    n_train = X_train.shape[0]
    gen = 0

    while True:
        # sample a fresh mini-batch shared across the whole population
        key, batch_key, ask_key, tell_key = random.split(key, 4)
        idx = random.randint(batch_key, (CFG.batch_size,), 0, n_train)
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        # ask: sample the population
        population, state = strategy.ask(ask_key, state, default_params)

        # evaluate: fitness for each population member on the shared batch
        fitness = fitness_fn(population, X_batch, y_batch)

        # tell: update strategy state
        state, metrics = strategy.tell(tell_key, population, fitness, state, default_params)

        gen += 1

        # periodic evaluation of the mean parameters on full test set
        if gen % CFG.eval_every_gens == 0:
            mean_params = strategy.get_mean(state)
            test_acc = evaluate_test_acc(model, mean_params, X_test, y_test)
            stop = tracker.update(gen, test_acc)
            print(f"[openai_es seed={seed}] gen={gen:5d}  "
                  f"fit_min={float(fitness.min()):.4f}  "
                  f"fit_mean={float(fitness.mean()):.4f}  "
                  f"test_acc={test_acc:.4f}  "
                  f"sigma={float(std_schedule(gen)):.4f}  "
                  f"wall_s={tracker.history[-1][0]:.1f}")
            if stop:
                print(f"[openai_es seed={seed}] STOPPED at gen={gen} "
                      f"(reason={tracker.stop_reason})")
                break

    result = tracker.finalise()
    result["method"] = "openai_es"
    result["seed"] = seed
    result["n_params"] = count_params(params_template)
    result["hp"] = {
        "population_size": CFG.openes_pop,
        "sigma_init": CFG.openes_sigma_init,
        "sigma_decay": CFG.openes_sigma_decay,
        "lr_init": CFG.openes_lr_init,
        "lr_decay": CFG.openes_lr_decay,
        "optimizer": "adam",
        "batch_size": CFG.batch_size,
    }
    return result
# ============================================================
# Method 3: Sep-CMA-ES (evosax 0.2.0)
# ============================================================
from evosax.algorithms import Sep_CMA_ES

def train_sep_cma_es(seed: int, X_train, y_train, X_test, y_test) -> dict:
    """Sep-CMA-ES on the MNIST MLP.

    Sep-CMA-ES uses a diagonal covariance matrix (as opposed to full CMA-ES,
    which stores an O(d²) covariance and is infeasible at d=335k). It adapts
    both σ and the diagonal covariance internally via evolution paths —
    no external optimiser or std schedule to configure.
    """
    print(f"\n[sep_cma_es seed={seed}] initialising...")

    model, params_template = init_params(seed=seed)

    strategy = Sep_CMA_ES(
        population_size=CFG.sepcma_pop,
        solution=params_template,
    )

    key = random.PRNGKey(seed)
    key, subkey = random.split(key)
    default_params = strategy.default_params
    state = strategy.init(subkey, params_template, default_params)
    fitness_fn = _make_fitness_fn(model, X_train, y_train)

    tracker = ConvergenceTracker(
        patience=CFG.patience,
        tol=CFG.tol,
        max_wall_seconds=CFG.max_wall_seconds,
    )
    tracker.start()

    n_train = X_train.shape[0]
    gen = 0

    while True:
        key, batch_key, ask_key, tell_key = random.split(key, 4)
        idx = random.randint(batch_key, (CFG.batch_size,), 0, n_train)
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        population, state = strategy.ask(ask_key, state, default_params)
        fitness = fitness_fn(population, X_batch, y_batch)
        state, metrics = strategy.tell(tell_key, population, fitness, state, default_params)

        gen += 1

        if gen % CFG.eval_every_gens == 0:
            mean_params = strategy.get_mean(state)
            test_acc = evaluate_test_acc(model, mean_params, X_test, y_test)
            stop = tracker.update(gen, test_acc)
            print(f"[sep_cma_es seed={seed}] gen={gen:5d}  "
                  f"fit_min={float(fitness.min()):.4f}  "
                  f"fit_mean={float(fitness.mean()):.4f}  "
                  f"test_acc={test_acc:.4f}  "
                  f"wall_s={tracker.history[-1][0]:.1f}")
            if stop:
                print(f"[sep_cma_es seed={seed}] STOPPED at gen={gen} "
                      f"(reason={tracker.stop_reason})")
                break

    result = tracker.finalise()
    result["method"] = "sep_cma_es"
    result["seed"] = seed
    result["n_params"] = count_params(params_template)
    result["hp"] = {
        "population_size": CFG.sepcma_pop,
        "batch_size": CFG.batch_size,
        "note": "Sep-CMA-ES adapts σ and diagonal covariance internally; no external optimiser or σ schedule.",
    }
    return result

# ============================================================
# Method 4: EGGROLL (HyperscaleES, rank=1)
# ============================================================
import hyperscalees as hs

def train_eggroll(seed: int, X_train, y_train, X_test, y_test, rank:int=None) -> dict:
    """EGGROLL on a 3L-256D ReLU MLP built via HyperscaleES's own MLP class.

    Different API from evosax: population is expressed by vmap'ing over
    thread_ids, and low-rank noise is deterministically reconstructed from
    the key + epoch + thread_id rather than stored. Antithetic sampling is
    built into the noiser (pop_size effective = num_envs, with num_envs/2
    unique noise directions, mirrored).

    Note: HyperscaleES maximises fitness (unlike evosax 0.2.0 which
    minimises). Fitness returned as -CE.
    """
    if rank is None:
        rank = CFG.eggroll_rank
    print(f"\n[eggroll rank={rank} seed={seed}] initialising...")

    NOISER = hs.noiser.eggroll.EggRoll
    MODEL = hs.models.common.MLP

    key = jax.random.key(seed)
    model_key = random.fold_in(key, 0)
    es_key = random.fold_in(key, 1)
    data_key = random.fold_in(key, 2)

    # Build the same 3L-256D architecture, but via HyperscaleES's MLP class.
    # in_dim=784 (flattened MNIST), out_dim=10 (classes), hidden=[256, 256, 256].
    # Uses ReLU to match Flax MLP used for backprop / OpenAI-ES / Sep-CMA-ES.
    frozen_params, params, scan_map, es_map = MODEL.rand_init(
        model_key,
        in_dim=CFG.input_dim,
        out_dim=CFG.n_classes,
        hidden_dims=[256, 256, 256],
        use_bias=True,
        activation="relu",
        dtype="float32",
    )
    es_tree_key = hs.models.common.simple_es_tree_key(params, es_key, scan_map)

    # Report parameter count for sanity
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"[eggroll rank={rank} seed={seed}] param count: {n_params:,}")

    # Initialise EggRoll: paper's Brax/IDP config
    # rank=1 (paper's headline claim), sigma=0.05, lr=0.01, Adam
    frozen_noiser_params, noiser_params = NOISER.init_noiser(
        params,
        sigma=CFG.eggroll_sigma_init,
        lr=CFG.eggroll_lr_init,
        solver=optax.adamw,
        solver_kwargs={"b1": 0.9, "b2": 0.999},
        rank=rank,
    )

    # ------------------------------------------------------------
    # Forward passes: training (with noise) and evaluation (mean params only)
    # ------------------------------------------------------------
    # Training: vmap over population (thread_ids). For fair comparison with
    # OpenAI-ES/Sep-CMA-ES, ALL members see the SAME mini-batch — so we
    # vmap over threads but keep the batch fixed. Then for each member,
    # we further vmap the model over the batch dimension.
    #
    # Model forward signature: (noiser, frozen_np, np, frozen_p, p, es_tree_key, iterinfo, x)
    # Batch fitness: mean cross-entropy across the mini-batch.

    # noiser_params and params are ARGUMENTS, not closure vars, so JIT traces them fresh each time.
    def per_member_batch_ce(np_current, p_current, iterinfo_i, X_batch, y_batch):
        def forward_one(x):
            return MODEL.forward(
                NOISER, frozen_noiser_params, np_current,
                frozen_params, p_current, es_tree_key, iterinfo_i, x
            )
        logits_batch = jax.vmap(forward_one)(X_batch)
        one_hot = jax.nn.one_hot(y_batch, CFG.n_classes)
        log_probs = jax.nn.log_softmax(logits_batch)
        ce = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))
        return -ce

    # vmap only over iterinfo_i (population axis). np_current, p_current,
    # X_batch, y_batch are shared across the population.
    jit_pop_fitness = jax.jit(jax.vmap(
        per_member_batch_ce, in_axes=(None, None, 0, None, None)
    ))

    # Evaluation: no noiser perturbation, just the mean params on the full test set.
    def eval_one(x):
        return MODEL.forward(
            NOISER, frozen_noiser_params, noiser_params,
            frozen_params, params, es_tree_key, None, x
        )
    jit_eval_batch = jax.jit(jax.vmap(eval_one))

    def eval_batch(np_current, p_current, X):
        def eval_one(x):
            return MODEL.forward(
                NOISER, frozen_noiser_params, np_current,
                frozen_params, p_current, es_tree_key, None, x
            )
        return jax.vmap(eval_one)(X)
    jit_eval_batch = jax.jit(eval_batch)

    def evaluate_mean_test_acc(np_current, p_current):
        logits = jit_eval_batch(np_current, p_current, X_test)
        return float(jnp.mean(jnp.argmax(logits, axis=-1) == y_test))

    # Update step (matches end-to-end test)
    jit_update = jax.jit(lambda n, p, f, i: NOISER.do_updates(
        frozen_noiser_params, n, p, es_tree_key, f, i, es_map
    ))

    # ------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------
    tracker = ConvergenceTracker(
        patience=CFG.patience,
        tol=CFG.tol,
        max_wall_seconds=CFG.max_wall_seconds,
    )
    tracker.start()

    n_train = X_train.shape[0]
    num_envs = CFG.eggroll_pop
    gen = 0

    while True:
        data_key, batch_key = random.split(data_key)
        idx = random.randint(batch_key, (CFG.batch_size,), 0, n_train)
        X_batch = X_train[idx]
        y_batch = y_train[idx]

        # iterinfo = (epoch broadcasted to pop, arange for thread_ids)
        iterinfo = (
            jnp.full(num_envs, gen, dtype=jnp.int32),
            jnp.arange(num_envs, dtype=jnp.int32),
        )

        # Per-member fitness (population size = num_envs)
        raw_fitness = jit_pop_fitness(noiser_params, params, iterinfo, X_batch, y_batch)
        # EGGROLL's own fitness shaping (rank-based, similar to OpenAI-ES centred ranks)
        fitness = NOISER.convert_fitnesses(frozen_noiser_params, noiser_params, raw_fitness)

        # Update params
        noiser_params, params = jit_update(noiser_params, params, fitness, iterinfo)

        gen += 1

        if gen % CFG.eval_every_gens == 0:
            test_acc = evaluate_mean_test_acc(noiser_params, params)
            stop = tracker.update(gen, test_acc)
            print(f"[eggroll rank={rank} seed={seed}] gen={gen:5d}  "
                  f"raw_fit_max={float(raw_fitness.max()):.4f}  "
                  f"raw_fit_mean={float(raw_fitness.mean()):.4f}  "
                  f"test_acc={test_acc:.4f}  "
                  f"wall_s={tracker.history[-1][0]:.1f}")
            if stop:
                print(f"[eggroll rank={rank}seed={seed}] STOPPED at gen={gen} "
                      f"(reason={tracker.stop_reason})")
                break

    result = tracker.finalise()
    result["method"] = f"eggroll_r{rank}"
    result["seed"] = seed
    result["n_params"] = n_params
    result["hp"] = {
        "rank": rank,
        "population_size": CFG.eggroll_pop,
        "sigma_init": CFG.eggroll_sigma_init,
        "sigma_decay": CFG.eggroll_sigma_decay,
        "lr_init": CFG.eggroll_lr_init,
        "lr_decay": CFG.eggroll_lr_decay,
        "optimizer": "adamw",
        "batch_size": CFG.batch_size,
        "note": "Uses HyperscaleES's own MLP (ReLU, 3L-256D). Matches Flax architecture. Antithetic sampling built into noiser.",
    }
    return result
# ============================================================
# All 4 methods × 3 seeds, plus EGGROLL rank sweep × 3 seeds
# ============================================================
if __name__ == "__main__":
    import time as _time
    sweep_start = _time.perf_counter()
    
    print("\n" + "="*70)
    print("MULTI-SEED SWEEP: 4 methods + EGGROLL rank sweep, 3 seeds each")
    print("="*70)
    
    print("\n[main] Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()
    
    seeds = CFG.seeds   # (0, 1, 2)
    all_results = []
    
    # -------- Backprop --------
    print("\n" + "="*70)
    print("METHOD: Backprop + Adam")
    print("="*70)
    for seed in seeds:
        print(f"\n>>> Backprop seed={seed}")
        result = train_backprop(seed, X_train, y_train, X_test, y_test)
        save_result(result)
        all_results.append(result)
    
    # -------- OpenAI-ES --------
    print("\n" + "="*70)
    print("METHOD: OpenAI-ES")
    print("="*70)
    for seed in seeds:
        print(f"\n>>> OpenAI-ES seed={seed}")
        result = train_openai_es(seed, X_train, y_train, X_test, y_test)
        save_result(result)
        all_results.append(result)
    
    # -------- Sep-CMA-ES --------
    print("\n" + "="*70)
    print("METHOD: Sep-CMA-ES")
    print("="*70)
    for seed in seeds:
        print(f"\n>>> Sep-CMA-ES seed={seed}")
        result = train_sep_cma_es(seed, X_train, y_train, X_test, y_test)
        save_result(result)
        all_results.append(result)
    
    # -------- EGGROLL rank sweep --------
    print("\n" + "="*70)
    print("METHOD: EGGROLL (ranks 1, 4, 16)")
    print("="*70)
    for rank in [1, 4, 16]:
        for seed in seeds:
            print(f"\n>>> EGGROLL rank={rank} seed={seed}")
            result = train_eggroll(seed, X_train, y_train, X_test, y_test, rank=rank)
            save_result(result)
            all_results.append(result)
    
    # -------- Summary --------
    total_wall = _time.perf_counter() - sweep_start
    print("\n" + "="*70)
    print(f"SWEEP COMPLETE — total wall-clock: {total_wall/60:.1f} minutes")
    print("="*70)
    print(f"\n{'method':<20} {'seed':<6} {'best_acc':<10} {'wall_s':<10} {'stop':<10}")
    print("-" * 70)
    for r in all_results:
        method = r['method']
        seed = r['seed']
        acc = r['best_test_acc']
        wall = r.get('converged_at_wall_s') or r.get('total_wall_s') or 0
        stop = r['stop_reason']
        print(f"{method:<20} {seed:<6} {acc:<10.4f} {wall:<10.1f} {stop:<10}")
    
    # Aggregate stats per method (mean ± std across seeds)
    print("\n" + "="*70)
    print("AGGREGATED (mean ± std across seeds)")
    print("="*70)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in all_results:
        grouped[r['method']].append(r)
    
    print(f"{'method':<20} {'best_acc (mean±std)':<25} {'wall_s (mean±std)':<25}")
    print("-" * 70)
    for method, runs in grouped.items():
        accs = [r['best_test_acc'] for r in runs]
        walls = [r.get('converged_at_wall_s') or r.get('total_wall_s') or 0 for r in runs]
        acc_str = f"{np.mean(accs):.4f} ± {np.std(accs):.4f}"
        wall_str = f"{np.mean(walls):.1f} ± {np.std(walls):.1f}"
        print(f"{method:<20} {acc_str:<25} {wall_str:<25}")
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("Next: run scripts/plot_comparison.py (to be written next)")