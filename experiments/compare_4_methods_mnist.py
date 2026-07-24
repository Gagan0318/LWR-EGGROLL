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
# Sanity: run backprop on seed 0 as a full check
# ============================================================
if __name__ == "__main__":
    print("\n[main] Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    print("\n[main] Running OpenAI-ES, seed=0...")
    result = train_openai_es(0, X_train, y_train, X_test, y_test)
    save_result(result)

    print(f"\n[main] Final: best_test_acc={result['best_test_acc']:.4f}  "
          f"wall_s={result['converged_at_wall_s']:.1f}  "
          f"gen={result['converged_at_step']}  "
          f"reason={result['stop_reason']}")