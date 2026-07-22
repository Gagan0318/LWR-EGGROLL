"""
Template for a new experiment. Copy this file, rename it, and edit.

Usage: python experiments/<name>.py
"""

import os
import json
import time
import jax
from huggingface_hub.constants import HF_HOME

# --- JAX cache setup (do this before importing anything heavy) ---
jax.config.update("jax_compilation_cache_dir", os.path.join(HF_HOME, "hyperscaleescomp"))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import optax
import jax.numpy as jnp
import hyperscalees as hs


# --- Config ---
CONFIG = {
    "experiment_name": "template",
    "seed": 0,
    # add hyperparameters here
}


# --- Main experiment logic ---
def run(cfg):
    """Run the experiment, return a results dict."""
    # your code goes here
    return {}


# --- Save results ---
def save_results(cfg, results):
    output = {
        "config": cfg,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    results_path = os.path.expanduser(
        f"~/dissertation/eggroll-diss/results/{cfg['experiment_name']}.json"
    )
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    print(f"=== {CONFIG['experiment_name']} ===", flush=True)
    t0 = time.time()
    results = run(CONFIG)
    save_results(CONFIG, results)
    print(f"Total time: {time.time() - t0:.1f}s", flush=True)
