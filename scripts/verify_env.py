"""Full environment sanity check. Run: python scripts/verify_env.py"""
import sys
import importlib

REQUIRED = [
    "jax", "jaxlib", "flax", "optax", "evosax",
    "gymnax", "brax", "wandb", "matplotlib",
    "pandas", "seaborn", "rich", "tqdm", "tyro",
]

print(f"Python: {sys.version.split()[0]}")
for name in REQUIRED:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        print(f"  [OK] {name:12s} {v}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

import jax, jax.numpy as jnp
print(f"\nJAX devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")

# GPU compute check
x = jnp.ones((2048, 2048))
y = (x @ x).block_until_ready()
print(f"GPU matmul: shape={y.shape}, device={y.device}")

# EGGROLL import check
try:
    import hyperscalees
    print(f"[OK] hyperscalees imported")
except Exception as e:
    print(f"[FAIL] hyperscalees: {e}")

print("\nAll checks passed if no [FAIL] lines above.")
