"""Brax Ant — Full LWR-EGGROLL experiment.

Architecture: MLP [27, 256, 256, 256, 8]
Layer shapes: {input: (256,27), hidden: (256,256) x2, output: (8,256)}

Six methods: eggroll_r4, eggroll_r1, lwr_{pilot}, lwr_8_4_0, lwr_4_1_2, lwr_8_1_4
Three seeds each, 300 generations (eggroll_r1 also at 300).

Crash recovery via 10-min checkpoints on Google Drive.
"""


import ast, json, pickle, time
from pathlib import Path

import cloudpickle
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import envs

import hyperscalees as hs
from hyperscalees.noiser.eggroll import EggRoll
from hyperscalees.noiser.lwr_eggroll import LWREggRoll

# ── Config ────────────────────────────────────────────────────────
ENV_NAME      = "ant"
EPISODE_LENGTH = 1000
POP_SIZE      = 2048
SIGMA         = 0.2
LR            = 0.1
SIGMA_DECAY   = 0.999
LR_DECAY      = 0.9995
RANK          = 4
N_LAYERS      = 3
LAYER_SIZE    = 256
ACTIVATION    = "pqn"
OPTIMIZER     = optax.sgd
VMAP_CHUNK    = 256          # Lower to 128/64 if OOM on T4
MAX_GENS      = 300
SEEDS         = [0, 1, 2]
NAN_REPLACEMENT = -1000.0
CHECKPOINT_INTERVAL_S = 600

PHASE1_CHECKPOINT_GENS = 25
PHASE1_ELEVATION_GENS  = 10
PHASE2_GENS            = 15
PHASE3_GENS            = 15
PILOT_SEEDS            = [0, 1]

LAYER_SHAPES = {
    "input":  (256, 27),
    "hidden": (256, 256),
    "output": (8, 256),
}
MODEL = hs.models.common.MLP

# ── Results on Drive (absolute paths, NO symlinks) ────────────────
DRIVE_DIR   = Path("/content/drive/MyDrive/dissertation/results/brax_ant")
CACHE_DIR   = DRIVE_DIR / "cache"
DRIVE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Environment ───────────────────────────────────────────────────
def make_env():
    return envs.get_environment(ENV_NAME)

_env_dims = None
def get_dims():
    global _env_dims
    if _env_dims is None:
        env = make_env()
        _env_dims = (env.observation_size, env.action_size)
    return _env_dims

OBS_DIM, ACT_DIM = get_dims()

# ── PRNG key compat (legacy ↔ typed) ─────────────────────────────
def _safe_wrap_key(leaf):
    if hasattr(leaf, "ndim") and leaf.ndim >= 1:
        return jax.random.wrap_key_data(leaf)
    return leaf

# ── Checkpoint I/O ────────────────────────────────────────────────
def _to_np(x):
    if hasattr(x, 'dtype') and hasattr(x, 'shape') and not isinstance(x, np.ndarray):
        try: return np.asarray(x)
        except: return x
    return x

def _to_jax(x):
    return jnp.asarray(x) if isinstance(x, np.ndarray) else x

def save_pkl(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'wb') as f:
        cloudpickle.dump(jax.tree_util.tree_map(_to_np, data), f,
                         protocol=pickle.HIGHEST_PROTOCOL)
    tmp.rename(path)

def load_pkl(path):
    path = Path(path)
    if not path.exists(): return None
    try:
        with open(path, 'rb') as f:
            return jax.tree_util.tree_map(_to_jax, cloudpickle.load(f))
    except Exception as e:
        print(f"  WARNING: corrupt checkpoint {path}: {e}")
        return None

def save_json(path, data):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    tmp.rename(path)

def load_json(path):
    path = Path(path)
    if not path.exists(): return None
    try:
        with open(path) as f: return json.load(f)
    except: return None

# ── Model + Noiser init ──────────────────────────────────────────
def init_model(key):
    return MODEL.rand_init(key, in_dim=OBS_DIM, out_dim=ACT_DIM,
                           hidden_dims=[LAYER_SIZE] * N_LAYERS,
                           use_bias=True, activation=ACTIVATION, dtype="float32")

def init_noiser(params, noiser_class, rank_spec, sigma=SIGMA, lr=LR):
    return noiser_class.init_noiser(params, sigma, lr,
                                    solver=OPTIMIZER, solver_kwargs={},
                                    rank=rank_spec)

# ── Training loop (vmapped population evaluation) ────────────────
def train(seed, noiser_class, rank_spec, label, max_gens=MAX_GENS,
          sigma=SIGMA, lr=LR, sigma_decay=SIGMA_DECAY, lr_decay=LR_DECAY,
          initial_state=None, return_checkpoint_at=None):
    NOISER = noiser_class
    env = make_env()

    ckpt_file = CACHE_DIR / f"train_{label}_seed{seed}.pkl"
    resumed_gen, history, best_fitness = 0, [], -float("inf")

    existing = load_pkl(ckpt_file)
    if existing is not None and initial_state is None:
        resumed_gen = existing['gen'] + 1
        history = existing['history']
        best_fitness = existing['best_fitness']
        fp = existing['frozen_params']
        cp = existing['params']
        sm, em = existing['scan_map'], existing['es_map']
        estk = jax.tree.map(_safe_wrap_key, existing['es_tree_key'])
        fnp = existing['frozen_noiser_params']
        cnp = existing['noiser_params']
        ep_key = existing['episode_key']
        print(f"  [{label} s{seed}] RESUME gen {resumed_gen} (best={best_fitness:.1f})",
              flush=True)
    else:
        if initial_state is not None:
            fp = initial_state['frozen_params']
            cp = initial_state['params']
            sm, em = initial_state['scan_map'], initial_state['es_map']
            estk = jax.tree.map(_safe_wrap_key, initial_state['es_tree_key'])
            fnp, cnp = init_noiser(cp, noiser_class, rank_spec, sigma=sigma, lr=lr)
        else:
            key = jax.random.PRNGKey(seed)
            key, mk, ek = jax.random.split(key, 3)
            fp, cp, sm, em = init_model(mk)
            estk = jax.tree.map(_safe_wrap_key,
                   hs.models.common.simple_es_tree_key(cp, ek, sm))
            fnp, cnp = init_noiser(cp, noiser_class, rank_spec, sigma=sigma, lr=lr)
        ep_key = jax.random.PRNGKey(seed + 10000)
        print(f"  [{label} s{seed}] init...", flush=True)

    n_params = sum(x.size for x in jax.tree_util.tree_leaves(cp))
    print(f"  [{label} s{seed}] {n_params:,} params", flush=True)

    # Vmapped evaluator (closures capture fnp, fp, estk — not JAX-traceable)
    all_mids = jnp.arange(POP_SIZE, dtype=jnp.int32)

    def _make_eval(fnp_, fp_, estk_):
        def _single(gen_i, mid_i, ep_k, cnp_, cp_):
            iterinfo = (gen_i, mid_i)
            def policy(obs):
                return jnp.clip(
                    MODEL.forward(NOISER, fnp_, cnp_, fp_, cp_, estk_, iterinfo, obs),
                    -1.0, 1.0)
            state = env.reset(ep_k)
            def step_fn(carry, _):
                st, tr, done = carry
                ns = env.step(st, policy(st.obs))
                r = ns.reward * (1.0 - done)
                done = jnp.logical_or(done, ns.done)
                return (ns, tr + r, done), None
            (_, tr, _), _ = jax.lax.scan(
                step_fn, (state, jnp.float32(0.0), jnp.bool_(False)),
                None, length=EPISODE_LENGTH)
            return tr
        return jax.jit(jax.vmap(_single, in_axes=(None, 0, 0, None, None)))

    eval_vmap = _make_eval(fnp, fp, estk)
    chunk = VMAP_CHUNK
    checkpoint_state = None
    t0 = time.time()
    last_ckpt = time.time()

    for gen in range(resumed_gen, max_gens):
        tg = time.time()
        keys = jax.random.split(ep_key, POP_SIZE + 1)
        ep_key = keys[0]
        mk = keys[1:]
        gi = jnp.int32(gen)

        # Evaluate with auto OOM halving
        try:
            parts = []
            for s in range(0, POP_SIZE, chunk):
                e = min(s + chunk, POP_SIZE)
                parts.append(eval_vmap(gi, all_mids[s:e], mk[s:e], cnp, cp))
            fits = jnp.concatenate(parts)
        except Exception as ex:
            if "out of memory" in str(ex).lower() or "resource_exhausted" in str(ex).lower():
                chunk = max(8, chunk // 2)
                print(f"  OOM → chunk={chunk}", flush=True)
                parts = []
                for s in range(0, POP_SIZE, chunk):
                    e = min(s + chunk, POP_SIZE)
                    parts.append(eval_vmap(gi, all_mids[s:e], mk[s:e], cnp, cp))
                fits = jnp.concatenate(parts)
            else:
                raise

        # NaN handling
        nans = int(jnp.sum(jnp.isnan(fits)))
        if nans > 0:
            fits = jnp.where(jnp.isnan(fits), NAN_REPLACEMENT, fits)

        gm = float(jnp.mean(fits))
        gb = float(jnp.max(fits))
        gv = float(jnp.var(fits))
        if gb > best_fitness:
            best_fitness = gb

        history.append({"gen": gen, "mean_fitness": gm, "best_fitness": gb,
                        "best_so_far": best_fitness, "fitness_variance": gv,
                        "nan_count": nans, "wall_s": time.time() - t0})

        # ES update
        iterinfo = (jnp.full(POP_SIZE, gen, dtype=jnp.int32), jnp.arange(POP_SIZE))
        converted = NOISER.convert_fitnesses(fnp, cnp, fits)
        cnp, cp = NOISER.do_updates(fnp, cnp, cp, estk, converted, iterinfo, em)

        if sigma_decay < 1.0:
            cnp['sigma'] = cnp['sigma'] * sigma_decay
        if lr_decay < 1.0 and 'lr' in cnp:
            cnp['lr'] = cnp['lr'] * lr_decay

        if return_checkpoint_at is not None and gen == return_checkpoint_at:
            checkpoint_state = {'frozen_params': fp, 'params': cp, 'scan_map': sm,
                                'es_map': em, 'es_tree_key': estk,
                                'frozen_noiser_params': fnp, 'noiser_params': cnp}

        # Crash checkpoint
        now = time.time()
        if now - last_ckpt >= CHECKPOINT_INTERVAL_S:
            save_pkl(ckpt_file, {
                'gen': gen, 'history': history, 'best_fitness': best_fitness,
                'frozen_params': fp, 'params': cp, 'scan_map': sm, 'es_map': em,
                'es_tree_key': estk, 'frozen_noiser_params': fnp,
                'noiser_params': cnp, 'episode_key': ep_key})
            last_ckpt = now

        if gen % 10 == 0 or gen == max_gens - 1:
            print(f"  gen {gen:4d}  mean={gm:.1f}  best={best_fitness:.1f}"
                  f"  σ={float(cnp['sigma']):.4f}  ({time.time()-tg:.1f}s)", flush=True)

    total = time.time() - t0
    fm = history[-1]["mean_fitness"] if history else 0.0
    print(f"  [{label} s{seed}] done {total:.0f}s, best={best_fitness:.1f}", flush=True)

    if ckpt_file.exists():
        ckpt_file.unlink()

    result = {"method": label, "seed": seed, "best_fitness": best_fitness,
              "final_mean_fitness": fm, "generations": len(history),
              "wall_seconds": total, "history": history}
    return (result, checkpoint_state) if return_checkpoint_at else result


# ── Pilot phases ──────────────────────────────────────────────────
def run_phase1():
    print(f"\n{'='*60}\nPHASE 1 — Elevation / Magnitude\n{'='*60}")
    shapes = list(LAYER_SHAPES.items())
    results = {}
    for seed in PILOT_SEEDS:
        cc = CACHE_DIR / f"p1_checkpoint_seed{seed}.json"
        cs = CACHE_DIR / f"p1_checkpoint_state_seed{seed}.pkl"
        cached = load_json(cc)
        if cached and load_pkl(cs):
            ckpt_fit = cached["checkpoint_fitness"]
            ckpt_state = load_pkl(cs)
            print(f"  Checkpoint s{seed} cached (fit={ckpt_fit:.1f})")
        else:
            r, ckpt_state = train(seed, EggRoll, RANK, "phase1_checkpoint",
                                  max_gens=PHASE1_CHECKPOINT_GENS,
                                  return_checkpoint_at=PHASE1_CHECKPOINT_GENS - 1)
            ckpt_fit = r["final_mean_fitness"]
            save_json(cc, {"checkpoint_fitness": ckpt_fit})
            save_pkl(cs, ckpt_state)
        for ln, ls in shapes:
            ec = CACHE_DIR / f"p1_elevate_{ln}_seed{seed}.json"
            cached_e = load_json(ec)
            if cached_e:
                results[f"{ln}_seed{seed}"] = cached_e
                continue
            spec = {s: RANK for s in LAYER_SHAPES.values()}
            spec[ls] = 8
            er = train(seed, LWREggRoll, spec, f"phase1_elevate_{ln}",
                       max_gens=PHASE1_ELEVATION_GENS, initial_state=ckpt_state)
            delta = abs(er["final_mean_fitness"] - ckpt_fit)
            entry = {"layer": ln, "seed": seed, "checkpoint_fitness": ckpt_fit,
                     "elevated_fitness": er["final_mean_fitness"], "abs_delta": delta}
            results[f"{ln}_seed{seed}"] = entry
            save_json(ec, entry)
    summary = {}
    for ln, _ in shapes:
        deltas = [results[f"{ln}_seed{s}"]["abs_delta"]
                  for s in PILOT_SEEDS if f"{ln}_seed{s}" in results]
        summary[ln] = {"mean_abs_delta": float(np.mean(deltas)), "deltas": deltas}
    ordering = sorted(summary, key=lambda n: summary[n]["mean_abs_delta"], reverse=True)
    ord_str = " > ".join(ordering)
    save_json(CACHE_DIR / "phase1_complete.json", {"ordering": ord_str, "summary": summary})
    print(f"  Phase 1: {ord_str}")
    return ord_str, summary

def run_phase2():
    print(f"\n{'='*60}\nPHASE 2 — Causal Ablation\n{'='*60}")
    shapes = list(LAYER_SHAPES.items())
    baselines = []
    for seed in PILOT_SEEDS:
        bc = CACHE_DIR / f"p2_baseline_seed{seed}.json"
        cached = load_json(bc)
        if cached:
            baselines.append(cached["final_mean_fitness"])
        else:
            r = train(seed, EggRoll, RANK, f"pilot_baseline_r{RANK}",
                      max_gens=PHASE2_GENS)
            baselines.append(r["final_mean_fitness"])
            save_json(bc, {"final_mean_fitness": r["final_mean_fitness"]})
    bl_mean = float(np.mean(baselines))
    degradations = {}
    for ln, ls in shapes:
        abl = []
        for seed in PILOT_SEEDS:
            ac = CACHE_DIR / f"p2_ablate_{ln}_seed{seed}.json"
            cached = load_json(ac)
            if cached:
                abl.append(cached["final_mean_fitness"])
            else:
                spec = {s: RANK for s in LAYER_SHAPES.values()}
                spec[ls] = 1
                r = train(seed, LWREggRoll, spec, f"pilot_ablate_{ln}",
                          max_gens=PHASE2_GENS)
                abl.append(r["final_mean_fitness"])
                save_json(ac, {"final_mean_fitness": r["final_mean_fitness"]})
        am = float(np.mean(abl))
        degradations[ln] = {"shape": ls, "mean_fitness": am, "degradation": bl_mean - am}
    ordering = sorted(degradations, key=lambda n: degradations[n]["degradation"],
                      reverse=True)
    ord_str = " > ".join(ordering)
    save_json(CACHE_DIR / "phase2_complete.json", {
        "baseline_mean": bl_mean,
        "degradations": {k: {**v, "shape": str(v["shape"])} for k, v in degradations.items()},
        "ordering": ord_str})
    print(f"  Phase 2: {ord_str}")
    return ordering, degradations, bl_mean

def run_phase3(ordering, degradations):
    ls_name = ordering[-1]
    ls_shape = degradations[ls_name]["shape"]
    print(f"\n{'='*60}\nPHASE 3 — Binary Inclusion ({ls_name})\n{'='*60}")
    r1_mean = degradations[ls_name]["mean_fitness"]
    frozen = []
    for seed in PILOT_SEEDS:
        pc = CACHE_DIR / f"p3_freeze_{ls_name}_seed{seed}.json"
        cached = load_json(pc)
        if cached:
            frozen.append(cached["final_mean_fitness"])
        else:
            spec = {s: RANK for s in LAYER_SHAPES.values()}
            spec[ls_shape] = 0
            r = train(seed, LWREggRoll, spec, f"pilot_freeze_{ls_name}",
                      max_gens=PHASE3_GENS)
            frozen.append(r["final_mean_fitness"])
            save_json(pc, {"final_mean_fitness": r["final_mean_fitness"]})
    r0_mean = float(np.mean(frozen))
    decision = 0 if r0_mean > r1_mean else 1
    save_json(CACHE_DIR / "phase3_complete.json", {
        "least_sensitive": ls_name, "r1_mean": r1_mean,
        "r0_mean": r0_mean, "decision": decision})
    print(f"  r=1: {r1_mean:.1f}, r=0: {r0_mean:.1f} → rank {decision}")
    return ls_name, decision, r0_mean, r1_mean


def run_pilot():
    """Run or load all 3 phases, return allocations for all methods."""
    p1c = load_json(CACHE_DIR / "phase1_complete.json")
    if p1c:
        p1_ord, p1_sum = p1c["ordering"], p1c["summary"]
        print(f"  Phase 1 cached: {p1_ord}")
    else:
        p1_ord, p1_sum = run_phase1()

    p2c = load_json(CACHE_DIR / "phase2_complete.json")
    if p2c:
        ordering_str = p2c["ordering"]
        ordering = ordering_str.split(" > ")
        bl_mean = p2c["baseline_mean"]
        degradations = {}
        for k, v in p2c["degradations"].items():
            degradations[k] = {**v, "shape": ast.literal_eval(v["shape"])}
        print(f"  Phase 2 cached: {ordering_str}")
    else:
        ordering, degradations, bl_mean = run_phase2()
        ordering_str = " > ".join(ordering)

    p3c = load_json(CACHE_DIR / "phase3_complete.json")
    if p3c:
        ls_name = p3c["least_sensitive"]
        p3_dec = p3c["decision"]
        print(f"  Phase 3 cached: {ls_name} → rank {p3_dec}")
    else:
        ls_name, p3_dec, _, _ = run_phase3(ordering, degradations)

    # Build allocations
    rank_tiers = [4, 2]
    alloc, alloc_named = {}, {}
    for i, ln in enumerate(ordering):
        sh = LAYER_SHAPES[ln]
        alloc[sh] = p3_dec if i == len(ordering) - 1 else rank_tiers[min(i, len(rank_tiers)-1)]
        alloc_named[ln] = alloc[sh]
    label = "_".join(str(alloc_named[n]) for n in ["input", "hidden", "output"])
    budget = alloc_named["input"] + 2*alloc_named["hidden"] + alloc_named["output"]

    save_json(DRIVE_DIR / "pilot_results.json", {
        "phase1": {"ordering": p1_ord, "summary": p1_sum},
        "phase2": {"baseline_mean": bl_mean,
                   "degradations": {k: {**v, "shape": str(v["shape"])}
                                    for k, v in degradations.items()},
                   "ordering": ordering_str},
        "phase3": {"least_sensitive": ls_name, "decision": p3_dec},
        "allocation": alloc_named, "allocation_label": label, "rank_budget": budget,
        "rank_spec": {str(k): v for k, v in alloc.items()}})

    print(f"\n  Allocation: {alloc_named} (label: {label}, budget: {budget})")
    return alloc, alloc_named, label


def load_or_run_pilot():
    """Load pilot from saved JSON, or run it."""
    pf = DRIVE_DIR / "pilot_results.json"
    if pf.exists():
        pr = load_json(pf)
        alloc_named = pr["allocation"]
        alloc = {LAYER_SHAPES[ln]: rv for ln, rv in alloc_named.items()}
        label = pr["allocation_label"]
        print(f"  Pilot loaded: {alloc_named} (label: {label})")
        return alloc, alloc_named, label
    return run_pilot()


# ── Method runner helper ──────────────────────────────────────────
def run_method(method_name, noiser_class, rank_spec, max_gens=MAX_GENS):
    """Run a single method across all seeds. Skips completed seeds."""
    print(f"\n{'~'*60}\n{method_name} ({max_gens} gens)\n{'~'*60}", flush=True)
    results = []
    for seed in SEEDS:
        rf = DRIVE_DIR / f"{method_name}_seed{seed}.json"
        if rf.exists():
            loaded = load_json(rf)
            if loaded and "best_fitness" in loaded:
                print(f"  [{method_name} s{seed}] exists (best={loaded['best_fitness']:.1f}), skipping.", flush=True)
                results.append(loaded)
                continue
            else:
                print(f"  [{method_name} s{seed}] corrupt JSON, re-running.", flush=True)
        r = train(seed, noiser_class, rank_spec, method_name, max_gens=max_gens)
        save_json(rf, r)
        results.append(r)
    bests = [r["best_fitness"] for r in results]
    finals = [r["final_mean_fitness"] for r in results]
    print(f"  → mean_best={np.mean(bests):.1f}±{np.std(bests):.1f}  "
          f"mean_final={np.mean(finals):.1f}±{np.std(finals):.1f}")
    return results


# ── Preflight ─────────────────────────────────────────────────────
def preflight():
    print("Pre-flight JIT test...")
    env = make_env()
    key = jax.random.PRNGKey(42)
    key, mk, ek, epk = jax.random.split(key, 4)
    fp, p, sm, em = init_model(mk)
    estk = jax.tree.map(_safe_wrap_key,
           hs.models.common.simple_es_tree_key(p, ek, sm))
    # EggRoll
    t0 = time.time()
    fnp, cnp = init_noiser(p, EggRoll, RANK)
    def _eval_egg(mid_i, ep_key):
        it = (jnp.int32(0), mid_i)
        def pol(obs): return jnp.clip(MODEL.forward(EggRoll, fnp, cnp, fp, p, estk, it, obs), -1, 1)
        state = env.reset(ep_key)
        def sf(c, _):
            st, tr, d = c; ns = env.step(st, pol(st.obs))
            return (ns, tr + ns.reward*(1-d), jnp.logical_or(d, ns.done)), None
        (_, tr, _), _ = jax.lax.scan(sf, (state, jnp.float32(0), jnp.bool_(False)), None, length=10)
        return tr
    vf = jax.jit(jax.vmap(_eval_egg, in_axes=(0, 0)))
    tm, tk = jnp.arange(4, dtype=jnp.int32), jax.random.split(epk, 4)
    r1 = vf(tm, tk)
    print(f"  EggRoll OK ({time.time()-t0:.1f}s)")
    # LWREggRoll
    t0 = time.time()
    lspec = {s: RANK for s in LAYER_SHAPES.values()}
    fnp2, cnp2 = init_noiser(p, LWREggRoll, lspec)
    def _eval_lwr(mid_i, ep_key):
        it = (jnp.int32(0), mid_i)
        def pol(obs): return jnp.clip(MODEL.forward(LWREggRoll, fnp2, cnp2, fp, p, estk, it, obs), -1, 1)
        state = env.reset(ep_key)
        def sf(c, _):
            st, tr, d = c; ns = env.step(st, pol(st.obs))
            return (ns, tr + ns.reward*(1-d), jnp.logical_or(d, ns.done)), None
        (_, tr, _), _ = jax.lax.scan(sf, (state, jnp.float32(0), jnp.bool_(False)), None, length=10)
        return tr
    vf2 = jax.jit(jax.vmap(_eval_lwr, in_axes=(0, 0)))
    r2 = vf2(tm, tk)
    print(f"  LWREggRoll OK ({time.time()-t0:.1f}s)")
    print("Pre-flight PASSED.\n")


def main():
    """Run the full Brax Ant experiment: pilot + six methods."""
    preflight()

    # Clear old Phase 3 cache so corrected decision logic re-runs
    for stale in [CACHE_DIR / "phase3_complete.json", DRIVE_DIR / "pilot_results.json"]:
        if stale.exists():
            stale.unlink()
            print(f"  Cleared stale: {stale.name}")

    alloc, alloc_named, alloc_label = load_or_run_pilot()

    # Build derived allocations for all methods
    alloc_r8 = {}
    alloc_named_r8 = {}
    for ln in ["input", "hidden", "output"]:
        alloc_named_r8[ln] = min(alloc_named[ln] * 2, 8)
        alloc_r8[LAYER_SHAPES[ln]] = alloc_named_r8[ln]

    alloc_412 = {LAYER_SHAPES["input"]: 4, LAYER_SHAPES["hidden"]: 1, LAYER_SHAPES["output"]: 2}
    alloc_814 = {LAYER_SHAPES["input"]: 8, LAYER_SHAPES["hidden"]: 1, LAYER_SHAPES["output"]: 4}

    print(f"\nMethod allocations:")
    print(f"  eggroll_r4:  uniform rank 4, budget 16")
    print(f"  eggroll_r1:  uniform rank 1, budget 4")
    print(f"  lwr_{alloc_label}: pilot-derived, budget {sum(alloc.values())}")
    print(f"  lwr_8_4_0:   scaled-up, budget {sum(alloc_r8.values())}")
    print(f"  lwr_4_1_2:   intermediate, budget 8")
    print(f"  lwr_8_1_4:   ceiling test, budget 14")

    # Run all six methods
    run_method("eggroll_r4", EggRoll, RANK)
    run_method("eggroll_r1", EggRoll, 1)
    run_method(f"lwr_{alloc_label}", LWREggRoll, alloc)
    run_method("lwr_8_4_0", LWREggRoll, alloc_r8)
    run_method("lwr_4_1_2", LWREggRoll, alloc_412)
    run_method("lwr_8_1_4", LWREggRoll, alloc_814)

    # Summary
    import statistics

    print(f"\n{'='*70}")
    print("BRAX ANT — RESULTS SUMMARY")
    print(f"{'='*70}")

    methods = ["eggroll_r4", "eggroll_r1", f"lwr_{alloc_label}",
               "lwr_8_4_0", "lwr_4_1_2", "lwr_8_1_4"]
    budgets = {"eggroll_r4": 16, "eggroll_r1": 4, f"lwr_{alloc_label}": sum(alloc.values()),
               "lwr_8_4_0": sum(alloc_r8.values()), "lwr_4_1_2": 8, "lwr_8_1_4": 14}

    print(f"\n{'Method':<20} {'Budget':>6} {'Mean Best':>12} {'Std':>8} {'Mean Final':>12}")
    print("-" * 65)

    all_results = {}
    for mn in methods:
        results = []
        for seed in SEEDS:
            rf = DRIVE_DIR / f"{mn}_seed{seed}.json"
            if rf.exists():
                results.append(load_json(rf))
        if not results:
            print(f"{mn:<20} {'—':>6} {'not found':>12}")
            continue
        bests = [r["best_fitness"] for r in results]
        finals = [r["final_mean_fitness"] for r in results]
        bm = statistics.mean(bests)
        bs = statistics.stdev(bests) if len(bests) > 1 else 0
        fm = statistics.mean(finals)
        print(f"{mn:<20} {budgets.get(mn, '?'):>6} {bm:>12.1f} {bs:>8.1f} {fm:>12.1f}")
        all_results[mn] = {"mean_best": bm, "std_best": bs, "mean_final": fm,
                           "per_seed_best": bests}

    save_json(DRIVE_DIR / "summary.json", {
        "env": ENV_NAME,
        "architecture": [OBS_DIM] + [LAYER_SIZE]*N_LAYERS + [ACT_DIM],
        "max_gens": MAX_GENS, "pop_size": POP_SIZE,
        "pilot_allocation": alloc_named,
        "results": all_results})
    print(f"\nSummary saved to {DRIVE_DIR}/summary.json")


if __name__ == "__main__":
    main()
