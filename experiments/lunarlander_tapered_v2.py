"""LunarLander-v3 (tapered v2): r=1 baseline and capped LWR allocations.
Architecture: [8, 256, 64, 4] — same as original tapered.
Changes vs original tapered run:
  - Added vanilla r=1 baseline (paper's recommendation)
  - Added LWR allocations capped at rank 4 max: (2,4,0) and (1,2,0)
    based on original pilot ordering: hidden > input > output
  - Seeds: 3 (standardised)
  - Keeps original conditions: POP=256, EVAL=5, MAX_GENS=300
Original results: eggroll_r4=291.6±2.9, lwr_4_8_0=283.5±5.3, reinforce=294.4±7.6
"""
import os; os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.30"
import json,time; from pathlib import Path; from multiprocessing import Pool,cpu_count
import gymnasium as gym; import jax; import jax.numpy as jnp; import numpy as np

ENV_NAME="LunarLander-v3"; OBS_DIM=8; ACT_DIM=4; HIDDEN_DIMS=(256,64); N_LAYERS=3
POP_SIZE=256; SIGMA=0.05; LR=0.01; MAX_GENS=300; N_EVAL_EPISODES=5
SEEDS=[0,1,2]; N_WORKERS=min(cpu_count(),8)
LAYER_SHAPES={"input":(256,8),"hidden":(64,256),"output":(4,64)}
RESULTS_DIR=Path("results/lunarlander_tapered_v2"); RESULTS_DIR.mkdir(parents=True,exist_ok=True)

def init_params(key):
    p={}; dims=[OBS_DIM]+list(HIDDEN_DIMS)+[ACT_DIM]
    for i in range(len(dims)-1):
        k1,k2,key=jax.random.split(key,3)
        p[f"w{i}"]=jax.random.normal(k1,(dims[i+1],dims[i]))*jnp.sqrt(2.0/dims[i])
        p[f"b{i}"]=jnp.zeros(dims[i+1])
    return p

def forward(p,obs):
    x=obs
    for i in range(N_LAYERS): x=p[f"w{i}"]@x+p[f"b{i}"]; x=jnp.tanh(x) if i<N_LAYERS-1 else x
    return x

def np_forward(p,obs):
    x=obs
    for i in range(N_LAYERS): x=p[f"w{i}"]@x+p[f"b{i}"]; x=np.tanh(x) if i<N_LAYERS-1 else x
    return x

def _eval_worker(args):
    p,en,ne=args; t=0.0
    for _ in range(ne):
        env=gym.make(en);obs,_=env.reset();d=False;r=0.0
        while not d: a=int(np.argmax(np_forward(p,obs)));obs,rw,te,tr,_=env.step(a);r+=rw;d=te or tr
        t+=r;env.close()
    return t/ne

def perturb_full_rank(params,key,sigma):
    n={};pe={}
    for name,p in params.items():
        k,key=jax.random.split(key);e=jax.random.normal(k,p.shape)*sigma;n[name]=e;pe[name]=p+e
    return pe,n

def perturb_low_rank(params,key,sigma,rank_spec):
    noise={};perturbed={}
    for name,p in params.items():
        k,key=jax.random.split(key)
        if p.ndim==2:
            shape=tuple(p.shape);rank=rank_spec.get(shape,4)
            if rank==0: noise[name]=jnp.zeros_like(p);perturbed[name]=p
            else:
                m,n=shape;eps=jnp.zeros((m,n))
                for _ in range(rank):
                    ka,kb,k=jax.random.split(k,3);eps=eps+jnp.outer(jax.random.normal(ka,(m,)),jax.random.normal(kb,(n,)))
                eps=eps*(sigma/jnp.sqrt(rank));noise[name]=eps;perturbed[name]=p+eps
        else: e=jax.random.normal(k,p.shape)*sigma;noise[name]=e;perturbed[name]=p+e
    return perturbed,noise

def to_numpy(params): return {k:np.array(v) for k,v in params.items()}

def run_es(seed,method_name,rank_spec=None):
    key=jax.random.PRNGKey(seed);key,ik=jax.random.split(key);params=init_params(ik)
    is_full=rank_spec is None;history=[];best=-float("inf")
    adam_m={n:jnp.zeros_like(params[n]) for n in params}
    adam_v={n:jnp.zeros_like(params[n]) for n in params}
    b1,b2,eps_adam=0.9,0.999,1e-8
    print(f"  {method_name} seed={seed}",flush=True);t0=time.time()
    pool=Pool(N_WORKERS)
    try:
        for gen in range(MAX_GENS):
            pp=[];pn=[]
            for _ in range(POP_SIZE):
                key,pk=jax.random.split(key)
                if is_full: p,n=perturb_full_rank(params,pk,SIGMA)
                else: p,n=perturb_low_rank(params,pk,SIGMA,rank_spec)
                pp.append(p);pn.append(n)
            fitnesses=np.array(pool.map(_eval_worker,[(to_numpy(p),ENV_NAME,N_EVAL_EPISODES) for p in pp]))
            gb=float(np.max(fitnesses));gm=float(np.mean(fitnesses));gv=float(np.var(fitnesses))
            if gb>best:best=gb
            history.append({"gen":gen,"mean_fitness":gm,"best_fitness":gb,"best_so_far":best,"fitness_variance":gv})
            if gen%10==0: print(f"    gen {gen:4d}  mean={gm:.1f}  best={best:.1f}  ({time.time()-t0:.0f}s)",flush=True)
            if gm>=200: print(f"    SOLVED gen {gen}!",flush=True);break
            fn=(fitnesses-np.mean(fitnesses));fs=np.std(fitnesses)
            if fs>1e-8: fn=fn/fs
            for name in params:
                grad=sum(fn[i]*pn[i][name] for i in range(POP_SIZE))/(POP_SIZE*SIGMA)
                adam_m[name]=b1*adam_m[name]+(1-b1)*grad
                adam_v[name]=b2*adam_v[name]+(1-b2)*grad**2
                mh=adam_m[name]/(1-b1**(gen+1));vh=adam_v[name]/(1-b2**(gen+1))
                params[name]=params[name]+LR*mh/(jnp.sqrt(vh)+eps_adam)
    finally: pool.close();pool.join()
    tt=time.time()-t0;print(f"    done {tt:.0f}s, best={best:.1f}",flush=True)
    return {"method":method_name,"seed":seed,"best_fitness":best,"final_mean_fitness":history[-1]["mean_fitness"],"generations":len(history),"wall_seconds":tt,"history":history}

def main():
    tp=sum(s[0]*s[1] for s in LAYER_SHAPES.values())
    print("="*60)
    print(f"LUNARLANDER-v3 — TAPERED V2 [8,256,64,4] ({tp:,} params)")
    print("r=1 baseline + capped LWR allocations (max rank 4)")
    print("="*60,flush=True)

    # Original pilot ordering: hidden > input > output
    # Original pilot allocation: input=4, hidden=8, output=0
    # Now test with lower allocations capped at rank 4:

    METHODS = {
        # Baselines
        "eggroll_r1": {s:1 for s in LAYER_SHAPES.values()},          # paper's recommendation
        "eggroll_r4": {s:4 for s in LAYER_SHAPES.values()},          # standard baseline
        # Binary inclusion: just freeze output, no elevation → total budget = 2
        "lwr_binary_1_1_0": {(256,8):1, (64,256):1, (4,64):0},
        # LWR capped at r=4: hidden > input > output ordering
        # hidden=4, input=2, output=0 → total budget = 6
        "lwr_2_4_0": {(256,8):2, (64,256):4, (4,64):0},
        # LWR minimal: hidden=2, input=1, output=0 → total budget = 3
        "lwr_1_2_0": {(256,8):1, (64,256):2, (4,64):0},
        # Original for reference: hidden=8, input=4, output=0 → total budget = 12
        "lwr_4_8_0": {(256,8):4, (64,256):8, (4,64):0},
    }

    all_results = {
        "pilot_ordering": "hidden > input > output",
        "original_results": {
            "eggroll_r4": "291.6 ± 2.9",
            "lwr_4_8_0": "283.5 ± 5.3",
            "openai_es": "287.0 ± 6.2",
            "reinforce": "294.4 ± 7.6",
        },
    }

    for mn, rs in METHODS.items():
        b = sum(rs.values())
        # Check if all seed results already exist
        existing = [RESULTS_DIR/f"{mn}_seed{seed}.json" for seed in SEEDS]
        if all(f.exists() for f in existing):
            print(f"\n--- {mn} (budget={b}) --- SKIPPED (results exist)", flush=True)
            mr = []
            for f in existing:
                with open(f) as fh: mr.append(json.load(fh))
            bests = [r["best_fitness"] for r in mr]; means = [r["final_mean_fitness"] for r in mr]
            all_results[mn] = {"mean_best": float(np.mean(bests)), "std_best": float(np.std(bests)),
                               "mean_final": float(np.mean(means)), "per_seed": bests,
                               "rank_budget": sum(rs.values())}
            continue
        print(f"\n--- {mn} (budget={b}) ---", flush=True); mr = []
        for seed in SEEDS:
            r = run_es(seed, mn, rs); mr.append(r)
            with open(RESULTS_DIR/f"{mn}_seed{seed}.json","w") as f: json.dump(r, f, indent=2)
        bests = [r["best_fitness"] for r in mr]; means = [r["final_mean_fitness"] for r in mr]
        all_results[mn] = {"mean_best": float(np.mean(bests)), "std_best": float(np.std(bests)),
                           "mean_final": float(np.mean(means)), "per_seed": bests,
                           "rank_budget": sum(rs.values())}

    print(f"\n{'='*60}\nLUNARLANDER TAPERED V2 SUMMARY\n{'='*60}")
    print(f"Pilot ordering: hidden > input > output")
    print(f"\n{'Method':<25} {'Budget':<8} {'Best Fitness':<20}"); print("-"*55)
    for m, s in all_results.items():
        if isinstance(s, dict) and "mean_best" in s:
            print(f"{m:<25} {s['rank_budget']:<8} {s['mean_best']:.1f} ± {s['std_best']:.1f}")
    print(f"\nOriginal lwr_4_8_0: 283.5 ± 5.3 (budget=12)")
    with open(RESULTS_DIR/"summary.json","w") as f:
        json.dump({"env": ENV_NAME, "architecture": [OBS_DIM]+list(HIDDEN_DIMS)+[ACT_DIM],
                   "note": "tapered v2 — r=1 baseline and capped LWR allocations",
                   "results": all_results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)

if __name__=="__main__": main()
