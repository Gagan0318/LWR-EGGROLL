"""LunarLander-v3 (symmetric, tuned): investigating ES viability with reduced fitness noise.
Architecture: [8, 64, 64, 4] — symmetric hidden widths.
Changes vs original symmetric run:
  - N_EVAL_EPISODES: 5 → 8    (reduce fitness noise, feasible runtime)
  - POP_SIZE: 256              (standard setting)
  - MAX_GENS: 300              (sufficient convergence window)
  - Added vanilla r=1 baseline (paper's recommendation)
  - Added lower LWR allocations capped at r=4
Justification: Original symmetric run produced near-zero ES fitness (10.9 ± 236).
This tests whether the failure was due to noisy fitness evaluation rather than
a fundamental ES limitation on LunarLander.
Optimizer: Adam. Sigma: 0.05, LR: 0.01. Seeds: 3, Episodes: 8.
Pilot: 50 gens (reduced from 100 for feasibility).
"""
import os; os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.30"
import json,time; from pathlib import Path; from multiprocessing import Pool,cpu_count
import gymnasium as gym; import jax; import jax.numpy as jnp; import numpy as np

ENV_NAME="LunarLander-v3"; OBS_DIM=8; ACT_DIM=4; HIDDEN_DIMS=(64,64); N_LAYERS=3
POP_SIZE=256; SIGMA=0.05; LR=0.01; MAX_GENS=300; N_EVAL_EPISODES=8
SEEDS=[0,1,2]; N_WORKERS=min(cpu_count(),8)
LAYER_SHAPES={"input":(64,8),"hidden":(64,64),"output":(4,64)}
RESULTS_DIR=Path("results/lunarlander_symmetric_tuned"); RESULTS_DIR.mkdir(parents=True,exist_ok=True)

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

def run_reinforce(seed,lr=0.001,max_episodes=2000,gamma=0.99):
    key=jax.random.PRNGKey(seed);key,ik=jax.random.split(key);params=init_params(ik)
    @jax.jit
    def loss(params,obs,acts,rets):
        lp=jax.vmap(lambda o,a:jax.nn.log_softmax(forward(params,o))[a])(obs,acts)
        return -jnp.sum(lp*rets)
    gf=jax.jit(jax.grad(loss));history=[];best=-float("inf");recent=[]
    print(f"  reinforce seed={seed}",flush=True);t0=time.time()
    for ep in range(max_episodes):
        env=gym.make(ENV_NAME);obs,_=env.reset();d=False;observations=[];actions=[];rewards=[]
        while not d:
            oj=jnp.array(obs,dtype=jnp.float32);probs=jax.nn.softmax(forward(params,oj))
            key,ak=jax.random.split(key);action=int(jax.random.categorical(ak,jnp.log(probs)))
            observations.append(obs);actions.append(action)
            obs,rw,te,tr,_=env.step(action);rewards.append(rw);d=te or tr
        env.close();epr=sum(rewards)
        if epr>best:best=epr
        recent.append(epr);
        if len(recent)>100:recent.pop(0)
        G=0;rets=[]
        for r in reversed(rewards):G=r+gamma*G;rets.insert(0,G)
        rets=np.array(rets);rets=(rets-np.mean(rets))/(np.std(rets)+1e-8)
        grads=gf(params,jnp.array(observations,dtype=jnp.float32),jnp.array(actions,dtype=jnp.int32),jnp.array(rets,dtype=jnp.float32))
        for n in params:params[n]=params[n]-lr*grads[n]
        if ep%50==0:
            ar=np.mean(recent[-50:]) if len(recent)>=50 else np.mean(recent)
            print(f"    ep {ep:4d}  return={epr:.0f}  avg50={ar:.1f}  best={best:.0f}  ({time.time()-t0:.0f}s)",flush=True)
            history.append({"episode":ep,"return":float(epr),"avg_recent":float(ar),"best_so_far":best})
        if len(recent)>=100 and np.mean(recent[-100:])>=200:print(f"    SOLVED ep {ep}!",flush=True);break
    tt=time.time()-t0;fa=float(np.mean(recent[-100:])) if len(recent)>=100 else float(np.mean(recent))
    print(f"    done {tt:.0f}s, best={best:.0f}",flush=True)
    return {"method":"reinforce","seed":seed,"best_fitness":best,"final_mean_fitness":fa,"episodes":ep+1,"wall_seconds":tt,"history":history}

def main():
    from rl_sensitivity_pilot import run_rl_pilot
    tp=sum(s[0]*s[1] for s in LAYER_SHAPES.values())
    print("="*60)
    print(f"LUNARLANDER-v3 — SYMMETRIC TUNED [8,64,64,4] ({tp:,} params)")
    print(f"Reduced: POP={POP_SIZE}, EVAL_EPS={N_EVAL_EPISODES}, MAX_GENS={MAX_GENS}, PILOT_GENS=50")
    print("="*60,flush=True)

    # Run sensitivity pilot to get ordering
    lwr_alloc,alloc_named,ordering,rec=run_rl_pilot(
        init_fn=init_params,perturb_low_rank_fn=perturb_low_rank,perturb_full_rank_fn=perturb_full_rank,
        to_numpy_fn=to_numpy,eval_worker_fn=_eval_worker,layer_shapes=LAYER_SHAPES,
        env_name=ENV_NAME,pop_size=POP_SIZE,sigma=SIGMA,lr=LR,max_gens=50,
        n_eval_episodes=N_EVAL_EPISODES,n_seeds=3,n_workers=N_WORKERS,
    )

    # Build methods from pilot ordering.
    # Three-way test: uniform r=1 vs binary inclusion vs elevated LWR
    ordered_layers = ordering if isinstance(ordering, list) else ordering.split(" > ")

    # Binary inclusion: rank 1 on sensitive layers, rank 0 on least sensitive
    # Tests whether just knowing WHICH layers to freeze is enough
    binary_alloc = {}
    for i, layer_name in enumerate(ordered_layers):
        shape = LAYER_SHAPES[layer_name]
        binary_alloc[shape] = 0 if i == len(ordered_layers)-1 else 1
    binary_label = "_".join(str(binary_alloc[LAYER_SHAPES[n]]) for n in ["input","hidden","output"])

    # Elevated LWR: capped at rank 4 max
    # Tests whether rank elevation adds value beyond binary inclusion
    capped_alloc = {}
    rank_tiers = [4, 2, 0]  # most sensitive → least sensitive
    for i, layer_name in enumerate(ordered_layers):
        shape = LAYER_SHAPES[layer_name]
        capped_alloc[shape] = rank_tiers[min(i, len(rank_tiers)-1)]
    cap_label = "_".join(str(capped_alloc[LAYER_SHAPES[n]]) for n in ["input","hidden","output"])

    # Pilot-derived (uncapped, for reference)
    pilot_label = "_".join(str(lwr_alloc[LAYER_SHAPES[n]]) for n in ["input","hidden","output"])

    # Capped LWR with floor rank 1: pilot-derived ordering, tiers [4,2,1]
    # Tests efficiency claim: match r=4 performance at lower budget, no rank-0 risk
    capped_floor1 = {}
    floor1_tiers = [4, 2, 1]  # most sensitive → least sensitive
    for i, layer_name in enumerate(ordered_layers):
        shape = LAYER_SHAPES[layer_name]
        capped_floor1[shape] = floor1_tiers[min(i, len(floor1_tiers)-1)]
    floor1_label = "_".join(str(capped_floor1[LAYER_SHAPES[n]]) for n in ["input","hidden","output"])

    METHODS = {
        # Baselines
        "eggroll_r1": {s:1 for s in LAYER_SHAPES.values()},
        "eggroll_r4": {s:4 for s in LAYER_SHAPES.values()},
        "openai_es": None,
        # Three-way test
        f"lwr_binary_{binary_label}": binary_alloc,       # just freeze least sensitive
        f"lwr_elevated_{cap_label}": capped_alloc,         # freeze + elevate sensitive
        f"lwr_pilot_{pilot_label}": lwr_alloc,             # pilot-derived (uncapped)
        # Capped floor-1: pilot ordering, no rank-0
        f"lwr_capped_{floor1_label}": capped_floor1,
    }

    all_results = {
        "pilot": {"allocation": alloc_named, "ordering": ordering},
        "tuning": {"pop_size": POP_SIZE, "n_eval_episodes": N_EVAL_EPISODES, "max_gens": MAX_GENS},
        "original_result": "eggroll_r4: 10.9 ± 236.3, lwr: 5.6 ± 250.4, reinforce: 309.7 ± 7.2",
    }

    for mn, rs in METHODS.items():
        b = "full" if rs is None else sum(rs.values())
        print(f"\n--- {mn} (budget={b}) ---", flush=True); mr = []
        for seed in SEEDS:
            result_file = RESULTS_DIR/f"{mn}_seed{seed}.json"
            if result_file.exists():
                print(f"  [{mn} seed={seed}] found existing result, skipping.", flush=True)
                with open(result_file) as f: r = json.load(f)
            else:
                r = run_es(seed, mn, rs)
                with open(result_file,"w") as f: json.dump(r, f, indent=2)
            mr.append(r)
        bests = [r["best_fitness"] for r in mr]; means = [r["final_mean_fitness"] for r in mr]
        all_results[mn] = {"mean_best": float(np.mean(bests)), "std_best": float(np.std(bests)),
                           "mean_final": float(np.mean(means)), "per_seed": bests}

    print(f"\n--- reinforce ---", flush=True); rr = []
    for seed in SEEDS:
        result_file = RESULTS_DIR/f"reinforce_seed{seed}.json"
        if result_file.exists():
            print(f"  [reinforce seed={seed}] found existing result, skipping.", flush=True)
            with open(result_file) as f: r = json.load(f)
            rr.append(r)
        else:
            r = run_reinforce(seed)
            with open(result_file,"w") as f: json.dump(r, f, indent=2)
            rr.append(r)
        rr.append(r)
    bests = [r["best_fitness"] for r in rr]
    all_results["reinforce"] = {"mean_best": float(np.mean(bests)), "std_best": float(np.std(bests)), "per_seed": bests}

    print(f"\n{'='*60}\nLUNARLANDER SYMMETRIC TUNED SUMMARY\n{'='*60}")
    print(f"Pilot ordering: {ordering}")
    print(f"Pilot allocation (uncapped): {alloc_named}")
    print(f"Binary allocation: {binary_alloc}")
    print(f"Elevated allocation: {capped_alloc}")
    print(f"\nOriginal (POP=256, EVAL=5): eggroll_r4=10.9±236, lwr=5.6±250, reinforce=309.7±7.2")
    print(f"\nTuned (POP={POP_SIZE}, EVAL={N_EVAL_EPISODES}):")
    print(f"{'Method':<30} {'Best Fitness':<20}"); print("-"*50)
    for m, s in all_results.items():
        if m in ("pilot","tuning","original_result"): continue
        print(f"{m:<30} {s['mean_best']:.1f} ± {s['std_best']:.1f}")
    print(f"\n--- THREE-WAY COMPARISON ---")
    print(f"  Uniform r=1:       {all_results.get('eggroll_r1',{}).get('mean_best','N/A')}")
    bl = [k for k in all_results if k.startswith("lwr_binary_")]
    el = [k for k in all_results if k.startswith("lwr_elevated_")]
    if bl: print(f"  Binary inclusion:  {all_results[bl[0]]['mean_best']:.1f} ± {all_results[bl[0]]['std_best']:.1f}")
    if el: print(f"  Elevated LWR:      {all_results[el[0]]['mean_best']:.1f} ± {all_results[el[0]]['std_best']:.1f}")
    with open(RESULTS_DIR/"summary.json","w") as f:
        json.dump({"env": ENV_NAME, "architecture": [OBS_DIM]+list(HIDDEN_DIMS)+[ACT_DIM],
                   "note": "symmetric tuned — investigating ES viability with reduced fitness noise",
                   "results": all_results}, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)

if __name__=="__main__": main()
