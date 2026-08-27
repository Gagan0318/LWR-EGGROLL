"""LunarLander-v3 (symmetric): ES method comparison with pilot-derived LWR.
Architecture: [8, 64, 64, 4] — symmetric hidden widths.
Optimizer: Adam — matches EGGROLL paper.
Population: 256, Sigma: 0.05, LR: 0.01 — consistent with CartPole experiments.
Seeds: 3, Episodes/eval: 5. LWR allocation derived by sensitivity pilot.
"""
import os; os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.30"
import json,time; from pathlib import Path; from multiprocessing import Pool,cpu_count
import gymnasium as gym; import jax; import jax.numpy as jnp; import numpy as np

ENV_NAME="LunarLander-v3"; OBS_DIM=8; ACT_DIM=4; HIDDEN_DIMS=(64,64); N_LAYERS=3
POP_SIZE=256; SIGMA=0.05; LR=0.01; MAX_GENS=300; N_EVAL_EPISODES=5
SEEDS=[0,1,2]; N_WORKERS=min(cpu_count(),8)
LAYER_SHAPES={"input":(64,8),"hidden":(64,64),"output":(4,64)}
RESULTS_DIR=Path("results/lunarlander"); RESULTS_DIR.mkdir(parents=True,exist_ok=True)

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
    print("="*60);print(f"LUNARLANDER-v3 — SYMMETRIC [8,64,64,4] ({tp:,} params)");print("="*60,flush=True)
    lwr_alloc,alloc_named,ordering,rec=run_rl_pilot(
        init_fn=init_params,perturb_low_rank_fn=perturb_low_rank,perturb_full_rank_fn=perturb_full_rank,
        to_numpy_fn=to_numpy,eval_worker_fn=_eval_worker,layer_shapes=LAYER_SHAPES,
        env_name=ENV_NAME,pop_size=POP_SIZE,sigma=SIGMA,lr=LR,max_gens=100,
        n_eval_episodes=N_EVAL_EPISODES,n_seeds=3,n_workers=N_WORKERS,
    )
    al="_".join(str(lwr_alloc[LAYER_SHAPES[n]]) for n in ["input","hidden","output"])
    METHODS={"openai_es":None,"eggroll_r4":{s:4 for s in LAYER_SHAPES.values()},f"lwr_{al}":lwr_alloc}
    all_results={"pilot":{"allocation":alloc_named,"ordering":ordering}}
    for mn,rs in METHODS.items():
        b="full" if rs is None else sum(rs.values())
        print(f"\n--- {mn} (budget={b}) ---",flush=True);mr=[]
        for seed in SEEDS:
            r=run_es(seed,mn,rs);mr.append(r)
            with open(RESULTS_DIR/f"{mn}_seed{seed}.json","w") as f:json.dump(r,f,indent=2)
        bests=[r["best_fitness"] for r in mr];means=[r["final_mean_fitness"] for r in mr]
        all_results[mn]={"mean_best":float(np.mean(bests)),"std_best":float(np.std(bests)),"mean_final":float(np.mean(means)),"per_seed":bests}
    print(f"\n--- reinforce ---",flush=True);rr=[]
    for seed in SEEDS:
        r=run_reinforce(seed);rr.append(r)
        with open(RESULTS_DIR/f"reinforce_seed{seed}.json","w") as f:json.dump(r,f,indent=2)
    bests=[r["best_fitness"] for r in rr]
    all_results["reinforce"]={"mean_best":float(np.mean(bests)),"std_best":float(np.std(bests)),"per_seed":bests}
    print(f"\n{'='*60}\nLUNARLANDER SYMMETRIC SUMMARY\n{'='*60}")
    print(f"Pilot allocation: {alloc_named}")
    print(f"\n{'Method':<25} {'Best Fitness':<20}");print("-"*45)
    for m,s in all_results.items():
        if m=="pilot":continue
        print(f"{m:<25} {s['mean_best']:.1f} ± {s['std_best']:.1f}")
    with open(RESULTS_DIR/"summary.json","w") as f:json.dump({"env":ENV_NAME,"architecture":[OBS_DIM]+list(HIDDEN_DIMS)+[ACT_DIM],"pilot_allocation":alloc_named,"results":all_results},f,indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/",flush=True)

if __name__=="__main__": main()
