"""Evaluation harness for the scale ladder, on fixed evaluation sets.

Computes the ONE-SHOT quantities the scale-ladder analysis is defined on (NOT msaug):

  realized benefit  = sas_oneshot - uniform_oneshot   (same SAS ckpt,
      learned-head-argmax start vs uniform start, the head unrestricted)
  capture                     = (sas_at50 - uniform_oneshot) / (best_of50 - uniform_oneshot)
      pool-aligned k=50: 50 feasible starts sampled uniform w/o replacement from
      {s: 2 d(depot,s) <= T}, min(50,|feasible|), fixed per-instance seed.
  pomo_oneshot                = independent POMO baseline (plain greedy, 1 rollout).

One checkpoint -> one JSON (+ per-instance npz). The SAS arms share ONE loaded
model (toggle policy.sas_arm). Eval set + budget are pinned:
  n100 eval_op_n100_first1000.pkl / T4 ; n200 .../T5.5 ; n500 .../T9.
"""

import os as _os, sys as _sys  # resolve sibling package modules
_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('sas',):
    _p = _os.path.join(_SRC, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np
import torch
from tensordict import TensorDict

from rl4co.envs import OPEnv
from rl4co.models.zoo.pomo import POMO

POOL_K = 50
POOL_SEED_BASE = 770000  # per-instance start-pool seed = base + idx (fixed)
# Evaluation batch shape is fixed and matches the budget-sweep harness (appendix A:
# batch-shape kernel numerics move greedy results).
ONE_CHUNK = 200          # one-shot forwards (budget_sweep_eval --chunk)
MS_CHUNK = 50            # multistart forwards (budget_sweep_eval --ms-chunk)
UNIFORM_DRAWS = 8        # uniform_oneshot = mean over 8 fixed-seed draws
UNIFORM_SEED_BASE = 123400  # draw seed = base + k (budget_sweep_eval verbatim)


def load_set(path, budget):
    with open(path, "rb") as f:
        data = pickle.load(f)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    B = len(data)
    n = len(data[0][1])
    depot = torch.tensor([d[0] for d in data], dtype=torch.float32)
    locs = torch.tensor([d[1] for d in data], dtype=torch.float32)
    prize = torch.tensor([d[2] for d in data], dtype=torch.float32)
    max_length = torch.full((B,), float(budget), dtype=torch.float32)
    td = TensorDict(dict(depot=depot, locs=locs, prize=prize, max_length=max_length),
                    batch_size=[B])
    return td, md5, B, n


def load_holdout(pooldir):
    """The regime study holdout: sharded npy pool -> td_all with PER-INSTANCE max_length
    (alpha*L varies), + the per-instance prize scheme (0/1/2) for per-layer reads.
    Deterministic order (shard order); every instance evaluated (no sampling)."""
    import glob
    co = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{pooldir}/coords_*.npy"))])   # [B,n+1,2]
    pr = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{pooldir}/prize_*.npy"))])     # [B,n]
    ml = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{pooldir}/maxlen_*.npy"))])     # [B]
    sch = np.concatenate([np.load(f)["scheme"] for f in sorted(glob.glob(f"{pooldir}/meta_*.npz"))])
    h = hashlib.md5()
    for f in sorted(glob.glob(f"{pooldir}/*.npy")) + sorted(glob.glob(f"{pooldir}/*.npz")):
        h.update(open(f, "rb").read())
    B, n = co.shape[0], co.shape[1] - 1
    td = TensorDict(dict(depot=torch.from_numpy(co[:, 0].astype(np.float32)),
                         locs=torch.from_numpy(co[:, 1:].astype(np.float32)),
                         prize=torch.from_numpy(pr.astype(np.float32)),
                         max_length=torch.from_numpy(ml.astype(np.float32))),
                    batch_size=[B])
    return td, h.hexdigest(), B, n, sch.astype(np.int64)


def build_pools(td, budget):
    """[B, K] start-node indices (into the n+1 = depot+locs action space).
    Feasible = {s>=1 : 2 d(depot,s) <= T}. Fixed per-instance seed.
    budget: scalar T, or per-instance array (the regime study holdout)."""
    depot = td["depot"].numpy()                      # [B,2]
    locs = td["locs"].numpy()                        # [B,n,2]
    B, n = locs.shape[0], locs.shape[1]
    d0 = np.linalg.norm(locs - depot[:, None, :], axis=-1)  # [B,n]
    barr = np.asarray(budget, np.float64).reshape(-1)
    pools = np.zeros((B, POOL_K), dtype=np.int64)
    for i in range(B):
        bi = barr[i] if barr.size > 1 else barr[0]
        feas = np.where(2.0 * d0[i] <= bi + 1e-12)[0] + 1       # +1: action idx (depot=0)
        rng = np.random.default_rng(POOL_SEED_BASE + i)
        if len(feas) >= POOL_K:
            pick = rng.choice(feas, size=POOL_K, replace=False)
        else:                                        # pad w/ replacement
            pick = np.concatenate([feas, rng.choice(feas, size=POOL_K - len(feas), replace=True)])
        pools[i] = pick
    return torch.from_numpy(pools)                   # [B,K]


def build_full_pools(td, budget):
    """[B, Kmax] ALL feasible start indices, short rows padded with repeats
    (harmless: we take max over starts). For the best-of-50-vs-full-oracle bias
    check (feasible only at n<=200; that IS why k=50 is fixed larger)."""
    depot = td["depot"].numpy(); locs = td["locs"].numpy()
    B = locs.shape[0]
    d0 = np.linalg.norm(locs - depot[:, None, :], axis=-1)
    feas_lists = [np.where(2.0 * d0[i] <= budget + 1e-12)[0] + 1 for i in range(B)]
    kmax = max(len(f) for f in feas_lists)
    pools = np.zeros((B, kmax), dtype=np.int64)
    for i, f in enumerate(feas_lists):
        if len(f) == 0:
            f = np.array([1])                            # degenerate; never at these T
        pools[i] = np.concatenate([f, np.repeat(f[-1], kmax - len(f))])
    return torch.from_numpy(pools), [len(f) for f in feas_lists]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["sas", "pomo"],
                    help="sas: loads SAS ckpt, computes sas/uniform/sas@50/bo50. "
                         "pomo: loads POMO ckpt, computes pomo_oneshot.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--holdout", default="", help="the regime study holdout pool dir: per-instance "
                    "max_length=alpha*L + prize-scheme layering; overrides --dataset/--budget.")
    ap.add_argument("--pool", default="", help="the regime study PoolOPGenerator dir (num_loc from pool)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--permute-labels", action="store_true",
                    help="null self-check: shuffle sas/uniform start assignment per instance")
    ap.add_argument("--full-oracle", action="store_true",
                    help="full-oracle bias check: also compute best-over-ALL-feasible-starts and "
                         "report best_of_50/full_oracle ratio (n<=200 only). SAS arm.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N instances (mechanics only)")
    args = ap.parse_args()

    dev = "cpu" if args.smoke else "cuda"
    scheme = None
    if args.holdout:                                  # the regime study: per-instance budget + layers
        td_all, md5, B, n, scheme = load_holdout(args.holdout)
        budget = td_all["max_length"].numpy()         # per-instance alpha*L
    else:
        td_all, md5, B, n = load_set(args.dataset, args.budget)
        budget = args.budget
    if args.limit:
        td_all = td_all[:args.limit]; B = args.limit
        if scheme is not None:
            scheme = scheme[:args.limit]
            budget = budget[:args.limit]
    env = OPEnv(generator_params=dict(num_loc=n, max_length=float(np.max(budget))))

    if args.arm == "pomo":
        model = POMO(env)
    else:
        from sas_pomo import SASPOMO
        model = SASPOMO(env, sas_arm="sas")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    print(f"loaded {args.arm} ckpt strict=True ({len(sd)} keys)", flush=True)
    policy = model.policy.to(dev).eval()

    pools = build_pools(td_all, budget) if args.arm == "sas" else None
    one_chunk = ONE_CHUNK if not args.smoke else 8   # one-shot forwards (=200)
    ms_chunk = MS_CHUNK if not args.smoke else 4     # multistart forwards (=50)
    results = dict(arm=args.arm, ckpt=args.ckpt, dataset=args.dataset or args.holdout,
                   dataset_md5=md5, n=n, budget=(args.budget if not args.holdout else "per-inst-alphaL"),
                   holdout=args.holdout, pool=args.pool, n_inst=B, device=dev,
                   pool_k=POOL_K, pool_seed_base=POOL_SEED_BASE,
                   permute_labels=args.permute_labels, schemes={})
    per_inst = {}
    if scheme is not None:                            # per-layer read (0/1/2)
        per_inst["scheme"] = scheme

    def oneshot(sas_arm, draw_seed=None):
        policy.sas_arm = sas_arm
        if draw_seed is not None:                     # budget_sweep_eval: fixed uniform draw
            torch.manual_seed(draw_seed)
        outs = []
        with torch.inference_mode():
            for i in range(0, B, one_chunk):
                td = env.reset(td_all[i:i+one_chunk].to(dev))
                r = policy(td, env, phase="test", decode_type="greedy",
                           num_starts=0, one_shot_start=True)["reward"]
                outs.append(r.detach().float().cpu())
        return torch.cat(outs)

    def forced_multistart(pool_chunk_fn, k):
        """greedy rollout from each of k forced starts; returns [B,k] rewards."""
        outs = []
        with torch.inference_mode():
            for i in range(0, B, ms_chunk):
                td = env.reset(td_all[i:i+ms_chunk].to(dev))
                b = td.shape[0]
                sel = pool_chunk_fn(i, i + b).to(dev)             # [b,k]
                fn = lambda td_, env_, num_starts, _s=sel: _s.transpose(0, 1).reshape(-1)  # (k b)
                r = policy(td, env, phase="test", decode_type="multistart_greedy",
                           num_starts=k, select_start_nodes_fn=fn)["reward"]
                outs.append(r.reshape(k, b).T.detach().float().cpu())  # [b,k]
        return torch.cat(outs, 0)

    if args.arm == "pomo":
        r = oneshot_pomo = None
        outs = []
        with torch.inference_mode():
            for i in range(0, B, one_chunk):
                td = env.reset(td_all[i:i+one_chunk].to(dev))
                outs.append(policy(td, env, phase="test",
                                   decode_type="greedy")["reward"].detach().float().cpu())
        pomo = torch.cat(outs)
        results["schemes"]["pomo_oneshot"] = dict(mean=round(float(pomo.mean()), 6))
        per_inst["pomo_oneshot"] = pomo.numpy()
        print(f"[pomo] oneshot mean={float(pomo.mean()):.6f}", flush=True)
    else:
        # null: permute which arm ("sas"/"uniform") each instance's head uses.
        # Implemented by computing both then, per instance, swapping with a fixed RNG.
        sas = oneshot("sas")
        # uniform_oneshot = mean over UNIFORM_DRAWS fixed-seed draws (budget_sweep_eval verbatim)
        udraws = torch.stack([oneshot("uniform", draw_seed=UNIFORM_SEED_BASE + k)
                              for k in range(UNIFORM_DRAWS)])   # [K,B]
        unif = udraws.mean(0)
        per_inst["uniform_oneshot_draws"] = udraws.numpy()
        bo50 = forced_multistart(lambda a, b: pools[a:b], POOL_K).max(-1).values
        # sas@50: head-argmax within the pool -> reward of that one forced start.
        sas50 = _sas_at_pool(policy, env, td_all, pools, ms_chunk, dev, B)
        if args.full_oracle:
            fpool, feas_sizes = build_full_pools(td_all, args.budget)  # scalar T only
            bofull = forced_multistart(lambda a, b: fpool[a:b], fpool.shape[1]).max(-1).values
            ratio = float((bo50.mean() / bofull.mean())) if float(bofull.mean()) != 0 else float("nan")
            results["schemes"]["best_of_full_oracle"] = dict(mean=round(float(bofull.mean()), 6))
            results["full_oracle"] = dict(kmax=int(fpool.shape[1]),
                                          feas_mean=float(np.mean(feas_sizes)),
                                          feas_min=int(np.min(feas_sizes)),
                                          best50_over_fulloracle=round(ratio, 6))
            per_inst["best_of_full_oracle"] = bofull.numpy()
            print(f"[full-oracle bias] best50/full_oracle={ratio:.6f}  kmax={fpool.shape[1]} "
                  f"feas_mean={np.mean(feas_sizes):.1f}", flush=True)
        if args.permute_labels:
            rng = np.random.default_rng(12345)
            flip = torch.from_numpy(rng.integers(0, 2, size=len(sas)).astype(bool))
            sas, unif = torch.where(flip, unif, sas), torch.where(flip, sas, unif)
        for name, v in [("sas_oneshot", sas), ("uniform_oneshot", unif),
                        ("best_of_50", bo50), ("sas_at_50", sas50)]:
            results["schemes"][name] = dict(mean=round(float(v.mean()), 6))
            per_inst[name] = v.numpy()
        print("schemes done (see JSON)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    np.savez_compressed(args.out.replace(".json", "_perinst.npz"), **per_inst)
    print("SCALE_LADDER_EVAL_DONE", args.out, flush=True)


def _sas_at_pool(policy, env, td_all, pools, chunk, dev, B):
    """head-argmax restricted to the 50-pool -> single forced-start greedy reward."""
    policy.sas_arm = "sas"
    outs = []
    with torch.inference_mode():
        for i in range(0, B, chunk):
            td = env.reset(td_all[i:i+chunk].to(dev))
            b = td.shape[0]
            hidden, _ = policy.encoder(td)
            scores = policy.start_head(hidden.float())          # [b, n+1]
            sel = pools[i:i+b].to(dev)                         # [b, K]
            pool_scores = scores.gather(1, sel)                # [b, K]
            best = sel.gather(1, pool_scores.argmax(1, keepdim=True)).squeeze(1)  # [b]
            # rl4co decoding.py:249 sets multistart=False when num_starts==1, which
            # silently DROPS select_start_nodes_fn -> plain greedy rollout. Use k=2
            # identical forced starts (multistart enabled) and keep column 0.
            fn = lambda td_, env_, num_starts, _s=best: _s.repeat(2)   # (k b), k=2
            r = policy(td, env, phase="test", decode_type="multistart_greedy",
                       num_starts=2, select_start_nodes_fn=fn)["reward"]
            outs.append(r.reshape(2, b)[0].detach().float().cpu())
    return torch.cat(outs)


if __name__ == "__main__":
    main()
