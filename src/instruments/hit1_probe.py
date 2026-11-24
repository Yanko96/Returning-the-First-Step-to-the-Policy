"""Forced-start probe on the controlled prize-structure pool, per prize layer.

hit@1 asks whether a start is THE best start in a 16-restart sample of the
near-optimal space, so it is a lower bound on how often the start is near-optimal:
where many starts are near-equivalent (the distance-prize layer produces them
geometrically) it can sit near chance while the start still reaches the optimum.
The probe measures that directly: force the solver to start at the given node,
solve, and compare with the unconstrained solve on the same instance.
  reported: fraction reaching >= 99.5% of the unconstrained optimum
            median achievement ratio among the misses
--control replaces the head's pick with a uniformly random feasible start (the
counts of Appendix G; no checkpoint needed).
Solver: op_ils_reference_forced (free-equivalence, lock and DP-oracle checks pass).

Stage 1 (--stage heads, GPU): dump the head argmax start per instance.
Stage 2 (--stage probe, CPU): run the forced/free comparison.
"""

import os as _os, sys as _sys  # resolve sibling package modules
_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('scale', 'sas'):
    _p = _os.path.join(_SRC, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GEN = {0: "gen1_const", 1: "gen2_unif", 2: "gen3_dist"}
R_FREE = 16
R_FORCED = 16
SEED0 = 840000
_S = None


def stage_heads(holdout, ckpt, out):
    import torch
    from rl4co.envs import OPEnv
    import scale_ladder_eval as FE
    from sas_pomo import SASPOMO
    td_all, md5, B, n, scheme = FE.load_holdout(holdout)
    budget = td_all["max_length"].numpy()
    env = OPEnv(generator_params=dict(num_loc=n, max_length=float(budget.max())))
    m = SASPOMO(env, sas_arm="sas")
    m.load_state_dict(__import__("torch").load(ckpt, map_location="cpu",
                                              weights_only=False)["state_dict"], strict=True)
    pol = m.policy.to("cuda").eval()
    pol.sas_arm = "sas"
    picks = []
    with torch.inference_mode():
        for i in range(0, B, FE.ONE_CHUNK):
            td = env.reset(td_all[i:i + FE.ONE_CHUNK].to("cuda"))
            h, _ = pol.encoder(td)
            sc = pol.start_head(h.float())
            mask = td["action_mask"].clone()
            mask[..., 0] = False
            sc = sc.masked_fill(~mask, float("-inf"))
            picks.append(sc.argmax(-1).cpu().numpy())
    picks = np.concatenate(picks)
    np.savez_compressed(out, head_pick=picks, scheme=scheme, budget=budget)
    print(f"stage-heads: {len(picks)} picks -> {out}", flush=True)


def _init(shared):
    global _S
    _S = shared


def job(a):
    idx, = a
    import op_ils_reference_forced as K
    co, pr, ml, pick = _S["co"][idx], _S["pr"][idx], _S["ml"][idx], int(_S["pick"][idx])
    D = np.sqrt(((co[:, None, :] - co[None, :, :]) ** 2).sum(-1))
    prize = np.concatenate([[0.0], pr.astype(float)])
    T = float(ml)
    s0 = SEED0 + idx * 100
    free = max(K.op_solve_restart(D, prize, T, s0 + 9000 + r)[0] for r in range(R_FREE))
    if _S.get("control"):
        m = D.shape[0]
        feas = [t for t in range(1, m) if 2 * D[0, t] <= T + 1e-9]
        if not feas:
            return idx, None
        pick = int(np.random.default_rng(s0 + 555).choice(feas))
    if pick < 1 or 2 * D[0, pick] > T + 1e-9:
        return idx, None
    forced = max(K.op_solve_restart(D, prize, T, s0 + r, forced=pick)[0]
                 for r in range(R_FORCED))
    return idx, dict(free=float(free), forced=float(forced),
                     ratio=float(forced / free) if free > 0 else float("nan"))


CONTROL = False


def stage_probe(holdout, heads, n_inst, procs, out):
    import glob
    # scale_ladder_eval is NOT imported here: stage_probe never touches it (FE is
    # used only by stage_heads), and importing it would pull in torch, which
    # this stage does not need. The probe is CPU-only local search.
    global _S
    z = np.load(heads)
    pick, scheme = z["head_pick"], z["scheme"]
    co = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{holdout}/coords_*.npy"))])
    pr = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{holdout}/prize_*.npy"))])
    ml = np.concatenate([np.load(f) for f in sorted(glob.glob(f"{holdout}/maxlen_*.npy"))])
    # equal count per layer so the layers are comparable
    idxs = []
    for c in (0, 1, 2):
        w = np.where(scheme == c)[0][:n_inst]
        idxs.extend(w.tolist())
    _S = dict(co=co, pr=pr, ml=ml, pick=pick, control=CONTROL)
    print(f"probe on {len(idxs)} instances ({n_inst}/layer), procs={procs}", flush=True)
    acc = {}
    t0 = time.time()
    with Pool(procs, initializer=_init, initargs=(_S,)) as pool:
        for k, (i, d) in enumerate(pool.imap_unordered(job, [(i,) for i in idxs], chunksize=1)):
            if d:
                acc[i] = d
            if (k + 1) % 50 == 0:
                print(f"  {k+1}/{len(idxs)} {time.time()-t0:.0f}s", flush=True)
    res = {}
    print()
    print("%-12s%8s%14s%16s%20s%14s"
          % ("layer", "n", "hit@1(lb)", "probe>=99.5%", "median miss ratio", "mean ratio"))
    HIT = {"gen1_const": 0.0671, "gen2_unif": 0.0557, "gen3_dist": 0.0171}
    for c, nm in GEN.items():
        r = [acc[i]["ratio"] for i in acc if scheme[i] == c]
        if not r:
            continue
        r = np.asarray(r)
        near = float((r >= 0.995).mean())
        miss = r[r < 0.995]
        res[nm] = dict(n=len(r), hit1_lower_bound=HIT[nm], probe_ge_995=near,
                       median_miss_ratio=float(np.median(miss)) if len(miss) else float("nan"),
                       mean_ratio=float(r.mean()))
        print("%-12s%8d%14.4f%16.4f%20.4f%14.4f"
              % (nm, len(r), HIT[nm], near,
                 res[nm]["median_miss_ratio"], res[nm]["mean_ratio"]))
    print()
    g = res.get("gen3_dist")
    if g:
        print("gen3 reading:")
        print(f"  hit@1 lower bound      = {g['hit1_lower_bound']:.4f}   (chance = 0.02)")
        print(f"  probe-corrected bound  = {g['probe_ge_995']:.4f}")
        print(f"  median miss reaches      {g['median_miss_ratio']*100:.1f}% of the free optimum")
        if g["probe_ge_995"] < 0.05:
            print("  => still near chance")
        elif g["probe_ge_995"] > 3 * g["hit1_lower_bound"]:
            print("  => substantially lifted: near-equivalent corridor")
        else:
            print("  => intermediate: lifted above chance, by no more than 3x the lower bound")
    json.dump(res, open(out, "w"), indent=2)
    print("HIT1_PROBE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["heads", "probe"])
    ap.add_argument("--holdout", default="pools/regime_holdout")
    ap.add_argument("--ckpt", default="checkpoints/sas_regime_seed42/last.ckpt")
    ap.add_argument("--heads", default="head_picks.npz")
    ap.add_argument("--n-inst", type=int, default=150)
    ap.add_argument("--procs", type=int, default=46)
    ap.add_argument("--out", default="hit1_probe.json")
    ap.add_argument("--control", action="store_true",
                    help="replace the head argmax with a uniform random feasible "
                         "start -- the chance baseline the probe metric needs")
    a = ap.parse_args()
    global CONTROL
    CONTROL = a.control
    if a.stage == "heads":
        stage_heads(a.holdout, a.ckpt, a.heads)
    else:
        stage_probe(a.holdout, a.heads, a.n_inst, a.procs, a.out)


if __name__ == "__main__":
    main()
