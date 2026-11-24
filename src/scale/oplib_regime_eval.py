"""OPLib bracket: 66 instances (three prize schemes x 22 geometries, same coords).

Normalisation goes through src/instruments/oplib_parser.py (bbox-min->0 + longest-side
scale; prize/max), i.e. the SAME pipeline as the synthetic side.
Per-instance n varies (101..299) and budget = COST_LIMIT/scale, so instances are
grouped by n and each group gets its own env. Schemes match scale_ladder_eval verbatim:
sas_oneshot, uniform_oneshot (mean of 8 fixed-seed draws), best_of_50, sas_at_50.

Also runs a PERMUTATION-EQUIVARIANCE check: node order in OPLib is FILE order
(parser note (d)), and the hardcore/density_partition synthetic regimes are also
order-carrying, so we verify the encoder is order-invariant rather than assume it.
"""

import os as _os, sys as _sys  # resolve sibling package modules
_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('instruments', 'sas', 'scale'):
    _p = _os.path.join(_SRC, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse, json, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tensordict import TensorDict
from rl4co.envs import OPEnv
import scale_ladder_eval as FE
from oplib_parser import parse_oplib, normalize

NAMES = "a280 bier127 d198 eil101 gil262 kroA150 kroA200 kroB150 kroB200 lin105 pr107 pr124 pr136 pr144 pr152 pr226 pr264 pr299 rat195 ts225 tsp225 u159".split()
# The 12 additional EUC_2D base instances with n <= 300, so that the set
# matches the paper's stated n<=300 rule (34 bases x 3 gens = 102) instead
# of the hardcoded 22-base middle band (66). Selected with --n300.
N300 = NAMES + "eil51 berlin52 st70 pr76 eil76 rat99 kroA100 kroB100 kroC100 kroD100 kroE100 rd100".split()

def load_bracket(root, names=None):
    names = names or NAMES
    out = []
    for g in (1, 2, 3):
        for nm in names:
            p = f"{root}/gen{g}/{nm}-gen{g}-50.oplib"
            d = normalize(parse_oplib(p))
            out.append(dict(gen=g, name=nm, depot=np.asarray(d["depot"], np.float32),
                            loc=np.asarray(d["loc"], np.float32),
                            prize=np.asarray(d["prize"], np.float32),
                            T=float(d["max_length"]),
                            ceil2d=bool(d.get("ceil_2d_approx", False))))
    return out

def td_of(rec):
    n = rec["loc"].shape[0]
    return TensorDict(dict(depot=torch.from_numpy(rec["depot"])[None],
                           locs=torch.from_numpy(rec["loc"])[None],
                           prize=torch.from_numpy(rec["prize"])[None],
                           max_length=torch.tensor([rec["T"]], dtype=torch.float32)),
                      batch_size=[1]), n

def build_policy(kind, ck, env):
    if kind == "sas":
        from sas_pomo import SASPOMO
        m = SASPOMO(env, sas_arm="sas")
    else:
        from rl4co.models.zoo.pomo import POMO
        m = POMO(env)
    m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False)["state_dict"], strict=True)
    return m.policy.to("cuda").eval()

def schemes(pol, env, td, n, T, kind):
    """returns dict of scalar rewards for ONE instance."""
    dev = "cuda"; out = {}
    with torch.inference_mode():
        t = env.reset(td.to(dev))
        if kind == "pomo":
            out["pomo_oneshot"] = float(pol(t, env, phase="test", decode_type="greedy")["reward"])
            return out
        pol.sas_arm = "sas"
        out["sas_oneshot"] = float(pol(env.reset(td.to(dev)), env, phase="test",
                                       decode_type="greedy", num_starts=0, one_shot_start=True)["reward"])
        pol.sas_arm = "uniform"; dr = []
        for k in range(FE.UNIFORM_DRAWS):
            torch.manual_seed(FE.UNIFORM_SEED_BASE + k)
            dr.append(float(pol(env.reset(td.to(dev)), env, phase="test",
                                decode_type="greedy", num_starts=0, one_shot_start=True)["reward"]))
        out["uniform_oneshot"] = float(np.mean(dr)); pol.sas_arm = "sas"
        # pool of k=50 feasible starts (same rule/seed convention as scale_ladder_eval)
        d0 = np.linalg.norm(td["locs"][0].numpy() - td["depot"][0].numpy(), axis=-1)
        feas = np.where(2.0 * d0 <= T + 1e-12)[0] + 1
        if len(feas) == 0:
            return out
        rng = np.random.default_rng(FE.POOL_SEED_BASE)
        k = min(FE.POOL_K, len(feas))
        pick = rng.choice(feas, size=k, replace=False) if len(feas) >= k else \
               np.concatenate([feas, rng.choice(feas, size=k - len(feas), replace=True)])
        sel = torch.from_numpy(pick.astype(np.int64))[None].to(dev)       # [1,k]
        fn = lambda td_, env_, ns, _s=sel: _s.transpose(0, 1).reshape(-1)
        r = pol(env.reset(td.to(dev)), env, phase="test", decode_type="multistart_greedy",
                num_starts=k, select_start_nodes_fn=fn)["reward"].reshape(k, 1)
        out["best_of_50"] = float(r.max())
        # sas@50: head-argmax restricted to the pool (k=2 trick, see scale_ladder_eval fix)
        h, _ = pol.encoder(env.reset(td.to(dev)))
        sc = pol.start_head(h.float())
        best = sel.gather(1, sc.gather(1, sel).argmax(1, keepdim=True)).squeeze(1)
        fn2 = lambda td_, env_, ns, _s=best: _s.repeat(2)
        r2 = pol(env.reset(td.to(dev)), env, phase="test", decode_type="multistart_greedy",
                 num_starts=2, select_start_nodes_fn=fn2)["reward"].reshape(2, 1)
        out["sas_at_50"] = float(r2[0])
    return out

def equivariance_check(recs):
    """permute node order of one instance; sas_oneshot must be invariant."""
    rec = recs[0]
    td, n = td_of(rec)
    env = OPEnv(generator_params=dict(num_loc=n, max_length=rec["T"]))
    pol = build_policy("sas", "checkpoints/sas_regime_seed42/last.ckpt", env)
    a = schemes(pol, env, td, n, rec["T"], "sas")["sas_oneshot"]
    p = np.random.default_rng(0).permutation(n)
    rec2 = dict(rec); rec2["loc"] = rec["loc"][p]; rec2["prize"] = rec["prize"][p]
    td2, _ = td_of(rec2)
    b = schemes(pol, env, td2, n, rec["T"], "sas")["sas_oneshot"]
    print(f"[EQUIVARIANCE] {rec['name']} gen{rec['gen']} n={n}: original={a:.6f} permuted={b:.6f} "
          f"absdiff={abs(a-b):.2e} -> {'INVARIANT' if abs(a-b)<1e-3 else 'ORDER-DEPENDENT (!)'}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="oplib"); ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--out", default="oplib_regime_eval.json")
    ap.add_argument("--sas-fmt", default="checkpoints/sas_regime_seed{s}/last.ckpt")
    ap.add_argument("--pomo-fmt", default="checkpoints/pomo_regime_seed{s}/last.ckpt")
    ap.add_argument("--no-equiv", action="store_true")
    ap.add_argument("--n300", action="store_true",
                    help="use the 34-base n<=300 set (102 instances)")
    a = ap.parse_args()
    recs = load_bracket(a.root, N300 if a.n300 else NAMES)
    print(f"parsed {len(recs)} instances; n range {min(r['loc'].shape[0] for r in recs)}"
          f"..{max(r['loc'].shape[0] for r in recs)}; ceil2d={sum(r['ceil2d'] for r in recs)}", flush=True)
    if not a.no_equiv:
        equivariance_check(recs)
    res = {}
    for kind in ("sas", "pomo"):
        for s in [int(x) for x in a.seeds.split(",")]:
            ck = (a.sas_fmt if kind == "sas" else a.pomo_fmt).format(s=s)
            rows = []
            envs = {}
            pol = None
            for rec in recs:
                n = rec["loc"].shape[0]
                if n not in envs:
                    envs[n] = OPEnv(generator_params=dict(num_loc=n, max_length=rec["T"]))
                env = envs[n]
                if pol is None:
                    pol = build_policy(kind, ck, env)
                td, _ = td_of(rec)
                sc = schemes(pol, env, td, n, rec["T"], kind)
                rows.append(dict(gen=rec["gen"], name=rec["name"], n=n, T=rec["T"], **sc))
            res[f"{kind}_seed{s}"] = rows
            print(f"  {kind} seed{s}: {len(rows)} instances done", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("OPLIB_REGIME_EVAL_DONE", a.out, flush=True)

if __name__ == "__main__":
    main()
