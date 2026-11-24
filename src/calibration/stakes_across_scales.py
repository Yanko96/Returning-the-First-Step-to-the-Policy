"""Start-stakes at the scale-ladder budgets, with the solver on each side configurable.

Appendix H measures the start-stakes of Section 4.3 at n = 50, 100, 200 and 500 under
three solver configurations:

  as_defined      forced: light solver, 2 restarts
                  reference: reference solver, 16 restarts      (the definition of Section 4.3)
  light_both      forced: light solver, 16 restarts
                  reference: light solver, 16 restarts
  reference_both  forced: reference solver with its first move locked to the forced start,
                  16 restarts
                  reference: reference solver, 16 restarts

  stakes_i = 1 - mean_s forced(i, s) / reference_i, clipped to [0, 1],

with 8 forced starts drawn uniformly without replacement from the feasible set
{s : 2 d(0, s) <= T}, on the first --n-inst instances of the evaluation set of a scale
(budgets T = 3.0 / 4.0 / 5.5 / 9.0 at n = 50 / 100 / 200 / 500; at n = 50 the set is the
OP50 test set, at larger n the held-out set named in the `dataset` field of the shipped
scale-ladder cell JSONs). Solvers: the light solver is `lite_solve` of
src/instruments/budget_sweep_stakes.py, the reference solver `op_solve_restart` of
src/instruments/op_ils_reference.py, and its locked form `op_solve_restart` of
src/instruments/op_ils_reference_forced.py. For as_defined the run also records the light
solver's unconstrained result at 8 restarts over the reference (`light_free_over_reference`).

Seeds, per instance idx:
  as_defined      reference restarts G + r, light restarts S + 9000 + r (unconstrained) and
                  S + 10 k + r (forced start k), forced starts drawn with seed S; at n = 50
                  G = 400000 + 100 idx and S = 500000 + 100 idx (the seeds of the n = 50 budget
                  sweep at T = 3), at larger n G = S = 700000 + 1000 n + 100 idx;
  light_both,     S = 820000 + 1000 n + 100 idx; reference restarts S + 9000 + r, forced
  reference_both  restarts S + 10 k + r, forced starts drawn with seed S.
With these seeds every run reproduces artifacts/stakes/stakes_across_scales.json exactly.

CPU only; the cost grows steeply with n: one as_defined instance takes about 15 s on one
core at n = 50 and about half an hour at n = 500, and reference_both costs several times
as much. Use --first and --n-inst to split a scale across machines.

    python src/calibration/stakes_across_scales.py --variant as_defined --n 100 \
        --budget 4.0 --dataset <held-out n=100 set> --out outputs/stakes_as_defined_n100.json

`python src/calibration/stakes_across_scales_report.py` prints the numbers Appendix H
reports from the shipped artifact.
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "instruments"))

from budget_sweep_stakes import lite_solve  # noqa: E402
import op_ils_reference as REF  # noqa: E402
import op_ils_reference_forced as REF_LOCKED  # noqa: E402

N_FORCED = 8
VARIANTS = {"as_defined": ("light", 2, "reference", 16),
            "light_both": ("light", 16, "light", 16),
            "reference_both": ("reference_locked", 16, "reference", 16)}
R_LIGHT_FREE = 8          # as_defined diagnostic: unconstrained light restarts
_G = {}


def solver(name):
    if name == "light":
        return lambda D, prize, B, seed, forced=None: lite_solve(D, prize, B, seed, forced=forced)
    if name == "reference":
        def f(D, prize, B, seed, forced=None):
            assert forced is None, "the reference solver has no forced-start mode"
            return REF.op_solve_restart(D, prize, B, seed)[0]
        return f
    if name == "reference_locked":
        return lambda D, prize, B, seed, forced=None: REF_LOCKED.op_solve_restart(
            D, prize, B, seed, forced=forced)[0]
    raise ValueError(name)


def seeds(variant, n, idx):
    """(reference seed base G, stakes seed base S, reference restart offset)."""
    if variant == "as_defined":
        if n == 50:
            return 400000 + idx * 100, 500000 + idx * 100, 0
        s = 700000 + n * 1000 + idx * 100
        return s, s, 0
    s = 820000 + n * 1000 + idx * 100
    return s, s, 9000


def instance(data, idx):
    r = data[idx]
    depot = np.asarray(r[0], float)
    loc = np.asarray(r[1], float)
    xy = np.vstack([depot[None, :], loc])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    return D, np.concatenate([[0.0], np.asarray(r[2], float)])


def job(idx):
    data, n, T, variant = _G["data"], _G["n"], _G["T"], _G["variant"]
    num_s, num_r, den_s, den_r = VARIANTS[variant]
    fnum, fden = solver(num_s), solver(den_s)
    D, prize = instance(data, idx)
    m = D.shape[0]
    g0, s0, off = seeds(variant, n, idx)
    den = max(fden(D, prize, T, g0 + off + r) for r in range(den_r))
    out = dict(idx=idx)
    if variant == "as_defined":
        free = max(lite_solve(D, prize, T, s0 + 9000 + r) for r in range(R_LIGHT_FREE))
        out["light_free_over_reference"] = float(free / max(den, 1e-9))
    rng = np.random.default_rng(s0)
    feas = [s for s in range(1, m) if 2 * D[0, s] <= T + 1e-12]
    if not feas:
        return None
    forced = rng.choice(feas, size=min(N_FORCED, len(feas)), replace=False)
    fv = np.asarray([max(fnum(D, prize, T, s0 + k * 10 + r, forced=int(s)) for r in range(num_r))
                     for k, s in enumerate(forced)], float)
    raw = 1.0 - fv.mean() / max(den, 1e-9)
    out.update(stakes=float(np.clip(raw, 0.0, 1.0)), stakes_raw=float(raw),
               clipped=bool(raw < 0.0 or raw > 1.0),
               start_se=float(fv.std(ddof=1) / np.sqrt(len(fv))) if len(fv) > 1 else 0.0,
               n_feas=len(feas))
    return out


def init(path, n, T, variant):
    with open(path, "rb") as f:
        _G.update(data=pickle.load(f), n=n, T=T, variant=variant)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--n", type=int, required=True, help="instance size of the evaluation set")
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--n-inst", type=int, default=200)
    ap.add_argument("--first", type=int, default=0,
                    help="first instance index (to split one scale across several runs)")
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = VARIANTS[a.variant]
    t0 = time.time()
    with Pool(a.procs, initializer=init, initargs=(a.dataset, a.n, a.budget, a.variant)) as pool:
        rows = [r for r in pool.imap(job, range(a.first, a.first + a.n_inst)) if r]
    sv = np.array([r["stakes"] for r in rows])
    res = dict(variant=a.variant, forced="%s@%d" % cfg[:2], reference="%s@%d" % cfg[2:],
               n=a.n, T=a.budget, n_inst=len(rows), first=a.first,
               stakes_mean=float(sv.mean()), stakes_se=float(sv.std(ddof=1) / np.sqrt(len(sv))),
               stakes_raw_mean=float(np.mean([r["stakes_raw"] for r in rows])),
               clip_events=int(sum(r["clipped"] for r in rows)),
               start_se_mean=float(np.mean([r["start_se"] for r in rows])),
               n_feas_mean=float(np.mean([r["n_feas"] for r in rows])),
               per_instance={str(r["idx"]): r["stakes"] for r in rows})
    if a.variant == "as_defined":
        q = [r["light_free_over_reference"] for r in rows]
        res["light_free_over_reference"] = float(np.mean(q))
        res["per_instance_light_free_over_reference"] = {str(r["idx"]): r["light_free_over_reference"]
                                                         for r in rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("%s n=%d T=%.1f: stakes %.4f (SE %.4f), %d instances, %.0fs -> %s"
          % (a.variant, a.n, a.budget, res["stakes_mean"], res["stakes_se"], len(rows),
             time.time() - t0, a.out))


if __name__ == "__main__":
    main()
