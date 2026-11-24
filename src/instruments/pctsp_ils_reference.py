#!/usr/bin/env python
"""PCTSP reference ILS solver, with a penalty-factor sensitivity self-check.

Settings: see the paper's appendix. Pure numpy, CPU, parallel at (instance, cell) level.

Kool-convention PCTSP n=50: depot+nodes ~ U([0,1]^2), prize ~ U(0, 4/n),
quota prize_required = 1, penalty beta ~ U(0, f*K_n/n) with K_50 = 3,
implemented with common random numbers (beta_base drawn once per instance,
scaled by f; marginal law identical across f).

Objective (minimize): tour length + sum of penalties of unvisited nodes,
subject to collected prize >= 1.

Solver: portfolio construction (5 criterion shapes) -> descent [2-opt/Or-opt
compression -> improving adds -> quota-safe improving drops -> 1-1 swaps] to
a fixed point -> ILS x160 (small/targeted/large drops + fresh-every-6,
elite + 2%-walk acceptance), 16 restarts per (instance, f).

Usage:
  python pctsp_ils_reference.py            # full run
  python pctsp_ils_reference.py --smoke    # quick code check
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import op_ils_reference as g  # dist_matrix, compress (2-opt/Or-opt)

EPS = 1e-12
N = 50
K_N = 3.0
QUOTA = 1.0
F_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]
CAL_LO, CAL_HI = 0.01, 20.0
N_INST = 24
N_RESTARTS = 16
ILS_ROUNDS = 160
WALK_TOL = 0.02          # accept walk states within 2% of elite objective
DROP_SMALL = (0.10, 0.30)
DROP_BIG = (0.40, 0.60)
FRESH_EVERY = 6
GAMMA_SET = [(1.0, 1.0), (0.5, 1.0), (2.0, 1.0), (4.0, 0.5), (1.0, 0.2)]
# fresh seed series
INST_SEED0 = 33000
RESTART_SEED0 = 43000


def gen_instance(seed):
    rng = np.random.default_rng(seed)
    depot = rng.random(2)
    loc = rng.random((N, 2))
    prize = rng.random(N) * (4.0 / N)
    beta_base = rng.random(N)          # CRN across f
    return depot, loc, prize, beta_base


def objective(D, pen_full, t):
    unv = np.ones(len(pen_full), bool)
    unv[t] = False
    return g.tour_len(D, t) + float(pen_full[unv].sum())


def best_insertion(D, t, cand):
    """detour matrix [len(t), len(cand)] of inserting cand after position k."""
    succ = np.roll(t, -1)
    det = D[np.ix_(t, cand)] + D[np.ix_(cand, succ)].T - D[t, succ][:, None]
    return np.maximum(det, 0.0)


def quota_fill(D, prize_full, t, unv, rng, gp, gd, noise):
    """Insert nodes by prize^gp/detour^gd until quota met."""
    coll = float(prize_full[t].sum())
    while coll < QUOTA - EPS:
        cand = np.where(unv)[0]
        if cand.size == 0:
            break
        det = best_insertion(D, t, cand)
        score = (prize_full[cand][None, :] ** gp) / ((det + 1e-9) ** gd)
        if rng is not None and noise > 0:
            score = score * rng.uniform(1 - noise, 1 + noise, size=score.shape)
        k, c = np.unravel_index(int(np.argmax(score)), score.shape)
        j = int(cand[c])
        t = np.insert(t, k + 1, j)
        unv[j] = False
        coll += prize_full[j]
    return t, unv


def improving_adds(D, pen_full, t, unv, rng=None, noise=0.0):
    while True:
        cand = np.where(unv)[0]
        if cand.size == 0:
            return t, unv
        det = best_insertion(D, t, cand)
        gain = pen_full[cand][None, :] - det        # >0 improves objective
        if rng is not None and noise > 0:
            gain = gain * rng.uniform(1 - noise, 1 + noise, size=gain.shape)
        k, c = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if gain[k, c] <= 1e-10:
            return t, unv
        j = int(cand[c])
        t = np.insert(t, k + 1, j)
        unv[j] = False


def improving_drops(D, prize_full, pen_full, t, unv):
    while len(t) > 2:
        coll = float(prize_full[t].sum())
        pred = np.roll(t, 1)
        succ = np.roll(t, -1)
        saving = D[pred, t] + D[t, succ] - D[pred, succ]
        gain = saving - pen_full[t]                 # >0 improves objective
        gain[0] = -np.inf                           # depot stays
        ok = (coll - prize_full[t]) >= QUOTA - EPS  # quota-safe
        gain = np.where(ok, gain, -np.inf)
        k = int(np.argmax(gain))
        if gain[k] <= 1e-10:
            return t, unv
        unv[t[k]] = True
        t = np.delete(t, k)


def improving_swaps(D, prize_full, pen_full, t, unv, max_iter=60):
    for _ in range(max_iter):
        cand = np.where(unv)[0]
        if cand.size == 0 or len(t) < 2:
            return t, unv
        coll = float(prize_full[t].sum())
        pred = np.roll(t, 1)
        succ = np.roll(t, -1)
        dl = D[np.ix_(pred, cand)] + D[np.ix_(cand, succ)].T \
            - (D[pred, t] + D[t, succ])[:, None]
        dobj = dl + pen_full[t][:, None] - pen_full[cand][None, :]
        feas = (coll - prize_full[t][:, None] + prize_full[cand][None, :]) >= QUOTA - EPS
        dobj = np.where(feas, dobj, np.inf)
        dobj[0, :] = np.inf                         # never swap the depot
        k, c = np.unravel_index(int(np.argmin(dobj)), dobj.shape)
        if dobj[k, c] >= -1e-10:
            return t, unv
        j = int(cand[c])
        unv[t[k]] = True
        unv[j] = False
        t = t.copy()
        t[k] = j


def descend(D, prize_full, pen_full, t, unv, max_rounds=40):
    obj = objective(D, pen_full, t)
    for _ in range(max_rounds):
        t, _ = g.compress(D, t)
        t, unv = improving_adds(D, pen_full, t, unv)
        t, unv = improving_drops(D, prize_full, pen_full, t, unv)
        t, unv = improving_swaps(D, prize_full, pen_full, t, unv)
        new = objective(D, pen_full, t)
        if new > obj - 1e-10:
            return t, unv, new
        obj = new
    return t, unv, obj


def fresh(D, prize_full, pen_full, rng, shape=None):
    m = D.shape[0]
    t = np.array([0], dtype=int)
    unv = np.ones(m, bool)
    unv[0] = False
    if shape is None:
        gp, gd = GAMMA_SET[rng.integers(len(GAMMA_SET))]
        noise = 0.15
    else:
        gp, gd = shape
        noise = 0.0
    t, unv = quota_fill(D, prize_full, t, unv, rng, gp, gd, noise)
    t, unv = improving_adds(D, pen_full, t, unv, rng=rng, noise=noise)
    return descend(D, prize_full, pen_full, t, unv)


def solve_restart(D, prize_full, pen_full, seed, ils_rounds):
    rng = np.random.default_rng(seed)
    m = D.shape[0]
    best_t, best_obj = None, np.inf
    for shape in GAMMA_SET:
        t2, unv2, o2 = fresh(D, prize_full, pen_full, rng, shape=shape)
        if o2 < best_obj - 1e-12:
            best_obj, best_t = o2, t2.copy()
    t = best_t.copy()
    unv = np.ones(m, bool)
    unv[t] = False
    for rd in range(ils_rounds):
        vis = t[1:]
        if rd % FRESH_EVERY == FRESH_EVERY - 1:
            t2, unv2, o2 = fresh(D, prize_full, pen_full, rng)
        else:
            mode = rd % 3
            if mode == 2:
                frac = rng.uniform(*DROP_BIG)
            else:
                frac = rng.uniform(*DROP_SMALL)
            if len(vis) == 0:
                continue
            ndrop = min(max(1, int(round(frac * len(vis)))), len(vis))
            if mode == 1:      # targeted: evict worst (saving - beta) nodes
                pred = np.roll(t, 1)
                succ = np.roll(t, -1)
                sav = D[pred, t] + D[t, succ] - D[pred, succ]
                score = (pen_full[t] - sav) * rng.uniform(0.6, 1.4, size=len(t))
                score[0] = np.inf
                drop = t[np.argsort(score)[:min(ndrop, 5)]]
            else:
                drop = rng.choice(vis, size=ndrop, replace=False)
            t2 = t[~np.isin(t, drop)]
            unv2 = np.ones(m, bool)
            unv2[t2] = False
            t2, unv2 = quota_fill(D, prize_full, t2, unv2, rng, 1.0, 1.0, 0.20)
            t2, unv2 = improving_adds(D, pen_full, t2, unv2, rng=rng, noise=0.20)
            t2, unv2, o2 = descend(D, prize_full, pen_full, t2, unv2)
        if o2 < best_obj - 1e-12:
            best_obj, best_t = o2, t2.copy()
        if o2 <= best_obj * (1.0 + 2.0 * WALK_TOL) + 1e-12:
            t, unv = t2, unv2
        else:
            t = best_t.copy()
            unv = np.ones(m, bool)
            unv[t] = False
    return best_obj, best_t


def binding_markers(D, prize_full, pen_full, t):
    """Economic binding = some visited node's removal improves the
    objective but is blocked by the quota. The strict marker is whether the
    collected prize sits at the quota."""
    coll = float(prize_full[t].sum())
    pred = np.roll(t, 1)
    succ = np.roll(t, -1)
    saving = D[pred, t] + D[t, succ] - D[pred, succ]
    gain = saving - pen_full[t]
    gain[0] = -np.inf
    blocked = (gain > 1e-9) & ((coll - prize_full[t]) < QUOTA - EPS)
    return bool(blocked.any()), bool(coll <= QUOTA + 1e-6), coll


def run_cell(args):
    i, label, f, restarts, ils_rounds, smoke = args
    seed = (99200 + i) if smoke else (INST_SEED0 + i)
    depot, loc, prize, beta_base = gen_instance(seed)
    D = g.dist_matrix(depot, loc)
    prize_full = np.concatenate([[0.0], prize])
    if label == "cal_hi":
        # constant penalties -- visit-all is provably optimal (each
        # penalty f*K/n = 1.2 exceeds any marginal detour), so the
        # certificate premise holds; the scaled-U(0,x) draw does not
        # guarantee this (min of 50 uniforms stays near 0).
        pen_full = np.concatenate([[0.0], np.full(N, f * K_N / N)])
    else:
        pen_full = np.concatenate([[0.0], beta_base * (f * K_N / N)])
    assert prize.sum() >= QUOTA + 0.1, "degenerate instance: total prize near quota"
    rest_objs, rest_v = [], []
    best_obj, best_t = np.inf, None
    for r in range(restarts):
        o, t = solve_restart(D, prize_full, pen_full,
                             RESTART_SEED0 + i * 100 + r, ils_rounds)
        assert t[0] == 0 and len(np.unique(t)) == len(t)
        assert prize_full[t].sum() >= QUOTA - 1e-9, "quota violated"
        rest_objs.append(o)
        rest_v.append((len(t) - 1) / N)
        if o < best_obj - 1e-12:
            best_obj, best_t = o, t.copy()
    bind_econ, bind_strict, coll = binding_markers(D, prize_full, pen_full, best_t)
    return dict(inst=i, label=label, f=f,
                best_obj=round(best_obj, 6),
                tour_len=round(g.tour_len(D, best_t), 6),
                pen_sum=round(best_obj - g.tour_len(D, best_t), 6),
                visit_rate=(len(best_t) - 1) / N,
                collected=round(coll, 6),
                binding_econ=bind_econ, binding_strict=bind_strict,
                restart_objs=[round(o, 6) for o in rest_objs])


def aggregate(cells):
    by = {}
    for c in cells:
        by.setdefault(c["label"], {})[c["inst"]] = c
    insts = sorted({c["inst"] for c in cells})
    v = {lab: np.array([by[lab][i]["visit_rate"] for i in insts]) for lab in by}
    R = {}
    # calibration-arm certificates
    lo_ok = sum(1 for i in insts
                if QUOTA <= by["cal_lo"][i]["collected"] <= QUOTA * 1.10)
    hi_ok = sum(1 for i in insts if by["cal_hi"][i]["visit_rate"] >= 0.99)
    R["cal_certificate"] = dict(cal_lo_ok=lo_ok, cal_hi_ok=hi_ok,
                                PASS=(lo_ok >= 22 and hi_ok >= 22))
    # monotonicity over F_GRID
    viol = tot = 0
    labs = [f"f{f:g}" for f in F_GRID]
    for a, b in zip(labs, labs[1:]):
        d = v[b] - v[a]
        viol += int((d < -0.005).sum())
        tot += len(d)
    R["monotone_visit"] = dict(violations=viol, pairs=tot, PASS=(viol / tot <= 0.05))
    # half-split at f=3
    hs = []
    for i in insts:
        ro = by["f3"][i]["restart_objs"]
        hs.append(abs(min(ro[:8]) - min(ro[8:])) / abs(min(ro)))
    R["half_split_noise"] = dict(mean=float(np.mean(hs)), PASS=(np.mean(hs) <= 0.01))
    # calibration criteria
    v3 = v["f3"]
    R["std_visit_rate"] = dict(v_std=float(v3.mean()), PASS=(0.35 <= v3.mean() <= 0.75))
    vf = np.array([v[f"f{f:g}"].mean() for f in F_GRID])
    f_sat = None
    for a in range(len(F_GRID) - 1):
        if vf[a] < 0.95 <= vf[a + 1]:
            f_sat = F_GRID[a] + (0.95 - vf[a]) / (vf[a + 1] - vf[a]) * (F_GRID[a + 1] - F_GRID[a])
            break
    if vf[0] >= 0.95:
        f_sat = F_GRID[0]
    R["saturation"] = dict(f_sat=(round(f_sat, 3) if f_sat else ">8"),
                           PASS=(f_sat is None or f_sat >= 4.0))
    bind = np.mean([by["f3"][i]["binding_econ"] for i in insts])
    bind_strict = np.mean([by["f3"][i]["binding_strict"] for i in insts])
    R["binding"] = dict(binding_econ=float(bind), binding_strict=float(bind_strict),
                        PASS=(0.10 <= bind <= 0.90))
    R["visit_spread"] = dict(delta=float(v["f4"].mean() - v["f2"].mean()),
                             PASS=(v["f4"].mean() - v["f2"].mean() >= 0.10))
    outl = int(((v3 < 0.25) | (v3 > 0.90)).sum())
    R["outliers"] = dict(outliers=outl, PASS=(outl <= 4))
    R["STATUS"] = "PASS" if all(R[k]["PASS"] for k in
                                ("cal_certificate", "monotone_visit", "half_split_noise",
                                 "std_visit_rate", "saturation", "binding", "visit_spread",
                                 "outliers")) else "FAIL"
    R["v_curve"] = {f"f{f:g}": round(float(v[f'f{f:g}'].mean()), 4) for f in F_GRID}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    n_inst = 2 if args.smoke else N_INST
    restarts = 4 if args.smoke else N_RESTARTS
    ils = 40 if args.smoke else ILS_ROUNDS
    grid = ([1.0, 3.0, 8.0] if args.smoke else F_GRID)
    cells = [(i, f"f{f:g}", f, restarts, ils, args.smoke)
             for i in range(n_inst) for f in grid]
    cells += [(i, "cal_lo", CAL_LO, restarts, ils, args.smoke) for i in range(n_inst)]
    cells += [(i, "cal_hi", CAL_HI, restarts, ils, args.smoke) for i in range(n_inst)]
    t0 = time.time()
    with Pool(processes=args.workers) as pool:
        out = []
        for k, rec in enumerate(pool.imap_unordered(run_cell, cells, chunksize=1)):
            out.append(rec)
            if (k + 1) % 24 == 0:
                print(f"{k + 1}/{len(cells)} cells, {time.time() - t0:.0f}s", flush=True)
    def _np(o):
        return o.item() if hasattr(o, "item") else str(o)

    out_root = os.environ.get("OUTPUT_ROOT", "outputs")
    os.makedirs(out_root, exist_ok=True)
    dst = os.path.join(out_root, "pctsp_ils_reference_result"
                       + ("_smoke" if args.smoke else "") + ".json")
    result = dict(cells=out, wall_s=round(time.time() - t0, 1))
    if not args.smoke:
        result["aggregation"] = aggregate(out)
        print(json.dumps(result["aggregation"], indent=2, default=_np))
    else:
        for lab in sorted({c["label"] for c in out}):
            vs = [c["visit_rate"] for c in out if c["label"] == lab]
            cs = [c["collected"] for c in out if c["label"] == lab]
            print(f"smoke {lab}: v={np.mean(vs):.3f} collected={np.mean(cs):.3f}")
    with open(dst, "w") as fjson:
        json.dump(result, fjson, default=_np)
    print("PCTSP_ILS_REFERENCE_DONE", dst, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
