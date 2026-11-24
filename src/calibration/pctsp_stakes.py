"""Start-stakes transplanted to PCTSP50 (minimization
mirror of the OP instrument, budget_sweep_stakes.py).

Mirror mapping (OP -> PCTSP):
  objective        max prize within budget  ->  min length + unvisited penalties,
                                                collected prize >= 1
  forced arm       t[1] = s fixed through every move (insertions at
                   positions >= 2, 2-opt segments starting at index >= 2,
                   s never dropped)
  feasible starts  {s: 2 d(0,s) <= T}  ->  all nodes (no budget limit)
  stakes_i         1 - mean_s best_forced / ref_best   (deficit vs anchor)
                   ->  mean_s best_forced / ref_best - 1, clip [0,1]
  denominator      ILS-reference 16-restart best objective per instance
                   (pctsp_nearopt.json best_obj; the paper's stakes definition)
  params           N_INST=200, N_FORCED=8, R_FORCED=2, R_FREE=8,
                   LITE_ROUNDS=40 (pinned equal to the OP instrument)
  seeds            810000 + idx*100 (+9000 free offset, +k*10 forced)

The lite solver mirrors budget_sweep_stakes' ILS-lite structure: sampled greedy
fill to quota, improving penalty-vs-detour adds, lock-aware 2-opt,
quota-safe drops, 40 drop-refill rounds. Same solver for both arms, so
solver-quality bias largely cancels; residual lite-vs-reference drift is
reported (lite_free_over_ref).
"""
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "pctsp", "pctsp50_test_seed1234.pkl")
NEAROPT = os.path.join(ARTIFACT_ROOT, "nearopt", "pctsp_nearopt.json")
QUOTA = 1.0
N_INST = 200
N_FORCED = 8
R_FORCED = 2
R_FREE = 8
LITE_ROUNDS = 40
SEED0_BASE = 810000

_G = {}


def _init():
    with open(PKL, "rb") as f:
        _G["data"] = pickle.load(f)
    _G["ils_ref"] = {r["idx"]: r["best_obj"]
                     for r in json.load(open(NEAROPT))["instances"]}


def tour_len(D, t):
    if len(t) == 1:
        return 0.0
    idx = np.asarray(t + [0])
    return float(D[idx[:-1], idx[1:]].sum())


def objective(D, pen, t):
    unv = np.ones(len(pen), bool)
    unv[np.asarray(t)] = False
    return tour_len(D, t) + float(pen[unv].sum())


def quota_fill(D, prize, t, rng, lock):
    """Sampled greedy insertion (prize/detour) until quota met."""
    t = list(t)
    unv = set(range(1, D.shape[0])) - set(t)
    coll = float(prize[t].sum())
    while coll < QUOTA - 1e-12 and unv:
        cand = rng.choice(sorted(unv), size=min(6, len(unv)), replace=False)
        best = None
        for j in cand:
            lo = 2 if lock else 1
            for pos in range(lo, len(t) + 1):
                a = t[pos - 1]
                b = t[pos] if pos < len(t) else 0
                dl = D[a, j] + D[j, b] - D[a, b]
                score = prize[j] / max(dl, 1e-9)
                if best is None or score > best[0]:
                    best = (score, j, pos)
        _, j, pos = best
        t.insert(pos, j)
        coll += prize[j]
        unv.discard(j)
    return t


def improving_adds(D, pen, t, lock):
    """Insert any unvisited node whose penalty exceeds its best detour."""
    t = list(t)
    unv = set(range(1, D.shape[0])) - set(t)
    while True:
        best = None
        lo = 2 if lock else 1
        for j in unv:
            for pos in range(lo, len(t) + 1):
                a = t[pos - 1]
                b = t[pos] if pos < len(t) else 0
                dl = D[a, j] + D[j, b] - D[a, b]
                gain = pen[j] - dl
                if gain > 1e-10 and (best is None or gain > best[0]):
                    best = (gain, j, pos)
        if best is None:
            return t
        _, j, pos = best
        t.insert(pos, j)
        unv.discard(j)


def improving_drops(D, prize, pen, t, lock):
    """Remove visited nodes whose detour saving beats their penalty,
    quota-safe; depot (and the locked first node) never dropped."""
    t = list(t)
    while len(t) > (2 if lock else 1):
        coll = float(prize[t].sum())
        best = None
        for k in range(2 if lock else 1, len(t)):
            a = t[k - 1]
            b = t[k + 1] if k + 1 < len(t) else 0
            sav = D[a, t[k]] + D[t[k], b] - D[a, b]
            gain = sav - pen[t[k]]
            if gain > 1e-10 and coll - prize[t[k]] >= QUOTA - 1e-12:
                if best is None or gain > best[0]:
                    best = (gain, k)
        if best is None:
            return t
        t.pop(best[1])
    return t


def two_opt(D, t, lock):
    imp = True
    while imp:
        imp = False
        lo = 2 if lock else 1
        k = len(t)
        for i in range(lo, k - 1):
            for j in range(i + 1, k):
                a, b = t[i - 1], t[i]
                c, d = t[j], t[j + 1] if j + 1 < k else 0
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-12:
                    t[i:j + 1] = t[i:j + 1][::-1]
                    imp = True
    return t


def lite_solve(D, prize, pen, seed, forced=None):
    """One ILS-lite restart; returns best objective (minimization)."""
    rng = np.random.default_rng(seed)
    lock = forced is not None
    t = [0, int(forced)] if lock else [0]
    t = quota_fill(D, prize, t, rng, lock)
    t = improving_adds(D, pen, t, lock)
    t = two_opt(D, t, lock)
    t = improving_drops(D, prize, pen, t, lock)
    best_t, best_o = list(t), objective(D, pen, t)
    for _ in range(LITE_ROUNDS):
        t = list(best_t)
        interior = t[2:] if lock else t[1:]
        if interior:
            nd = max(1, int(0.2 * len(interior)))
            drop = set(rng.choice(interior, size=min(nd, len(interior)),
                                  replace=False).tolist())
            t = [x for x in t if x not in drop or x == 0 or (lock and x == t[1])]
        t = quota_fill(D, prize, t, rng, lock)
        t = improving_adds(D, pen, t, lock)
        t = two_opt(D, t, lock)
        t = improving_drops(D, prize, pen, t, lock)
        o = objective(D, pen, t)
        if o < best_o - 1e-12:
            best_t, best_o = list(t), o
    assert float(prize[best_t].sum()) >= QUOTA - 1e-9
    if lock:
        assert best_t[1] == forced
    return best_o


def job(idx):
    depot, loc, penalty, det_prize, _ = _G["data"][idx]
    xy = np.vstack([np.asarray(depot, float)[None, :], np.asarray(loc, float)])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    prize = np.concatenate([[0.0], np.asarray(det_prize, float)])
    pen = np.concatenate([[0.0], np.asarray(penalty, float)])
    n = D.shape[0] - 1
    seed0 = SEED0_BASE + idx * 100
    rng = np.random.default_rng(seed0)
    forced = rng.choice(np.arange(1, n + 1), size=N_FORCED, replace=False)
    best_free = min(lite_solve(D, prize, pen, seed0 + 9000 + r)
                    for r in range(R_FREE))
    fvals = [min(lite_solve(D, prize, pen, seed0 + k * 10 + r, forced=int(s))
                 for r in range(R_FORCED)) for k, s in enumerate(forced)]
    denom = _G["ils_ref"][idx]
    stakes = float(np.clip(float(np.mean(fvals)) / max(denom, 1e-9) - 1.0, 0.0, 1.0))
    return idx, stakes, best_free, float(np.mean(fvals)), denom


def main():
    t0 = time.time()
    recs = {}
    with Pool(processes=8, initializer=_init) as pool:
        for k, res in enumerate(pool.imap_unordered(job, range(N_INST), chunksize=1)):
            recs[res[0]] = res
            if (k + 1) % 20 == 0 or k == 0:
                print(f"{k + 1}/{N_INST}, {time.time() - t0:.0f}s", flush=True)
    rows = [dict(idx=i, stakes=recs[i][1], lite_free=recs[i][2],
                 mean_forced=recs[i][3], ref_best=recs[i][4])
            for i in range(N_INST)]
    sv = np.array([r["stakes"] for r in rows])
    lref = np.array([r["lite_free"] / max(r["ref_best"], 1e-9) for r in rows])
    raw = np.array([r["mean_forced"] / max(r["ref_best"], 1e-9) - 1.0 for r in rows])
    out = dict(problem="pctsp50", instrument="minimization mirror of budget_sweep_stakes",
               denominator="16-restart reference best objective (pctsp_nearopt)",
               n_inst=N_INST, n_forced=N_FORCED, r_forced=R_FORCED, r_free=R_FREE,
               lite_rounds=LITE_ROUNDS, seed0_base=SEED0_BASE,
               stakes_mean=float(sv.mean()), stakes_std=float(sv.std(ddof=1)),
               stakes_se=float(sv.std(ddof=1) / np.sqrt(len(sv))),
               lite_free_over_ref_mean=float(lref.mean()),
               clipped=int((raw < 0).sum()),
               per_instance=rows)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    path = os.path.join(OUTPUT_ROOT, "stakes_pctsp50.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"pctsp50: stakes {sv.mean():.4f} +/- {sv.std(ddof=1):.4f} "
          f"(se {out['stakes_se']:.4f}), lite/ref {lref.mean():.4f}, "
          f"clipped {out['clipped']}")
    print(f"PCTSP_STAKES_DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
