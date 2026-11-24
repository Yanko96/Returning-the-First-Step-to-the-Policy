"""Budget-sweep calibration secondary predictor — model-free start-stakes.

Self-contained ILS-lite used for BOTH arms (free / forced-first-node), so
solver quality biases cancel in the ratio:

  stakes_i = 1 - mean_s best_forced(i, s) / max(best_free(i), max_s best_forced(i, s))

Forced arm keeps t[1] = s invariant through every move (insertions at
positions >= 2, 2-opt segments starting at index >= 2, s never dropped).
200 instances/tier, 8 forced starts from the feasible set {s: 2 d(0,s) <= T}
(uniform without replacement), 2 restarts per forced start, 8 free restarts.
Seeds: 300000 + tier_idx*100000 + idx*100 + k (forced), +9000 offset (free).
Descriptive only.
"""
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
TIERS = [1.5, 2.0, 3.0, 4.0, 4.5]
N_INST = 200
N_FORCED = 8
R_FORCED = 2
R_FREE = 8
LITE_ROUNDS = 40

with open(PKL, "rb") as f:
    DATA = pickle.load(f)


def tour_len(D, t):
    if len(t) == 1:
        return 0.0
    idx = np.asarray(t + [0])
    return float(D[idx[:-1], idx[1:]].sum())


def greedy_fill(D, prize, B, t, rng, lock_first):
    """Insert unvisited nodes (best prize/detour, sampled) while budget allows."""
    t = list(t)
    L = tour_len(D, t)
    unv = set(range(1, D.shape[0])) - set(t)
    while unv:
        cand = rng.choice(sorted(unv), size=min(6, len(unv)), replace=False)
        best = None
        for j in cand:
            lo = 2 if lock_first else 1
            for pos in range(lo, len(t) + 1):
                a = t[pos - 1]
                b = t[pos] if pos < len(t) else 0
                dl = D[a, j] + D[j, b] - D[a, b]
                if L + dl <= B + 1e-12:
                    score = prize[j] / max(dl, 1e-9)
                    if best is None or score > best[0]:
                        best = (score, j, pos, dl)
        if best is None:
            break
        _, j, pos, dl = best
        t.insert(pos, j)
        L += dl
        unv.discard(j)
    return t, L


def two_opt(D, B, t, L, lock_first):
    imp = True
    while imp:
        imp = False
        lo = 2 if lock_first else 1
        k = len(t)
        for i in range(lo, k - 1):
            for j in range(i + 1, k):
                a, b = t[i - 1], t[i]
                c, d = t[j], t[j + 1] if j + 1 < k else 0
                delta = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                if delta < -1e-12:
                    t[i:j + 1] = t[i:j + 1][::-1]
                    L += delta
                    imp = True
    return t, L


def lite_solve(D, prize, B, seed, forced=None):
    """One ILS-lite restart. forced = node index required at t[1]."""
    rng = np.random.default_rng(seed)
    lock = forced is not None
    if lock:
        if 2 * D[0, forced] > B + 1e-12:
            return 0.0
        t = [0, int(forced)]
    else:
        t = [0]
    t, L = greedy_fill(D, prize, B, t, rng, lock)
    t, L = two_opt(D, B, t, L, lock)
    t, L = greedy_fill(D, prize, B, t, rng, lock)
    best_t, best_p = list(t), float(prize[t].sum())
    for _ in range(LITE_ROUNDS):
        t = list(best_t)
        interior = t[2:] if lock else t[1:]
        if interior:
            nd = max(1, int(0.2 * len(interior)))
            drop = set(rng.choice(interior, size=min(nd, len(interior)),
                                  replace=False).tolist())
            t = [x for x in t if x not in drop or x == 0 or (lock and x == t[1])]
        L = tour_len(D, t)
        t, L = greedy_fill(D, prize, B, t, rng, lock)
        t, L = two_opt(D, B, t, L, lock)
        t, L = greedy_fill(D, prize, B, t, rng, lock)
        p = float(prize[t].sum())
        if p > best_p + 1e-12:
            best_t, best_p = list(t), p
    assert tour_len(D, best_t) <= B + 1e-9
    if lock:
        assert best_t[1] == forced
    return best_p


def job(args):
    tier_idx, idx = args
    T = TIERS[tier_idx]
    depot, loc, prize_raw, _ = DATA[idx]
    depot = np.asarray(depot, float)
    loc = np.asarray(loc, float)
    xy = np.vstack([depot[None, :], loc])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    prize = np.concatenate([[0.0], np.asarray(prize_raw, float)])
    seed0 = 300000 + tier_idx * 100000 + idx * 100
    rng = np.random.default_rng(seed0)
    feas = [s for s in range(1, 51) if 2 * D[0, s] <= T + 1e-12]
    forced = rng.choice(feas, size=min(N_FORCED, len(feas)), replace=False)
    best_free = max(lite_solve(D, prize, T, seed0 + 9000 + r) for r in range(R_FREE))
    fvals = []
    for k, s in enumerate(forced):
        fvals.append(max(lite_solve(D, prize, T, seed0 + k * 10 + r, forced=int(s))
                         for r in range(R_FORCED)))
    denom = max(best_free, max(fvals))
    stakes = 1.0 - float(np.mean(fvals)) / max(denom, 1e-9)
    return tier_idx, idx, stakes, best_free, float(np.mean(fvals))


def main():
    t0 = time.time()
    jobs = [(ti, i) for ti in range(len(TIERS)) for i in range(N_INST)]
    acc = {ti: {} for ti in range(len(TIERS))}
    with Pool(processes=14) as pool:
        for k, (ti, i, st, bf, mf) in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            acc[ti][i] = dict(stakes=st, best_free=bf, mean_forced=mf)
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True)
    out = {}
    for ti, T in enumerate(TIERS):
        sv = np.array([acc[ti][i]["stakes"] for i in range(N_INST)])
        out[str(T)] = dict(stakes_mean=float(sv.mean()), stakes_std=float(sv.std(ddof=1)),
                           per_instance=[acc[ti][i] for i in range(N_INST)])
        print(f"T={T}: stakes {sv.mean():.4f} +/- {sv.std(ddof=1):.4f}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "stakes_first200.json"), "w") as f:
        json.dump(dict(n_inst=N_INST, n_forced=N_FORCED, r_forced=R_FORCED,
                       r_free=R_FREE, lite_rounds=LITE_ROUNDS, seed0_base=300000,
                       tiers=out), f)
    print(f"STAKES_DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
