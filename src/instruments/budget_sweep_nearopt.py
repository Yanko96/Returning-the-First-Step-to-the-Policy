"""Budget-sweep calibration — per-tier near-optimal start sets (predictor), CPU.
Conventions (see the paper's appendix): reference ILS solver verbatim, 16 restarts, eps=0.995,
first 1000 canonical instances with budget overridden to tier T, restart
seeds 200000 + tier_idx*100000 + idx*100 + r (tier_idx by ascending T over
the full 5-tier list; anchor T=3.0 reuses the existing instrument file and
is not re-run here).
"""
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import op_ils_reference as ils_ref  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
TIERS = [1.5, 2.0, 3.0, 4.0, 4.5]  # tier_idx by this order
N_FIRST = 1000
N_RESTARTS = 16
NEAR_OPT = 0.995

with open(PKL, "rb") as f:
    DATA = pickle.load(f)

T = float(sys.argv[1])
TIER_IDX = TIERS.index(T)
SEED0 = 200000 + TIER_IDX * 100000


def job(args):
    idx, r = args
    depot, loc, prize, _ = DATA[idx]
    D = ils_ref.dist_matrix(np.asarray(depot, float), np.asarray(loc, float))
    prize_full = np.concatenate([[0.0], np.asarray(prize, float)])
    p, t, L = ils_ref.op_solve_restart(D, prize_full, T, SEED0 + idx * 100 + r)
    assert L <= T + 1e-9 and t[0] == 0
    return idx, r, p, ([int(t[1]), int(t[-1])] if len(t) > 1 else []), len(t) - 1


def main():
    t0 = time.time()
    jobs = [(i, r) for i in range(N_FIRST) for r in range(N_RESTARTS)]
    acc = {i: [] for i in range(N_FIRST)}
    with Pool(processes=14) as pool:
        for k, (i, r, p, pair, nv) in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            acc[i].append((p, pair, nv))
            if (k + 1) % 2000 == 0:
                print(f"T={T}: {k + 1}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True)
    out = []
    for i in range(N_FIRST):
        rp = acc[i]
        best = max(p for p, _, _ in rp)
        sset = sorted({s for p, pair, _ in rp if p >= NEAR_OPT * best for s in pair})
        nv_best = max(nv for p, _, nv in rp if p >= best - 1e-12)
        out.append(dict(idx=i, best_prize=round(best, 6),
                        nearopt_start_set=sset, best_visits=nv_best))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    dst = os.path.join(OUTPUT_ROOT, f"nearopt_T{T:g}_first1000.json")
    with open(dst, "w") as f:
        json.dump(dict(pkl_md5="47a828139a9fcf1e642a060f40837d9e", budget=T,
                       n_restarts=N_RESTARTS, nearopt_frac=NEAR_OPT, seed0=SEED0,
                       instances=out), f)
    sizes = [len(r["nearopt_start_set"]) for r in out]
    vf = [r["best_visits"] / 50.0 for r in out]
    print(f"T={T}: |S| mean {np.mean(sizes):.2f} min {min(sizes)} max {max(sizes)}; "
          f"ref prize {np.mean([r['best_prize'] for r in out]):.4f}; "
          f"visit frac {np.mean(vf):.3f}")
    print(f"NEAROPT_T{T:g}_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
