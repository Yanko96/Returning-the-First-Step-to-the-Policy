"""PCTSP near-optimal start sets, first 200 canonical instances.
Spec (see the paper's appendix): reference PCTSP ILS solver verbatim (validated), 16 restarts,
seeds 90000-series, near-opt = obj <= 1.005 x best, depot-adjacent pairs,
sampling -> lower bound."""
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

import op_ils_reference as g
import pctsp_ils_reference as pg

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "pctsp", "pctsp50_test_seed1234.pkl")
N_FIRST = 200
N_RESTARTS = 16
NEAR_OPT = 1.005
SEED0 = 90000

with open(PKL, "rb") as f:
    DATA = pickle.load(f)


def job(args):
    idx, r = args
    depot, loc, penalty, det_prize, _ = DATA[idx]
    D = g.dist_matrix(np.asarray(depot, float), np.asarray(loc, float))
    prize_full = np.concatenate([[0.0], np.asarray(det_prize, float)])
    pen_full = np.concatenate([[0.0], np.asarray(penalty, float)])
    obj, t = pg.solve_restart(D, prize_full, pen_full,
                              SEED0 + idx * 100 + r, pg.ILS_ROUNDS)
    return idx, r, obj, ([int(t[1]), int(t[-1])] if len(t) > 1 else [])


def main():
    t0 = time.time()
    jobs = [(i, r) for i in range(N_FIRST) for r in range(N_RESTARTS)]
    acc = {i: [] for i in range(N_FIRST)}
    with Pool(processes=80) as pool:
        for k, (i, r, obj, pair) in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            acc[i].append((obj, pair))
            if (k + 1) % 400 == 0:
                print(f"{k + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    out = []
    for i in range(N_FIRST):
        rp = acc[i]
        best = min(o for o, _ in rp)
        sset = sorted({s for o, pair in rp if o <= NEAR_OPT * best for s in pair})
        out.append(dict(idx=i, best_obj=round(best, 6),
                        restart_objs=[round(o, 6) for o, _ in rp],
                        restart_starts=[pair for _, pair in rp],
                        nearopt_start_set=sset))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "pctsp_nearopt.json"), "w") as f:
        json.dump(dict(pkl_md5="df23990603d21aa25ed41f87640db4fa",
                       n_restarts=N_RESTARTS, nearopt_frac=NEAR_OPT, seed0=SEED0,
                       metric="minimization mirror of the OP 99.5% rule",
                       instances=out), f)
    sizes = [len(r["nearopt_start_set"]) for r in out]
    print(f"start-set sizes: mean {np.mean(sizes):.2f} min {min(sizes)} max {max(sizes)}")
    print(f"PCTSP_NEAROPT_DONE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
