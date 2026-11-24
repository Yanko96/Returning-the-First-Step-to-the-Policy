"""Near-optimal start sets, first 200 canonical OP100 instances.
Spec: identical to the OP50/OPLib runs
(reference-ILS gamma-portfolio x160, 16 restarts, 99.5% threshold,
depot-adjacent start pairs, sampling -> lower bound). Seeds 80000-series."""
import json
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np

import op_ils_reference as ils_ref

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "op", "op_dist100_test_seed1234.pkl")
N_FIRST = 200
N_RESTARTS = 16
NEAR_OPT = 0.995
SEED0 = 80000

with open(PKL, "rb") as f:
    DATA = pickle.load(f)


def job(args):
    idx, r = args
    depot, loc, prize, max_length = DATA[idx]
    D = ils_ref.dist_matrix(np.asarray(depot, float), np.asarray(loc, float))
    prize_full = np.concatenate([[0.0], np.asarray(prize, float)])
    bp, bt, bL = ils_ref.op_solve_restart(D, prize_full, float(max_length),
                                          SEED0 + idx * 100 + r)
    return idx, r, bp, ([int(bt[1]), int(bt[-1])] if len(bt) > 1 else [])


def main():
    t0 = time.time()
    jobs = [(i, r) for i in range(N_FIRST) for r in range(N_RESTARTS)]
    acc = {i: [] for i in range(N_FIRST)}
    with Pool(processes=96) as pool:
        for k, (i, r, bp, pair) in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            acc[i].append((bp, pair))
            if (k + 1) % 400 == 0:
                print(f"{k + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    out = []
    for i in range(N_FIRST):
        rp = acc[i]
        best = max(b for b, _ in rp)
        sset = sorted({s for b, pair in rp if b >= NEAR_OPT * best for s in pair})
        out.append(dict(idx=i, best_prize=round(best, 6),
                        restart_prizes=[round(b, 6) for b, _ in rp],
                        restart_starts=[pair for _, pair in rp],
                        nearopt_start_set=sset))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "op100_nearopt.json"), "w") as f:
        json.dump(dict(pkl_md5="53eeff929a4339563a044b9fdee5fd69",
                       n_restarts=N_RESTARTS, nearopt_frac=NEAR_OPT, seed0=SEED0,
                       instances=out), f)
    sizes = [len(r["nearopt_start_set"]) for r in out]
    print(f"start-set sizes: mean {np.mean(sizes):.2f} min {min(sizes)} max {max(sizes)}")
    print(f"NEAROPT_ANALYSIS_COMPLETE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
