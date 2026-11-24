"""Budget-sweep calibration — budget tiers (T -> visit fraction), CPU.

Sets the 5 budget tiers ONLY. No model, no outcome, no benefit numbers are
computed here (visit fraction of ILS tours is a property of the instance
distribution, not of any treatment).

Method: canonical OP50 test instances 5000..5063 (away from the 0..999
evaluation subsets), budget overridden to each T on a grid; the reference
ILS solver verbatim, 4 restarts per (instance, T), seeds 70000 + idx*1000 + gi*10 + r;
visit fraction = (len(best tour) - 1) / 50.
"""
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "instruments"))
import op_ils_reference as ils_ref  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
I0, NI = 5000, 64
GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]
N_RESTARTS = 4
SEED0 = 70000

with open(PKL, "rb") as f:
    DATA = pickle.load(f)


def job(args):
    idx, gi = args
    depot, loc, prize, _ = DATA[idx]
    D = ils_ref.dist_matrix(np.asarray(depot, float), np.asarray(loc, float))
    prize_full = np.concatenate([[0.0], np.asarray(prize, float)])
    B = GRID[gi]
    best_p, best_t = -1.0, None
    for r in range(N_RESTARTS):
        p, t, L = ils_ref.op_solve_restart(D, prize_full, B,
                                           SEED0 + (idx - I0) * 1000 + gi * 10 + r)
        if p > best_p + 1e-12:
            best_p, best_t = p, t
    return gi, idx, (len(best_t) - 1) / 50.0


def main():
    t0 = time.time()
    jobs = [(i, gi) for i in range(I0, I0 + NI) for gi in range(len(GRID))]
    acc = {gi: [] for gi in range(len(GRID))}
    with Pool(processes=14) as pool:
        for k, (gi, idx, vf) in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            acc[gi].append(vf)
            if (k + 1) % 80 == 0:
                print(f"{k + 1}/{len(jobs)}, {time.time() - t0:.0f}s", flush=True)
    out = {}
    for gi, T in enumerate(GRID):
        v = np.array(acc[gi])
        out[str(T)] = dict(mean_visit_frac=float(v.mean()), std=float(v.std(ddof=1)))
        print(f"T={T}: visit fraction {v.mean():.3f} +/- {v.std(ddof=1):.3f}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "budget_calibration.json"), "w") as f:
        json.dump(dict(instances=f"{I0}..{I0 + NI - 1}", n_restarts=N_RESTARTS,
                       seed0=SEED0, grid=out), f, indent=2)
    print(f"CALIBRATION_DONE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
