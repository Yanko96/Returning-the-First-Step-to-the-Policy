"""Scale ladder — budget calibration (T -> visit fraction) at scale n, local CPU.

Picks max_length @ visit-fraction ~= 0.55 per n for the scale ladder. No model,
no outcome, no benefit numbers: visit fraction of ILS tours is a property of the
instance distribution, not of any treatment. Reference ILS solver
(op_ils_reference.py) verbatim.

Multiprocessing: workers get DATA/GRID/... via Pool initializer
(spawn does not inherit globals set in __main__).
"""

import os as _os, sys as _sys  # resolve sibling package modules
_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('instruments',):
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import op_ils_reference as ils_ref  # noqa: E402

DATA = GRID = N = NREST = SEED0 = None


def _init(pkl, grid, n, nrest, seed0):
    global DATA, GRID, N, NREST, SEED0
    import pickle
    with open(pkl, "rb") as f:
        DATA = pickle.load(f)
    GRID, N, NREST, SEED0 = grid, n, nrest, seed0


def job(args):
    idx, gi = args
    depot, loc, prize, _ = DATA[idx]
    D = ils_ref.dist_matrix(np.asarray(depot, float), np.asarray(loc, float))
    prize_full = np.concatenate([[0.0], np.asarray(prize, float)])
    B = GRID[gi]
    best_p, best_t = -1.0, None
    for r in range(NREST):
        p, t, L = ils_ref.op_solve_restart(D, prize_full, B,
                                           SEED0 + idx * 1000 + gi * 10 + r)
        if p > best_p + 1e-12:
            best_p, best_t = p, t
    return gi, idx, (len(best_t) - 1) / float(N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--grid", required=True, help="comma-separated budgets")
    ap.add_argument("--ni", type=int, default=128, help="instances (idx 0..ni-1)")
    ap.add_argument("--nrest", type=int, default=4)
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--seed0", type=int, default=810000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    grid = [float(x) for x in args.grid.split(",")]
    t0 = time.time()
    jobs = [(i, gi) for i in range(args.ni) for gi in range(len(grid))]
    acc = {gi: [] for gi in range(len(grid))}
    with Pool(processes=args.procs, initializer=_init,
              initargs=(args.pkl, grid, args.n, args.nrest, args.seed0)) as pool:
        for k, (gi, idx, vf) in enumerate(pool.imap_unordered(job, jobs, chunksize=1)):
            acc[gi].append(vf)
            if (k + 1) % 50 == 0:
                print(f"{k + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)

    out = {}
    for gi, T in enumerate(grid):
        v = np.array(acc[gi])
        out[f"{T:g}"] = dict(mean_visit_frac=float(v.mean()),
                             std=float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        print(f"n={args.n} T={T:g}: visit_frac {v.mean():.3f} "
              f"+/- {out[f'{T:g}']['std']:.3f}", flush=True)
    # closest tier to 0.55
    best = min(out.items(), key=lambda kv: abs(kv[1]["mean_visit_frac"] - 0.55))
    print(f"n={args.n} PICK @vf~0.55 -> T={best[0]} (vf={best[1]['mean_visit_frac']:.3f})")
    with open(args.out, "w") as f:
        json.dump(dict(n=args.n, pkl=args.pkl, ni=args.ni, nrest=args.nrest,
                       seed0=args.seed0, grid=out, pick_055=best[0]), f, indent=2)
    print(f"CALIB_DONE n={args.n} {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
