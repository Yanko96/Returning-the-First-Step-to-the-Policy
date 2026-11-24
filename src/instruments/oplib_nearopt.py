"""Near-optimal start sets on OPLib instances with n<=300. Reference ILS
solver verbatim (op_ils_reference), 16 restarts, seeds 70000+idx*100+r,
run on the NORMALIZED continuous instance (the same geometry the head sees).
Parallel at (instance, restart) level."""
import glob
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import op_ils_reference as ils_ref
from oplib_parser import parse_oplib, normalize

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
ROOT = os.environ.get("OPLIB_ROOT", os.path.join(DATA_ROOT, "OPLib-master"))
N_RESTARTS = 16
NEAR_OPT = 0.995
SEED0 = 70000

items = []
for gen in (1, 2, 3):
    for p in sorted(glob.glob(os.path.join(ROOT, "instances", f"gen{gen}", "*.oplib"))):
        try:
            raw = parse_oplib(p)
        except NotImplementedError:
            continue
        if raw.edge_weight_type == "EUC_2D" and raw.dimension - 1 <= 300:
            items.append((p, gen, raw.dimension - 1))
items.sort(key=lambda t: t[2])
print(f"{len(items)} instances (n<=300)", flush=True)


def job(args):
    idx, r = args
    p, gen, n = items[idx]
    raw = parse_oplib(p)
    norm = normalize(raw)
    D = ils_ref.dist_matrix(norm["depot"], norm["loc"])
    prize_full = np.concatenate([[0.0], norm["prize"]])
    bp, bt, bL = ils_ref.op_solve_restart(D, prize_full, float(norm["max_length"]),
                                          SEED0 + idx * 100 + r)
    pair = [int(bt[1]), int(bt[-1])] if len(bt) > 1 else []
    return idx, r, bp, pair


def main():
    t0 = time.time()
    jobs = [(i, r) for i in range(len(items)) for r in range(N_RESTARTS)]
    acc = {i: [] for i in range(len(items))}
    with Pool(processes=96) as pool:
        for k, (i, r, bp, pair) in enumerate(pool.imap_unordered(job, jobs, chunksize=4)):
            acc[i].append((bp, pair))
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{len(jobs)} jobs, {time.time() - t0:.0f}s", flush=True)
    out = []
    for i, (p, gen, n) in enumerate(items):
        rp = acc[i]
        best = max(b for b, _ in rp)
        sset = sorted({s for b, pair in rp if b >= NEAR_OPT * best for s in pair})
        out.append(dict(name=os.path.basename(p).replace(".oplib", ""), gen=gen, n=n,
                        best_prize=round(best, 6),
                        restart_prizes=[round(b, 6) for b, _ in rp],
                        restart_starts=[pair for _, pair in rp],
                        nearopt_start_set=sset))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "oplib_nearopt.json"), "w") as f:
        json.dump(dict(n_restarts=N_RESTARTS, nearopt_frac=NEAR_OPT, seed0=SEED0,
                       metric="normalized continuous (head geometry)",
                       instances=out), f)
    sizes = [len(r["nearopt_start_set"]) for r in out]
    print(f"start-set sizes: mean {np.mean(sizes):.2f} min {min(sizes)} max {max(sizes)}")
    print(f"NEAROPT_ANALYSIS_COMPLETE {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
