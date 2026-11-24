"""The constructor of constructor_baseline.py on all 10,000 OP50 test instances,
scored against the Compass reference (16.17) that Table 1 uses, so the constructor
and the neural one-shot decode share instances and reference.

    python src/calibration/constructor_op50_full.py [--procs 16] [--n-inst 10000]

Same construction, criterion portfolio and noise seeds as the OPLib and first-1,000
runs of constructor_baseline.py (best-of-16: seeds 820000 + idx*20 + k).
Outputs: outputs/constructor_op50_full.json and outputs/constructor_op50_full_perinst.npz
(the shipped copies are in artifacts/). CPU only; about 15 s at 16 processes.
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
sys.path.insert(0, HERE)
import constructor_baseline as C  # noqa: E402  (module import only; its main() is not run)

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
COMPASS = 16.17
_D = {}


def _init(n_inst):
    with open(C.OP_PKL, "rb") as f:
        _D["data"] = pickle.load(f)[:n_inst]


def _job(idx):
    depot, loc, prize_raw, _ = _D["data"][idx]
    xy = np.vstack([np.asarray(depot, float)[None, :], np.asarray(loc, float)])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    prize = np.concatenate([[0.0], np.asarray(prize_raw, float)])
    tours = C.kbest(D, prize, C.T_OP50, idx)
    return idx, {k: max(float(prize[t].sum()) for t in v) for k, v in tours.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--n-inst", type=int, default=10000)
    a = ap.parse_args()
    t0 = time.time()
    res = {}
    with Pool(processes=a.procs, initializer=_init, initargs=(a.n_inst,)) as pool:
        for idx, best in pool.imap_unordered(_job, range(a.n_inst), chunksize=16):
            res[idx] = best
    names = {"c1": "best_of_1", "c5": "best_of_5", "c16": "best_of_16"}
    out = dict(n_inst=a.n_inst, procs=a.procs, wall_s=round(time.time() - t0, 1),
               reference=COMPASS, dataset=os.path.basename(C.OP_PKL))
    arrays = {}
    for k in sorted(res[0]):
        arr = np.array([res[i][k] for i in range(a.n_inst)])
        arrays[names[k]] = arr
        out[names[k]] = dict(mean_prize=float(arr.mean()),
                             mean_prize_first1000=float(arr[:1000].mean()),
                             gap_vs_compass=float((COMPASS - arr.mean()) / COMPASS * 100))
        print(f"{names[k]:10s} mean prize {arr.mean():.4f}  gap vs Compass "
              f"{out[names[k]]['gap_vs_compass']:.2f}%")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "constructor_op50_full.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(OUTPUT_ROOT, "constructor_op50_full_perinst.npz"), **arrays)
    print("written", os.path.join(OUTPUT_ROOT, "constructor_op50_full.json"))


if __name__ == "__main__":
    main()
