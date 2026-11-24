"""Start-stakes on OP100 and OPLib with the OP50
instrument, unchanged: lite_solve/greedy_fill/two_opt imported verbatim from
budget_sweep_stakes.py, parameters pinned (8 forced starts, 2
restarts each, 8 free restarts, 40 lite rounds), denominator = ILS-reference
16-restart best per instance (the paper's stakes definition; OP100/OPLib nearopt files),
clip [0,1]. Only the DRIVER is new: instance source, feasible-set size n,
and fresh seed bases (documented below). No OP50 quantity is recomputed.

  OP100  first 200 canonical Kool op_dist100 test instances, T = 4.0
         (the OP100 evaluation budget), seeds 600000 + idx*100.
  OPLib  the 102 gen1-3 EUC_2D instances with n <= 300, NORMALIZED
         geometry (same tuple the head sees; OPLib nearopt convention),
         seeds 700000 + idx*100.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "instruments"))

from budget_sweep_stakes import lite_solve, N_FORCED, R_FORCED, R_FREE  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
OP100_PKL = os.path.join(DATA_ROOT, "op", "op_dist100_test_seed1234.pkl")
OP100_NEAROPT = os.path.join(ARTIFACT_ROOT, "nearopt", "op100_nearopt.json")
OPLIB_ROOT = os.environ.get("OPLIB_ROOT",
                            os.path.join(DATA_ROOT, "OPLib-master"))
OPLIB_NEAROPT = os.path.join(ARTIFACT_ROOT, "nearopt", "oplib_nearopt.json")
T_OP100 = 4.0
N_INST_OP100 = 200

_G = {}


def _init_op100():
    with open(OP100_PKL, "rb") as f:
        _G["data"] = pickle.load(f)
    _G["ils_ref"] = {r["idx"]: r["best_prize"]
                     for r in json.load(open(OP100_NEAROPT))["instances"]}


def _op100_job(idx):
    depot, loc, prize_raw, _ = _G["data"][idx]
    xy = np.vstack([np.asarray(depot, float)[None, :], np.asarray(loc, float)])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    prize = np.concatenate([[0.0], np.asarray(prize_raw, float)])
    n = D.shape[0] - 1
    seed0 = 600000 + idx * 100
    rng = np.random.default_rng(seed0)
    feas = [s for s in range(1, n + 1) if 2 * D[0, s] <= T_OP100 + 1e-12]
    forced = rng.choice(feas, size=min(N_FORCED, len(feas)), replace=False)
    best_free = max(lite_solve(D, prize, T_OP100, seed0 + 9000 + r)
                    for r in range(R_FREE))
    fvals = [max(lite_solve(D, prize, T_OP100, seed0 + k * 10 + r, forced=int(s))
                 for r in range(R_FORCED)) for k, s in enumerate(forced)]
    denom = _G["ils_ref"][idx]
    stakes = float(np.clip(1.0 - float(np.mean(fvals)) / max(denom, 1e-9), 0.0, 1.0))
    return idx, stakes, best_free, float(np.mean(fvals)), denom


def _init_oplib():
    from oplib_parser import parse_oplib, normalize
    items = []
    for gen in (1, 2, 3):
        for p in sorted(glob.glob(os.path.join(
                OPLIB_ROOT, "instances", f"gen{gen}", "*.oplib"))):
            try:
                raw = parse_oplib(p)
            except NotImplementedError:
                continue
            if raw.edge_weight_type == "EUC_2D" and raw.dimension - 1 <= 300:
                items.append((p, gen, raw.dimension - 1))
    items.sort(key=lambda t: t[2])
    _G["items"] = items
    _G["parse"] = parse_oplib
    _G["norm"] = normalize
    _G["ils_ref"] = {r["name"]: r["best_prize"]
                     for r in json.load(open(OPLIB_NEAROPT))["instances"]}


def _oplib_job(idx):
    p, gen, n = _G["items"][idx]
    name = os.path.basename(p).replace(".oplib", "")
    raw = _G["parse"](p)
    norm = _G["norm"](raw)
    xy = np.vstack([np.asarray(norm["depot"], float)[None, :],
                    np.asarray(norm["loc"], float)])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    prize = np.concatenate([[0.0], np.asarray(norm["prize"], float)])
    B = float(norm["max_length"])
    seed0 = 700000 + idx * 100
    rng = np.random.default_rng(seed0)
    feas = [s for s in range(1, n + 1) if 2 * D[0, s] <= B + 1e-12]
    forced = rng.choice(feas, size=min(N_FORCED, len(feas)), replace=False)
    best_free = max(lite_solve(D, prize, B, seed0 + 9000 + r)
                    for r in range(R_FREE))
    fvals = [max(lite_solve(D, prize, B, seed0 + k * 10 + r, forced=int(s))
                 for r in range(R_FORCED)) for k, s in enumerate(forced)]
    denom = _G["ils_ref"][name]
    stakes = float(np.clip(1.0 - float(np.mean(fvals)) / max(denom, 1e-9), 0.0, 1.0))
    return idx, stakes, best_free, float(np.mean(fvals)), denom, name, gen, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, choices=["op100", "oplib"])
    ap.add_argument("--procs", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="smoke: first K only")
    args = ap.parse_args()
    t0 = time.time()

    if args.set == "op100":
        init, jobf = _init_op100, _op100_job
        idxs = list(range(N_INST_OP100))
    else:
        init, jobf = _init_oplib, _oplib_job
        _init_oplib()
        idxs = list(range(len(_G["items"])))
    if args.limit:
        idxs = idxs[: args.limit]

    recs = {}
    with Pool(processes=args.procs, initializer=init) as pool:
        for k, res in enumerate(pool.imap_unordered(jobf, idxs, chunksize=1)):
            recs[res[0]] = res
            if (k + 1) % 10 == 0 or k == 0:
                print(f"{k + 1}/{len(idxs)}, {time.time() - t0:.0f}s", flush=True)

    if args.set == "op100":
        rows = [dict(idx=i, stakes=recs[i][1], lite_free=recs[i][2],
                     mean_forced=recs[i][3], ref_best=recs[i][4]) for i in idxs]
    else:
        rows = [dict(idx=i, name=recs[i][5], gen=recs[i][6], n=recs[i][7],
                     stakes=recs[i][1], lite_free=recs[i][2],
                     mean_forced=recs[i][3], ref_best=recs[i][4]) for i in idxs]
    sv = np.array([r["stakes"] for r in rows])
    lref = np.array([r["lite_free"] / max(r["ref_best"], 1e-9) for r in rows])
    out = dict(set=args.set, instrument="budget_sweep_stakes lite_solve verbatim",
               denominator="16-restart reference best (the paper's stakes definition)",
               n_forced=N_FORCED, r_forced=R_FORCED, r_free=R_FREE,
               seed0_base=600000 if args.set == "op100" else 700000,
               budget=(T_OP100 if args.set == "op100" else "per-instance normalized"),
               n_inst=len(rows),
               stakes_mean=float(sv.mean()), stakes_std=float(sv.std(ddof=1)),
               stakes_se=float(sv.std(ddof=1) / np.sqrt(len(sv))),
               lite_free_over_ref_mean=float(lref.mean()),
               clipped=int(((1.0 - np.array([r["mean_forced"] for r in rows])
                             / np.maximum([r["ref_best"] for r in rows], 1e-9)) < 0).sum()),
               per_instance=rows)
    suffix = f"_first{args.limit}" if args.limit else ""
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    path = os.path.join(OUTPUT_ROOT, f"stakes_{args.set}{suffix}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{args.set}: stakes {sv.mean():.4f} +/- {sv.std(ddof=1):.4f} "
          f"(se {out['stakes_se']:.4f}), lite/ref {lref.mean():.4f}, "
          f"clipped {out['clipped']}")
    print(f"OOD_STAKES_DONE {path} {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
