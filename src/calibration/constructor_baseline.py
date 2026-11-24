"""Classical prize-per-detour constructor baseline.

The construction stage of the reference ILS (op_ils_reference op_insert_loop),
run WITHOUT any descent/local search, as a deployment baseline:

  best-of-1    deterministic greedy insertion, score = prize/detour
               (gamma (1,1), no noise)
  best-of-5    the solver's deterministic 5-shape criterion portfolio
  best-of-16   16 noised constructions (shape sampled, noise 0.15),
               seeds 820000 + idx*20 + k

OP50   T = 3.0, first 1000 canonical test instances; gap vs the same
       per-instance ILS reference used for the paper's one-shot table.
OPLib  n <= 300 EUC_2D subset (102 instances, the near-opt-computable
       subset of the paper's OPLib table); construction on the NORMALIZED
       instance (what the neural arms see), tours re-validated under the
       ORIGINAL integer metric and scored against the EA4OP reference,
       mirroring the model evaluation. Voided (over-budget) constructions
       counted; per-instance best over valid ones.
Latency  single-process wall time per instance for best-of-1 and
       best-of-16 on the first 50 OP50 instances (CPU, this machine;
       reported as measured, no batching).
"""
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

import op_ils_reference as g  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
OP_PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
OP_NEAROPT = os.path.join(ARTIFACT_ROOT, "nearopt",
                          "nearopt_starts_op50_first1000.json")
OPLIB_ROOT = os.environ.get("OPLIB_ROOT",
                            os.path.join(DATA_ROOT, "OPLib-master"))
T_OP50 = 3.0
N_OP50 = 1000
SEED0 = 820000

_G = {}


def construct(D, prize_full, B, rng=None, noise=0.0, gp=1.0, gd=1.0):
    m = D.shape[0]
    t = np.array([0], dtype=int)
    unv = np.ones(m, bool)
    unv[0] = False
    t, L, unv, _ = g.op_insert_loop(D, prize_full, t, 0.0, unv, B,
                                    rng=rng, noise=noise, gp=gp, gd=gd)
    return t, L


def kbest(D, prize_full, B, idx):
    """Returns dict of tours for best-of-1 / 5 / 16 (constructor only)."""
    tours = {}
    t1, _ = construct(D, prize_full, B)
    tours["c1"] = [t1]
    tours["c5"] = [construct(D, prize_full, B, gp=gp, gd=gd)[0]
                   for gp, gd in g.GAMMA_SET]
    rs = []
    for k in range(16):
        rng = np.random.default_rng(SEED0 + idx * 20 + k)
        gp, gd = g.GAMMA_SET[rng.integers(len(g.GAMMA_SET))]
        rs.append(construct(D, prize_full, B, rng=rng, noise=0.15,
                            gp=gp, gd=gd)[0])
    tours["c16"] = rs
    return tours


def _init_op50():
    with open(OP_PKL, "rb") as f:
        _G["data"] = pickle.load(f)[:N_OP50]
    _G["ref"] = {r["idx"]: r["best_prize"]
                 for r in json.load(open(OP_NEAROPT))["instances"]}


def _op50_job(idx):
    depot, loc, prize_raw, _ = _G["data"][idx]
    xy = np.vstack([np.asarray(depot, float)[None, :], np.asarray(loc, float)])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    prize = np.concatenate([[0.0], np.asarray(prize_raw, float)])
    tours = kbest(D, prize, T_OP50, idx)
    ref = _G["ref"][idx]
    return idx, {k: max(float(prize[t].sum()) for t in v)
                 for k, v in tours.items()}, ref


def _init_oplib():
    from oplib_parser import parse_oplib, normalize, _tsplib_distance
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
    _G["tsplib_d"] = _tsplib_distance


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
    tours = kbest(D, prize, B, idx)
    sol = os.path.join(OPLIB_ROOT, "solutions", "ea4op", f"gen{gen}",
                       os.path.basename(p).replace(".oplib", ".sol"))
    kv = {}
    with open(sol) as f:
        for ln in f:
            if ":" in ln:
                k, v = ln.split(":", 1)
                kv[k.strip()] = v.strip()
            if ln.strip() == "NODE_SEQUENCE_SECTION":
                break
    ref = int(kv["ROUTE_SCORE"])
    res = {}
    for kname, ts in tours.items():
        best, nvoid = 0, 0
        for t in ts:
            seq = np.array([0] + [int(x) for x in t[1:]] + [0])
            pts = raw.coords[seq]
            int_len = int(_G["tsplib_d"](pts[:-1], pts[1:],
                                         raw.edge_weight_type).sum())
            int_score = int(raw.scores[np.array([int(x) for x in t[1:]],
                                                dtype=int)].sum()) if len(t) > 1 else 0
            if int_len > raw.cost_limit:
                nvoid += 1
            else:
                best = max(best, int_score)
        res[kname] = dict(best=best, n_void=nvoid,
                          gap=(ref - best) / ref * 100.0)
    return idx, name, gen, n, ref, res


def main():
    t0 = time.time()
    out = dict(seed0=SEED0, gamma_set=[list(x) for x in g.GAMMA_SET])

    # --- OP50 ---
    _init_op50()
    recs = {}
    with Pool(processes=8, initializer=_init_op50) as pool:
        for idx, best, ref in pool.imap_unordered(_op50_job, range(N_OP50),
                                                  chunksize=8):
            recs[idx] = (best, ref)
    refm = float(np.mean([recs[i][1] for i in range(N_OP50)]))
    op50 = {}
    for k in ("c1", "c5", "c16"):
        pm = float(np.mean([recs[i][0][k] for i in range(N_OP50)]))
        op50[k] = dict(mean_prize=pm, gap_vs_ils_pct=(refm - pm) / refm * 100.0)
    out["op50"] = dict(T=T_OP50, n_inst=N_OP50, ils_ref_mean=refm, schemes=op50)
    print("OP50:", {k: round(v["gap_vs_ils_pct"], 2) for k, v in op50.items()},
          flush=True)

    # --- OP50 latency, single process ---
    lat = {}
    for k, fn in (("c1", lambda D, pr, i: construct(D, pr, T_OP50)),
                  ("c16", lambda D, pr, i: kbest(D, pr, T_OP50, i)["c16"])):
        ts = []
        for i in range(50):
            depot, loc, prize_raw, _ = _G["data"][i]
            xy = np.vstack([np.asarray(depot, float)[None, :],
                            np.asarray(loc, float)])
            D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
            prize = np.concatenate([[0.0], np.asarray(prize_raw, float)])
            t1 = time.perf_counter()
            fn(D, prize, i)
            ts.append(time.perf_counter() - t1)
        lat[k] = dict(mean_ms=float(np.mean(ts) * 1000),
                      median_ms=float(np.median(ts) * 1000))
    out["op50_latency_single_cpu"] = lat
    print("latency:", {k: round(v["mean_ms"], 2) for k, v in lat.items()},
          flush=True)

    # --- OPLib n<=300 ---
    _init_oplib()
    n_items = len(_G["items"])
    rows = []
    with Pool(processes=8, initializer=_init_oplib) as pool:
        for idx, name, gen, n, ref, res in pool.imap_unordered(
                _oplib_job, range(n_items), chunksize=1):
            rows.append(dict(idx=idx, name=name, gen=gen, n=n, ref=ref, **{
                k: res[k] for k in ("c1", "c5", "c16")}))
    rows.sort(key=lambda r: r["idx"])
    oplib = {}
    for k in ("c1", "c5", "c16"):
        gaps = [r[k]["gap"] for r in rows]
        oplib[k] = dict(mean_gap=float(np.mean(gaps)),
                        median_gap=float(np.median(gaps)),
                        n_all_void=sum(1 for r in rows
                                       if r[k]["best"] == 0 and r[k]["n_void"] > 0))
    out["oplib_le300"] = dict(n_inst=n_items, schemes=oplib, per_instance=rows)
    print("OPLib<=300:", {k: round(v["mean_gap"], 2) for k, v in oplib.items()},
          flush=True)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    json.dump(out, open(os.path.join(OUTPUT_ROOT, "constructor_baseline.json"), "w"), indent=1)
    print(f"CONSTRUCTOR_ANALYSIS_COMPLETE {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
