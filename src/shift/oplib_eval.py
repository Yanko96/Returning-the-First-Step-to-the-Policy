"""OPLib zero-shot evaluation (settings as in the paper's appendix).

Subset rule: instances/gen{1,2,3}/*.oplib, EUC_2D only; every
excluded file recorded with its reason. Run all, report all.

Reporting metric (decoupled from model input normalization):
  - model input = oplib_parser.normalize() Kool tuple (choices a-d, fixed);
  - every rollout's tour is mapped back to ORIGINAL coordinates and
    re-validated under the TSPLIB integer metric (EUC_2D nint);
    integer length > COST_LIMIT -> rollout VOID (counted, reported);
  - score = sum of ORIGINAL integer node scores of visited nodes
    (depot carries no collectable score);
  - gap% = (ref - best_valid_score) / ref * 100, ref = vendored EA4OP
    per-instance ROUTE_SCORE (solutions/ea4op, Kobeaga 2018; heuristic
    best-known, not certified optima).

Schemes (flag-listed):
  greedy_head (sas/uniform) or plain greedy (pomo) at 1 rollout;
  ms10 / ms25 / ms50 with num_starts = min(k, n) (guard recorded).

Self-check: the continuous prize of the extracted tour must equal the
model's reported reward (catches any tour-extraction mismatch such as a
dropped forced first step).
"""
import argparse
import glob
import json
import os
import time

import numpy as np
import torch
from tensordict import TensorDict

import sys

from rl4co.envs import OPEnv
from rl4co.models.zoo.pomo import POMO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "instruments"))
sys.path.insert(0, os.path.join(HERE, "..", "sas"))
from oplib_parser import parse_oplib, normalize, _tsplib_distance  # noqa: E402

# ---- tool-layer tolerance relaxation (documented in the paper's appendix) ----
# OPEnv.check_solution_validity asserts length <= max_length + 1e-5.
# OPLib budgets are non-terminating floats after normalization; fp32
# bookkeeping drift can overshoot by ~1-2e-5 (never triggered on Kool
# data where max_length is exact). Relax the LIBRARY numeric tolerance
# to 1e-4 for this evaluation only. Direction of the change: it can only
# ADMIT borderline rollouts into reward computation; the actual
# feasibility gate of this study is the ORIGINAL-integer-metric
# re-validation below, which voids any true budget violator.
_orig_check = OPEnv.check_solution_validity
TOL_EVENTS = []           # batch-level tolerance events
TOL_CTX = {"cur": ""}     # set to "<instance>|<scheme>" before each decode


def _relaxed_check(*a, **kw):
    try:
        return _orig_check(*a, **kw)
    except AssertionError as e:
        msg = str(e)
        if "Max length exceeded by" in msg:
            overshoot = float(msg.rsplit(" ", 1)[-1])
            if overshoot <= 1e-4:
                TOL_EVENTS.append(dict(ctx=TOL_CTX["cur"], max_overshoot=overshoot))
                return None  # tolerated fp drift; the integer-metric check below decides
        raise


OPEnv.check_solution_validity = staticmethod(_relaxed_check)
# ----------------------------------------------------------------------

SCHEME_STARTS = {"ms10": 10, "ms25": 25, "ms50": 50}
BUCKETS = [(0, 100), (101, 300), (301, 1000), (1001, 10 ** 9)]  # presentation only;
# per-instance records are saved, so bucketing can be redone without rerun.


def load_ref(sol_path):
    kv = {}
    with open(sol_path) as f:
        for ln in f:
            if ":" in ln:
                k, v = ln.split(":", 1)
                kv[k.strip()] = v.strip()
            if ln.strip() == "NODE_SEQUENCE_SECTION":
                break
    return dict(ref_score=int(kv["ROUTE_SCORE"]), ref_cost=int(kv["ROUTE_COST"]),
                ref_nodes=int(kv["ROUTE_NODES"]))


def build_model(arm, env):
    if arm == "pomo":
        return POMO(env)
    from sas_pomo import SASPOMO

    return SASPOMO(env, sas_arm=("uniform" if arm == "uniform" else "sas"))


def tours_from_actions(actions):
    """actions [R, L] -> list of visited-node index lists (env indexing,
    0 = depot; prefix up to the first 0)."""
    tours = []
    for row in actions:
        row = row.tolist()
        t = []
        for a in row:
            if a == 0:
                break
            t.append(int(a))
        tours.append(t)
    return tours


def int_metric_check(raw, tour_env_idx):
    """tour in env indexing (1..n, depot excluded) -> (int_len, int_score).
    Assumes depot is file node 0 (asserted at parse time), so env index i
    maps to file node i."""
    seq = [0] + tour_env_idx + [0]
    pts = raw.coords[np.array(seq)]
    d = _tsplib_distance(pts[:-1], pts[1:], raw.edge_weight_type)
    int_len = int(d.sum())
    int_score = int(raw.scores[np.array(tour_env_idx, dtype=int)].sum()) if tour_env_idx else 0
    return int_len, int_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pomo", "sas", "uniform"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--oplib-root",
                    default=os.environ.get(
                        "OPLIB_ROOT",
                        os.path.join(os.environ.get("DATA_ROOT", "data"),
                                     "OPLib-master")))
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", type=int, default=0, help="evaluate only the K smallest instances")
    args = ap.parse_args()
    dev = args.device

    # ---- enumerate the fixed subset ----------------------------------
    included, excluded = [], []
    for gen in (1, 2, 3):
        for path in sorted(glob.glob(os.path.join(args.oplib_root, "instances", f"gen{gen}", "*.oplib"))):
            try:
                raw = parse_oplib(path)
            except NotImplementedError as e:
                excluded.append(dict(file=os.path.basename(path), gen=gen, reason=str(e).split("(")[0].strip()))
                continue
            if raw.edge_weight_type != "EUC_2D":  # CEIL_2D excluded by subset rule
                excluded.append(dict(file=os.path.basename(path), gen=gen,
                                     reason=f"{raw.edge_weight_type} excluded by subset rule"))
                continue
            assert raw.depot_index == 0, f"{raw.name}: depot not file node 1"
            sol = os.path.join(args.oplib_root, "solutions", "ea4op", f"gen{gen}",
                               os.path.basename(path).replace(".oplib", ".sol"))
            if not os.path.exists(sol):
                excluded.append(dict(file=os.path.basename(path), gen=gen, reason="no ea4op reference"))
                continue
            included.append(dict(path=path, gen=gen, dim=raw.dimension))
    included.sort(key=lambda r: (r["dim"], r["gen"]))
    if args.smoke:
        included = included[: args.smoke]
    print(f"included {len(included)} instances, excluded {len(excluded)}", flush=True)

    # ---- model --------------------------------------------------------
    env0 = OPEnv(generator_params=dict(num_loc=100))  # placeholder; per-instance envs below
    model = build_model(args.arm, env0)
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    print(f"loaded ckpt strict=True ({len(sd)} keys)", flush=True)
    policy = model.policy.to(dev).eval()

    one_shot = "greedy" if args.arm == "pomo" else "greedy_head"
    schemes = [one_shot] + list(SCHEME_STARTS)

    records = []
    t00 = time.time()
    for j, item in enumerate(included):
        raw = parse_oplib(item["path"])
        norm = normalize(raw)
        n = raw.dimension - 1
        td0 = TensorDict(dict(
            depot=torch.tensor(norm["depot"], dtype=torch.float32)[None],
            locs=torch.tensor(norm["loc"], dtype=torch.float32)[None],
            prize=torch.tensor(norm["prize"], dtype=torch.float32)[None],
            max_length=torch.tensor([norm["max_length"]], dtype=torch.float32),
        ), batch_size=[1])
        env = OPEnv(generator_params=dict(num_loc=n))
        ref = load_ref(os.path.join(args.oplib_root, "solutions", "ea4op",
                                    f"gen{item['gen']}",
                                    os.path.basename(item["path"]).replace(".oplib", ".sol")))
        rec = dict(name=os.path.basename(item["path"]).replace(".oplib", ""),
                   gen=item["gen"], n=n, cost_limit=raw.cost_limit,
                   norm_budget=round(norm["max_length"], 4), **ref, schemes={})
        loc_full = np.concatenate([norm["depot"][None], norm["loc"]]).astype(np.float64)

        def cont_len(t):
            seq = np.array([0] + t + [0])
            p = loc_full[seq]
            return float(np.sqrt(((p[1:] - p[:-1]) ** 2).sum(-1)).sum())

        t0 = time.time()
        for scheme in schemes:
            TOL_CTX["cur"] = f"{rec['name']}|{scheme}"
            with torch.inference_mode():
                td = env.reset(td0.clone()).to(dev)
                if scheme == "greedy_head":
                    out = policy(td, env, phase="test", decode_type="greedy",
                                 num_starts=0, one_shot_start=True, return_actions=True)
                elif scheme == "greedy":
                    out = policy(td, env, phase="test", decode_type="greedy",
                                 return_actions=True)
                else:
                    k = min(SCHEME_STARTS[scheme], n)
                    out = policy(td, env, phase="test", decode_type="multistart_greedy",
                                 num_starts=k, return_actions=True)
            rewards = out["reward"].detach().float().cpu().numpy().reshape(-1)
            tours = tours_from_actions(out["actions"].detach().cpu().numpy().reshape(len(rewards), -1))
            # self-check: extracted tour must reproduce the model's reward
            prize_np = np.concatenate([[0.0], norm["prize"]])
            for r, t in zip(rewards, tours):
                assert abs(prize_np[t].sum() - r) < 1e-3, \
                    (rec["name"], scheme, float(prize_np[t].sum()), float(r))
            # Tolerance-band evidence: per-rollout continuous overshoot band.
            # The library's strict tolerance is 1e-5; the relaxation admits (1e-5, 1e-4].
            # For each rollout in that band, record whether the ORIGINAL
            # integer gate voids it (fp64 recompute over the same fp32 coords;
            # per-rollout proxy for the env's fp32 bookkeeping).
            n_void = 0
            best = 0
            best_len = None
            tol_band = tol_band_voided = over_band = 0
            max_overshoot = -1e9
            for t in tours:
                int_len, int_score = int_metric_check(raw, t)
                voided = int_len > raw.cost_limit
                ov = cont_len(t) - norm["max_length"]
                max_overshoot = max(max_overshoot, ov)
                if 1e-5 < ov <= 1e-4:
                    tol_band += 1
                    tol_band_voided += voided
                elif ov > 1e-4:
                    over_band += 1  # must stay 0; anything here is a real bug
                if voided:
                    n_void += 1
                elif int_score > best:
                    best, best_len = int_score, int_len
            k_eff = len(rewards)
            gap = (ref["ref_score"] - best) / ref["ref_score"] * 100.0
            rec["schemes"][scheme] = dict(
                rollouts=k_eff, n_void=n_void, all_void=(n_void == k_eff),
                best_int_score=best, best_int_len=best_len,
                gap_pct=round(gap, 3),
                start_guard=(scheme in SCHEME_STARTS and SCHEME_STARTS[scheme] > n),
                tol_band=tol_band, tol_band_voided=tol_band_voided,
                over_band=over_band,
                max_cont_overshoot=round(max_overshoot, 8),
            )
        rec["wall_s"] = round(time.time() - t0, 1)
        records.append(rec)
        print(f"[{args.arm}] {j + 1}/{len(included)} {rec['name']} n={n} "
              + " ".join(f"{s}:{rec['schemes'][s]['gap_pct']}%" for s in schemes)
              + f" void={[rec['schemes'][s]['n_void'] for s in schemes]} ({rec['wall_s']}s)",
              flush=True)

    # ---- aggregates (per gen x scheme, per bucket x scheme) -----------
    def agg(rows, scheme):
        gaps = [r["schemes"][scheme]["gap_pct"] for r in rows]
        voids = sum(r["schemes"][scheme]["n_void"] for r in rows)
        total = sum(r["schemes"][scheme]["rollouts"] for r in rows)
        return dict(n_inst=len(rows), mean_gap=round(float(np.mean(gaps)), 3),
                    median_gap=round(float(np.median(gaps)), 3),
                    worst_gap=round(float(np.max(gaps)), 3),
                    void_rate=round(voids / max(total, 1), 5),
                    n_all_void=sum(r["schemes"][scheme]["all_void"] for r in rows))

    aggregates = {}
    for scheme in schemes:
        aggregates[scheme] = dict(
            overall=agg(records, scheme),
            by_gen={f"gen{g}": agg([r for r in records if r["gen"] == g], scheme)
                    for g in (1, 2, 3)},
            by_size={f"{lo}-{hi if hi < 10**9 else 'max'}":
                     agg([r for r in records if lo <= r["n"] <= hi], scheme)
                     for lo, hi in BUCKETS
                     if any(lo <= r["n"] <= hi for r in records)},
        )

    tol_summary = dict(
        batch_tolerance_events=len(TOL_EVENTS),
        events=TOL_EVENTS,
        rollouts_in_band=sum(r["schemes"][s]["tol_band"] for r in records for s in schemes),
        band_voided_by_integer_gate=sum(r["schemes"][s]["tol_band_voided"]
                                        for r in records for s in schemes),
        over_band_rollouts=sum(r["schemes"][s]["over_band"] for r in records for s in schemes),
    )
    print(f"tolerance band: band={tol_summary['rollouts_in_band']} "
          f"voided={tol_summary['band_voided_by_integer_gate']} "
          f"over_band={tol_summary['over_band_rollouts']} "
          f"batch_events={tol_summary['batch_tolerance_events']}", flush=True)

    out = dict(arm=args.arm, ckpt=args.ckpt, device=dev,
               subset_rule="gen1-3 EUC_2D", n_included=len(records),
               excluded=excluded, aggregates=aggregates, instances=records,
               tol_summary=tol_summary,
               total_wall_s=round(time.time() - t00, 1))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print("OPLIB_EVAL_DONE", args.out, flush=True)


if __name__ == "__main__":
    main()
