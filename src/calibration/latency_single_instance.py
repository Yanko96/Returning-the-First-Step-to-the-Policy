"""Single-instance latency on OP50 and on the OPLib n <= 300 subset, one machine.

Every row runs on the same node (one GPU and one core of its host CPU), so the
neural decode and classical construction are compared at matched hardware on
both test sets. Batch size 1, one instance per call, wall time per instance
(time.perf_counter, CUDA synchronized), input preparation (environment reset,
distance matrix) outside the timer as in constructor_baseline.py.

Rows
  sas_oneshot_gpu / sas_oneshot_cpu  SAS one-shot decode: one rollout from the
                                     head's argmax start, decoded greedily
  sas_ms50_gpu                       SAS multistart, num_starts = min(50, n)
  cons_c1                            constructor best-of-1 (deterministic
                                     prize-per-detour insertion)
  cons_c16                           constructor best-of-16: only the 16 noised
                                     constructions of the paper's c16 scheme
                                     (constructor_baseline.kbest also builds
                                     c1 and c5, so timing kbest()["c16"]
                                     timed 22 constructions)
  ils_r1 / ils_r16                   local-search solver, 1 and 16 restarts,
                                     OP50 only, first 32 instances, the
                                     restart seeds of batched_throughput.py

Test sets
  OP50   first 100 canonical test instances, T = 3
  OPLib  the 102 EUC_2D instances with n <= 300 (the subset of the paper's
         classical-construction comparison), normalized as in oplib_eval.py

Fast rows (neural, c1) are timed REPS times per instance and summarized by the
per-instance median; c16 and ILS are timed once. Per-instance times and decoded
prizes are saved for the neural and constructor rows, with each instance's n (the
ILS rows keep only their summary). The OPLib prizes can be compared with the shipped
seed-42 evaluation as a checkpoint and pipeline check.
"""
import argparse
import glob
import json
import os
import pickle
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from tensordict import TensorDict

from rl4co.envs import OPEnv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sas"))
sys.path.insert(0, os.path.join(HERE, "..", "instruments"))
from sas_pomo import SASPOMO  # noqa: E402
import op_ils_reference as g  # noqa: E402
from oplib_parser import parse_oplib, normalize  # noqa: E402

# Same library tolerance relaxation as oplib_eval.py: normalized OPLib budgets
# are non-terminating floats and fp32 bookkeeping can overshoot by ~1e-5.
_orig_check = OPEnv.check_solution_validity


def _relaxed_check(*a, **kw):
    try:
        return _orig_check(*a, **kw)
    except AssertionError as e:
        msg = str(e)
        if "Max length exceeded by" in msg and float(msg.rsplit(" ", 1)[-1]) <= 1e-4:
            return None
        raise


OPEnv.check_solution_validity = staticmethod(_relaxed_check)

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
OPLIB_ROOT = os.environ.get("OPLIB_ROOT", os.path.join(DATA_ROOT, "OPLib-master"))
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
SEED0 = 820000  # constructor_baseline.py


# ---------------------------------------------------------------- instances
def op50_instances(n_inst):
    with open(os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl"), "rb") as f:
        data = pickle.load(f)[:n_inst]
    out = []
    for i, (depot, loc, prize, T) in enumerate(data):
        out.append(dict(name=f"op50_{i}", idx=i, n=len(loc),
                        depot=np.asarray(depot, float), loc=np.asarray(loc, float),
                        prize=np.asarray(prize, float), B=float(T)))
    return out


def oplib_instances():
    items = []
    for gen in (1, 2, 3):
        for p in sorted(glob.glob(os.path.join(OPLIB_ROOT, "instances", f"gen{gen}", "*.oplib"))):
            try:
                raw = parse_oplib(p)
            except NotImplementedError:
                continue
            if raw.edge_weight_type == "EUC_2D" and raw.dimension - 1 <= 300:
                items.append((p, gen, raw.dimension - 1))
    items.sort(key=lambda t: t[2])  # same order as constructor_baseline.py
    out = []
    for idx, (p, gen, n) in enumerate(items):
        norm = normalize(parse_oplib(p))
        out.append(dict(name=os.path.basename(p).replace(".oplib", ""), idx=idx, n=n, gen=gen,
                        depot=np.asarray(norm["depot"], float), loc=np.asarray(norm["loc"], float),
                        prize=np.asarray(norm["prize"], float), B=float(norm["max_length"])))
    return out


# ---------------------------------------------------------------- neural rows
def make_td(inst):
    return TensorDict(dict(
        depot=torch.tensor(inst["depot"], dtype=torch.float32)[None],
        locs=torch.tensor(inst["loc"], dtype=torch.float32)[None],
        prize=torch.tensor(inst["prize"], dtype=torch.float32)[None],
        max_length=torch.tensor([inst["B"]], dtype=torch.float32),
    ), batch_size=[1])


def neural_kw(row, n):
    if row == "oneshot":
        return dict(decode_type="greedy", num_starts=0, one_shot_start=True)
    return dict(decode_type="multistart_greedy", num_starts=min(50, n))


def bench_neural(policy, insts, dev, row, reps, n_warm=10):
    envs = {}
    tds = [make_td(x) for x in insts]

    def env_for(n):
        if n not in envs:
            envs[n] = OPEnv(generator_params=dict(num_loc=n))
        return envs[n]

    times, prizes = [], []
    with torch.inference_mode():
        for j in range(min(n_warm, len(insts))):  # warmup
            env = env_for(insts[j]["n"])
            policy(env.reset(tds[j].clone().to(dev)), env, phase="test",
                   **neural_kw(row, insts[j]["n"]))
        if dev == "cuda":
            torch.cuda.synchronize()
        for j, inst in enumerate(insts):
            env = env_for(inst["n"])
            ts = []
            for _ in range(reps):
                td = env.reset(tds[j].clone().to(dev))
                if dev == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = policy(td, env, phase="test", **neural_kw(row, inst["n"]))
                if dev == "cuda":
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            times.append(ts)
            prizes.append(float(out["reward"].max()))
    return times, prizes


# ---------------------------------------------------------------- classical rows
def dmat(inst):
    xy = np.vstack([inst["depot"][None, :], inst["loc"]])
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))


def cons_c1(D, pf, B, idx):
    t = np.array([0], dtype=int)
    unv = np.ones(D.shape[0], bool)
    unv[0] = False
    t, _, _, _ = g.op_insert_loop(D, pf, t, 0.0, unv, B)
    return float(pf[t].sum())


def cons_c16(D, pf, B, idx):
    best = 0.0
    for k in range(16):
        rng = np.random.default_rng(SEED0 + idx * 20 + k)
        gp, gd = g.GAMMA_SET[rng.integers(len(g.GAMMA_SET))]
        t = np.array([0], dtype=int)
        unv = np.ones(D.shape[0], bool)
        unv[0] = False
        t, _, _, _ = g.op_insert_loop(D, pf, t, 0.0, unv, B, rng=rng, noise=0.15, gp=gp, gd=gd)
        best = max(best, float(pf[t].sum()))
    return best


def bench_classical(fn, insts, reps):
    times, prizes = [], []
    for inst in insts:
        D = dmat(inst)
        pf = np.concatenate([[0.0], inst["prize"]])
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            p = fn(D, pf, inst["B"], inst["idx"])
            ts.append(time.perf_counter() - t0)
        times.append(ts)
        prizes.append(p)
    return times, prizes


def bench_ils(insts):
    t1, t16 = [], []
    for i, inst in enumerate(insts):
        D = g.dist_matrix(inst["depot"], inst["loc"])
        pf = np.concatenate([[0.0], inst["prize"]])
        s = time.perf_counter()
        g.op_solve_restart(D, pf, inst["B"], 77000 + i)
        t1.append([time.perf_counter() - s])
        s = time.perf_counter()
        for r in range(16):
            g.op_solve_restart(D, pf, inst["B"], 78000 + i * 100 + r)
        t16.append([time.perf_counter() - s])
    return t1, t16


# ---------------------------------------------------------------- summary
def summarize(times, insts):
    med = np.array([float(np.median(t)) for t in times]) * 1000.0
    ns = np.array([x["n"] for x in insts[:len(med)]])
    s = dict(mean_ms=float(med.mean()), median_ms=float(np.median(med)),
             min_ms=float(med.min()), max_ms=float(med.max()), n_inst=int(len(med)))
    if ns.min() != ns.max():
        s["by_n"] = {f"{lo}-{hi}": dict(n_inst=int(((ns >= lo) & (ns <= hi)).sum()),
                                        mean_ms=float(med[(ns >= lo) & (ns <= hi)].mean()))
                     for lo, hi in ((1, 100), (101, 200), (201, 300))
                     if ((ns >= lo) & (ns <= hi)).any()}
    return s


def hardware():
    hw = dict(python=platform.python_version(), torch=torch.__version__,
              node=platform.node(), torch_threads=torch.get_num_threads())
    try:
        cpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        hw["cpu"] = next((ln.split(":", 1)[1].strip() for ln in cpu.splitlines()
                          if ln.startswith("Model name")), "unknown")
    except OSError:
        hw["cpu"] = platform.processor()
    if torch.cuda.is_available():
        hw["gpu"] = torch.cuda.get_device_name(0)
    return hw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(CHECKPOINT_ROOT, "sas_op50_seed42", "last.ckpt"))
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--op50-n", type=int, default=100)
    ap.add_argument("--smoke", action="store_true", help="few instances per set, for a dry run")
    ap.add_argument("--devices", default="cuda,cpu",
                    help="neural rows to time: cuda, cpu or both (classical rows run with cpu)")
    ap.add_argument("--out", default=os.path.join(OUTPUT_ROOT, "latency_single_instance.json"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    t00 = time.time()
    sets = {"op50": op50_instances(args.op50_n), "oplib_le300": oplib_instances()}
    if args.smoke:
        sets = {k: v[:12] for k, v in sets.items()}
    print({k: len(v) for k, v in sets.items()}, flush=True)

    env0 = OPEnv(generator_params=dict(num_loc=50, max_length=3.0))
    model = SASPOMO(env0, sas_arm="sas")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    print("loaded strict=True", args.ckpt, flush=True)

    out = dict(ckpt=args.ckpt, reps=args.reps, hardware=hardware(),
               protocol="batch 1; per-instance median over reps for neural and c1, one timing for "
                        "c16 and ILS; input preparation outside the timer; one CPU thread",
               sets={})
    devices = [d for d in args.devices.split(",") if d]
    for name, insts in sets.items():
        res, per = {}, dict(name=[x["name"] for x in insts], n=[x["n"] for x in insts])
        for dev in devices:
            policy = model.policy.to(dev).eval()
            rows = (("oneshot", "ms50") if dev == "cuda" else ("oneshot",))
            for row in rows:
                key = f"sas_{row}_{'gpu' if dev == 'cuda' else 'cpu'}"
                ts, pz = bench_neural(policy, insts, dev, row, args.reps)
                res[key] = summarize(ts, insts)
                per[key] = dict(times_s=ts, prize=pz)
                print(name, key, round(res[key]["mean_ms"], 2), "ms", flush=True)
        if "cpu" not in devices:
            out["sets"][name] = dict(summary=res, per_instance=per)
            continue
        for key, fn, reps in (("cons_c1", cons_c1, args.reps), ("cons_c16", cons_c16, 1)):
            ts, pz = bench_classical(fn, insts, reps)
            res[key] = summarize(ts, insts)
            per[key] = dict(times_s=ts, prize=pz)
            print(name, key, round(res[key]["mean_ms"], 2), "ms", flush=True)
        if name == "op50":
            t1, t16 = bench_ils(insts[:32] if not args.smoke else insts[:3])
            res["ils_r1"] = summarize(t1, insts)
            res["ils_r16"] = summarize(t16, insts)
            print(name, "ils", round(res["ils_r1"]["mean_ms"], 1), round(res["ils_r16"]["mean_ms"], 1),
                  "ms", flush=True)
        out["sets"][name] = dict(summary=res, per_instance=per)

    out["wall_s"] = round(time.time() - t00, 1)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("LATENCY_SINGLE_INSTANCE_DONE", out["wall_s"], "s", flush=True)


if __name__ == "__main__":
    main()
