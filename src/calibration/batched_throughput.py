"""Batched decoding throughput (the batched rows of Table 6) and the ILS cost per restart.

GPU rows: each scheme decodes the full 10,000-instance OP50 test set in fixed chunks after
one warmup chunk, and the wall time divided by 10,000 is the cost per instance. Rows
reported in Table 6: POMO plain greedy, SAS one-shot, SAS multistart 50 and SAS
multistart 50 x 16 augmentations (the ten- and twenty-five-start rows are measured too).
CPU rows: one and sixteen restarts of the local-search solver of Appendix E
(src/instruments/op_ils_reference.py) on the first 32 test instances, single process.
Table 6 does not use these CPU rows. Its ILS rows come from
src/calibration/latency_single_instance.py on one Intel Xeon Gold 6240R core.

    python src/calibration/batched_throughput.py

The paper's run (single RTX 4090 for the GPU rows, one EPYC 7352 core for the ILS rows)
ships as artifacts/timing/batched_throughput.json. Wall times are machine-dependent by
design; the chunk sizes are fixed so that reruns use the same batch shapes.
"""
import json
import os
import pickle
import platform
import sys
import time

import numpy as np
import torch

from rl4co.data.transforms import StateAugmentation
from rl4co.envs import OPEnv
from rl4co.models.zoo.pomo import POMO

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sas"))
sys.path.insert(0, os.path.join(HERE, "..", "instruments"))
from op_eval import load_dataset  # noqa: E402
from op_ils_reference import dist_matrix, op_solve_restart  # noqa: E402
from sas_pomo import SASPOMO  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
OP50 = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
CKPTS = (("pomo", os.path.join(CHECKPOINT_ROOT, "pomo_op50_seed1234", "last.ckpt")),
         ("sas", os.path.join(CHECKPOINT_ROOT, "sas_op50_seed42", "last.ckpt")))
CHUNKS = {"greedy": 2000, "greedy_head": 2000, "ms10": 512, "ms25": 256,
          "multistart": 128, "msaug": 12}

env = OPEnv(generator_params=dict(num_loc=50))
td_all, md5 = load_dataset(OP50, 50)
assert md5 == "47a828139a9fcf1e642a060f40837d9e", md5
B = td_all.batch_size[0]
out = dict(hardware=dict(gpu=torch.cuda.get_device_name(0), cpu=platform.processor()),
           dataset_md5=md5, timing={})


def build(arm, ckpt):
    m = POMO(env) if arm == "pomo" else SASPOMO(env, sas_arm="sas")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict(sd, strict=True)
    return m.policy.to("cuda").eval()


def timed(fn, chunk):
    with torch.inference_mode():  # warmup
        fn(env.reset(td_all[:chunk].to("cuda")))
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    vals = []
    with torch.inference_mode():
        for i in range(0, B, chunk):
            td = env.reset(td_all[i:i + chunk].to("cuda"))
            vals.append(fn(td).detach().float().cpu())
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    return wall / B * 1000.0, float(torch.cat(vals).mean())


def ms_best(policy, td, ns):
    o = policy(td.clone(), env, phase="test", decode_type="multistart_greedy",
               num_starts=ns)
    return o["reward"].reshape(ns, td.shape[0]).T.max(-1).values


AUG = StateAugmentation(num_augment=16, augment_fn="symmetric")


def msaug_best(policy, td):
    b0 = td.shape[0]
    o = policy(AUG(td.clone()), env, phase="test",
               decode_type="multistart_greedy", num_starts=50)
    return o["reward"].reshape(50 * 16, b0).T.max(-1).values


for arm, ckpt in CKPTS:
    pol = build(arm, ckpt)
    rows = {}
    if arm == "pomo":
        rows["greedy"] = lambda td, p=pol: p(td, env, phase="test",
                                             decode_type="greedy")["reward"]
    else:
        rows["greedy_head"] = lambda td, p=pol: p(td, env, phase="test",
                                                  decode_type="greedy", num_starts=0,
                                                  one_shot_start=True)["reward"]
    for ns, nm in ((10, "ms10"), (25, "ms25"), (50, "multistart")):
        rows[nm] = lambda td, p=pol, n=ns: ms_best(p, td, n)
    rows["msaug"] = lambda td, p=pol: msaug_best(p, td)
    for nm, fn in rows.items():
        ms, mp = timed(fn, CHUNKS[nm])
        out["timing"][f"{arm}.{nm}"] = dict(ms_per_instance=round(ms, 3),
                                            mean_best_prize=round(mp, 4))
        print(f"[{arm}] {nm}: {ms:.3f} ms/inst (prize {mp:.4f})", flush=True)

# ILS: one process, first 32 instances, the restart seeds of the paper's run.
data = pickle.load(open(OP50, "rb"))
t1 = t16 = 0.0
for i in range(32):
    depot, loc, prize, T = data[i]
    D = dist_matrix(np.asarray(depot, float), np.asarray(loc, float))
    pf = np.concatenate([[0.0], np.asarray(prize, float)])
    s = time.perf_counter()
    op_solve_restart(D, pf, float(T), 77000 + i)
    t1 += time.perf_counter() - s
    s = time.perf_counter()
    for r in range(16):
        op_solve_restart(D, pf, float(T), 78000 + i * 100 + r)
    t16 += time.perf_counter() - s
out["timing"]["ils.restart1"] = dict(s_per_instance=round(t1 / 32, 3))
out["timing"]["ils.restart16"] = dict(s_per_instance=round(t16 / 32, 3))
print(f"[ils] 1-restart {t1 / 32:.2f} s/inst, 16-restart {t16 / 32:.2f} s/inst", flush=True)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(os.path.join(OUTPUT_ROOT, "batched_throughput.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
