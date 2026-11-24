"""POMO+-ported agent hit@1 on the first 1000 canonical OP50 instances,
local CPU forwards from retrained checkpoints. Same procedure
as head_hit1_first1000.py (SAS): agent scores -> depot masked -> argmax,
compared against nearopt_starts_op50_first1000.json; Wilson 95% CIs.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from rl4co.envs import OPEnv  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from pomoplus_pomo import PPlusPOMO  # noqa: E402

N_FIRST, CHUNK = 1000, 200
DATA_ROOT = os.environ.get("DATA_ROOT", "data")
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
CKPT = os.path.join(CHECKPOINT_ROOT, "pomoplus_op50_seed{sd}", "last.ckpt")
NEAR = os.path.join(ARTIFACT_ROOT, "nearopt", "nearopt_starts_op50_first1000.json")


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - hw, ctr + hw


with open(PKL, "rb") as f:
    data = pickle.load(f)[:N_FIRST]
td_all = TensorDict(dict(
    depot=torch.tensor([d[0] for d in data], dtype=torch.float32),
    locs=torch.tensor([d[1] for d in data], dtype=torch.float32),
    prize=torch.tensor([d[2] for d in data], dtype=torch.float32),
    max_length=torch.tensor([d[3] for d in data], dtype=torch.float32),
), batch_size=[N_FIRST])

near = json.load(open(NEAR))
S = [set(r["nearopt_start_set"]) for r in near["instances"]]
env = OPEnv(generator_params=dict(num_loc=50))

out = dict(n_instances=N_FIRST,
           chance_hit1=float(np.mean([len(s) for s in S]) / 50.0), per_seed={})
for sd in ("42", "43", "44"):
    model = PPlusPOMO(env)
    sdict = torch.load(CKPT.format(sd=sd), map_location="cpu",
                       weights_only=False)["state_dict"]
    model.load_state_dict(sdict, strict=True)
    policy = model.policy.eval()
    probs_all = []
    with torch.inference_mode():
        for i in range(0, N_FIRST, CHUNK):
            td = env.reset(td_all[i:i + CHUNK].clone())
            h, _ = policy.encoder(td)
            mask = td["action_mask"].clone()
            mask[..., 0] = False
            logits = policy.start_agent(h).masked_fill(~mask, float("-inf"))
            probs_all.append(torch.softmax(logits, dim=-1).numpy())
    hp = np.concatenate(probs_all)
    arg = hp.argmax(1)
    hits = np.array([int(arg[i]) in S[i] for i in range(N_FIRST)])
    top1 = hp.max(1)
    lo, hi = wilson(int(hits.sum()), N_FIRST)
    out["per_seed"][sd] = dict(hit_at_1=float(hits.mean()),
                               wilson95=[float(lo), float(hi)],
                               mean_top1_prob=float(top1.mean()))
    print(f"seed {sd}: hit@1 {hits.mean():.4f} (CI {lo:.4f}-{hi:.4f}), "
          f"mean top-1 prob {top1.mean():.3f}", flush=True)
h = [out["per_seed"][s]["hit_at_1"] for s in out["per_seed"]]
out["hit_at_1_mean"], out["hit_at_1_std"] = float(np.mean(h)), float(np.std(h, ddof=1))
os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(os.path.join(OUTPUT_ROOT, "pplus_hit1_op50.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"mean {np.mean(h):.4f} +/- {np.std(h, ddof=1):.4f}, "
      f"chance {out['chance_hit1']:.4f}")
print("PPLUS_HIT1_DONE")
