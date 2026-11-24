"""Budget-sweep calibration — per-(tier, seed) outcome measurement on ONE model.

All quantities from the same SAS checkpoint, on the first --n-eval canonical
test instances with the tier budget substituted (base pkl md5-pinned; the
override value is recorded in the output):

  sas_oneshot     greedy_head: learned argmax start, greedy decode (1 rollout)
  uniform_oneshot mean over --uniform-draws random feasible starts, greedy
                  decode (1 rollout each; fixed draw seeds)
  oracle          best one-shot over ALL feasible starts (multistart n=50)
  head_probs      agent softmax snapshot (depot masked) for hit@1 / top-1

Realized benefit and gaps are computed downstream against the tier's ILS
reference (budget_sweep_nearopt output); this script only measures and stores raw
mean prizes plus per-instance arrays.
"""
import argparse
import hashlib
import json
import os
import pickle
import sys

import numpy as np
import torch
from tensordict import TensorDict

from rl4co.envs import OPEnv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sas"))
from sas_pomo import SASPOMO  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--budget", type=float, required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--n-eval", type=int, default=1000)
ap.add_argument("--uniform-draws", type=int, default=8)
ap.add_argument("--chunk", type=int, default=200)
ap.add_argument("--ms-chunk", type=int, default=50)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"

with open(args.dataset, "rb") as f:
    data = pickle.load(f)[: args.n_eval]
md5 = hashlib.md5(open(args.dataset, "rb").read()).hexdigest()
B = len(data)
td_all = TensorDict(dict(
    depot=torch.tensor([d[0] for d in data], dtype=torch.float32),
    locs=torch.tensor([d[1] for d in data], dtype=torch.float32),
    prize=torch.tensor([d[2] for d in data], dtype=torch.float32),
    max_length=torch.full((B,), args.budget, dtype=torch.float32),
), batch_size=[B])

env = OPEnv(generator_params=dict(num_loc=50, max_length=args.budget))
model = SASPOMO(env, sas_arm="sas")
sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
model.load_state_dict(sd, strict=True)
policy = model.policy.to(dev).eval()
print(f"loaded {args.ckpt} strict=True")

out = dict(ckpt=args.ckpt, budget=args.budget, dataset=args.dataset,
           dataset_md5=md5, n_eval=B, uniform_draws=args.uniform_draws,
           device=dev, chunks=dict(chunk=args.chunk, ms=args.ms_chunk))
per = {}


def run_oneshot(arm, draw_seed=None):
    policy.sas_arm = arm
    if draw_seed is not None:
        torch.manual_seed(draw_seed)
    rr = []
    with torch.inference_mode():
        for i in range(0, B, args.chunk):
            td = env.reset(td_all[i:i + args.chunk].clone().to(dev))
            o = policy(td, env, phase="test", decode_type="greedy",
                       num_starts=0, one_shot_start=True)
            rr.append(o["reward"].detach().float().cpu())
    return torch.cat(rr).numpy()


# 1) learned one-shot
per["sas_oneshot"] = run_oneshot("sas")
out["sas_oneshot_mean"] = float(per["sas_oneshot"].mean())
print(f"sas_oneshot mean {out['sas_oneshot_mean']:.4f}", flush=True)

# 2) uniform one-shot, fixed draws
draws = []
for k in range(args.uniform_draws):
    draws.append(run_oneshot("uniform", draw_seed=123400 + k))
per["uniform_oneshot_draws"] = np.stack(draws)  # [K, B]
out["uniform_oneshot_mean"] = float(per["uniform_oneshot_draws"].mean())
print(f"uniform_oneshot mean {out['uniform_oneshot_mean']:.4f}", flush=True)
policy.sas_arm = "sas"

# 3) oracle: best one-shot over all starts (deterministic top-50 = all)
rr = []
with torch.inference_mode():
    for i in range(0, B, args.ms_chunk):
        td = env.reset(td_all[i:i + args.ms_chunk].clone().to(dev))
        o = policy(td, env, phase="test", decode_type="multistart_greedy",
                   num_starts=50)
        rr.append(o["reward"].reshape(50, -1).T.max(-1).values.detach().float().cpu())
per["oracle_beststart"] = torch.cat(rr).numpy()
out["oracle_mean"] = float(per["oracle_beststart"].mean())
print(f"oracle mean {out['oracle_mean']:.4f}", flush=True)

# 4) head snapshot
hp = []
with torch.inference_mode():
    for i in range(0, B, args.chunk):
        td = env.reset(td_all[i:i + args.chunk].clone().to(dev))
        h, _ = policy.encoder(td)
        mask = td["action_mask"].clone()
        mask[..., 0] = False
        logits = policy.start_head(h.float()).masked_fill(~mask, float("-inf"))
        hp.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
per["head_probs"] = np.concatenate(hp)
out["head_top1_prob_mean"] = float(per["head_probs"].max(1).mean())

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(out, f, indent=2)
np.savez_compressed(args.out.replace(".json", "_perinst.npz"), **per)
print("BUDGET_SWEEP_EVAL_DONE", args.out)
