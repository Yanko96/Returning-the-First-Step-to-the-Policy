"""Must-visit control evaluation — head uniformity + SAS-vs-POMO parity
on TSP50/CVRP50.
Head uniformity thresholds: entropy >= 0.90 ln(n_feas), KL to uniform <= 0.15,
top-1 <= 3/n. Parity thresholds: |gh - greedy| <= 0.5pp, multistart rows <= 0.2pp.
Costs and deltas only; NO wall-clock numbers anywhere.
Self-check: recompute tour cost from actions on a PRE-reset clone
(env.reset mutates its input)."""
import argparse
import json
import os
import sys

import numpy as np
import torch

from rl4co.envs import TSPEnv, CVRPEnv
from rl4co.models.zoo.pomo import POMO
from rl4co.data.transforms import StateAugmentation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sas"))
from sas_pomo import SASPOMO  # noqa: E402

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")

ap = argparse.ArgumentParser()
ap.add_argument("--problem", choices=["tsp", "cvrp"], required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

PROB = args.problem
Env = TSPEnv if PROB == "tsp" else CVRPEnv
MASK_FIRST = PROB != "tsp"
env = Env(generator_params=dict(num_loc=50))

def build(arm, ckpt):
    m = (POMO(env) if arm == "pomo"
         else SASPOMO(env, sas_arm="sas", start_mask_first=MASK_FIRST))
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict(sd, strict=True)
    return m.policy.to("cuda").eval()

# fixed evaluation data: 10k instances, eval seed 1234 via the env
# generator (deterministic given the seed; per-problem convention)
torch.manual_seed(1234)
td_all = env.generator(batch_size=[10_000]).cpu()
# head-stats instances: 1000 fresh, seed 5150
torch.manual_seed(5150)
td_head = env.generator(batch_size=[1_000]).cpu()


def self_check(policy):
    raw = td_all[:8].clone()
    td = env.reset(raw.clone().to("cuda"))
    with torch.inference_mode():
        out = policy(td.clone(), env, phase="test", decode_type="greedy",
                     return_actions=True)
    r = out["reward"].cpu().numpy()
    acts = out["actions"].cpu().numpy()
    locs = raw["locs"].numpy()
    for b in range(8):
        if PROB == "tsp":
            pts = locs[b][acts[b]]
            pts = np.concatenate([pts, pts[:1]])
        else:
            depot = raw["depot"][b].numpy()
            route = [a for a in acts[b]]
            xy = [depot] + [depot if a == 0 else locs[b][a - 1] for a in route] + [depot]
            pts = np.array(xy)
        cost = float(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(-1)).sum())
        assert abs(cost - (-r[b])) < 1e-3, (b, cost, -r[b])
    print("self-check OK (cost == -reward)")


def head_stats(policy):
    with torch.inference_mode():
        td = env.reset(td_head.clone().to("cuda"))
        h, _ = policy.encoder(td)
        mask = td["action_mask"].clone()
        if MASK_FIRST:
            mask[..., 0] = False
        logits = policy.start_head(h).masked_fill(~mask, float("-inf"))
        p = torch.softmax(logits, dim=-1)
        nf = mask.sum(-1).float()
        ent = -(p.clamp_min(1e-12).log() * p).sum(-1)
        kl = (nf.log() - ent)          # KL(p || uniform over feasible)
        top1 = p.max(-1).values
    return dict(mean_entropy=round(float(ent.mean()), 4),
                ln_n_feasible=round(float(nf.log().mean()), 4),
                entropy_ratio=round(float((ent / nf.log()).mean()), 4),
                mean_kl_to_uniform=round(float(kl.mean()), 4),
                mean_top1=round(float(top1.mean()), 5),
                uniform_top1=round(float((1.0 / nf).mean()), 5))


def decode_rows(policy, has_head):
    rows = {}
    aug = StateAugmentation(num_augment=16, augment_fn="symmetric")

    def run(fn, chunk):
        vals = []
        with torch.inference_mode():
            for i in range(0, 10_000, chunk):
                td = env.reset(td_all[i:i + chunk].clone().to("cuda"))
                vals.append(fn(td).detach().float().cpu())
        return float((-torch.cat(vals)).mean())  # mean cost

    rows["greedy"] = run(lambda td: policy(td, env, phase="test",
                                           decode_type="greedy")["reward"], 1000)
    if has_head:
        rows["greedy_head"] = run(
            lambda td: policy(td, env, phase="test", decode_type="greedy",
                              num_starts=0, one_shot_start=True)["reward"], 1000)

    def ms(td, ns):
        o = policy(td.clone(), env, phase="test",
                   decode_type="multistart_greedy", num_starts=ns)
        return o["reward"].reshape(ns, td.shape[0]).T.max(-1).values

    rows["ms10"] = run(lambda td: ms(td, 10), 256)
    rows["ms50"] = run(lambda td: ms(td, 50), 128)

    def msaug(td):
        B0 = td.shape[0]
        o = policy(aug(td.clone()), env, phase="test",
                   decode_type="multistart_greedy", num_starts=50)
        return o["reward"].reshape(50 * 16, B0).T.max(-1).values

    rows["msaug"] = run(msaug, 8)
    return rows


out = dict(problem=PROB, n=50, eval_note="costs only; no wall-clock",
           arms={})
pomo = build("pomo", os.path.join(CHECKPOINT_ROOT, f"pomo_{PROB}50", "last.ckpt"))
self_check(pomo)
out["arms"]["pomo"] = dict(rows=decode_rows(pomo, has_head=False))
for s in (42, 43):
    pol = build("sas", os.path.join(CHECKPOINT_ROOT,
                                    f"sas_{PROB}50_seed{s}", "last.ckpt"))
    out["arms"][f"sas{s}"] = dict(head=head_stats(pol),
                                  rows=decode_rows(pol, has_head=True))
    print(f"[{PROB}] sas{s} head:", out["arms"][f"sas{s}"]["head"], flush=True)

# parity deltas (pp of POMO cost)
base = out["arms"]["pomo"]["rows"]
for s in (42, 43):
    r = out["arms"][f"sas{s}"]["rows"]
    out["arms"][f"sas{s}"]["parity_pp"] = {
        "gh_vs_greedy": round((r["greedy_head"] - base["greedy"]) / base["greedy"] * 100, 3),
        **{k: round((r[k] - base[k]) / base[k] * 100, 3) for k in ("ms10", "ms50", "msaug")},
    }
    print(f"[{PROB}] sas{s} parity:", out["arms"][f"sas{s}"]["parity_pp"], flush=True)

with open(args.out, "w") as f:
    json.dump(out, f, indent=1)
print(f"CONTROL_EVAL_COMPLETE {PROB}", flush=True)
