"""Start-rule and budget-matched decoding baselines on OP50, on every Table 1 checkpoint.

Same test set, Compass reference, fixed chunk sizes and decode paths as op_eval.py,
which produced Table 1. Schemes, with rollouts per instance in brackets:

  head   SAS checkpoints only, their own learned head; must reproduce Table 1:
         head1 [1], head10 [10], head25 [25]
  rule   prize / distance-to-depot scores in place of the start head:
         rule1 [1], rule10 [10], rule25 [25], rule50 [50]
         rule50 takes every start, so it must equal Table 1's multistart-50 value
  samp   POMO checkpoints only. Each rollout starts at a uniformly random feasible
         node (depot excluded, distinct starts within an instance) and samples every
         later step from the policy at temperature 1; best rollout kept:
         samp1, samp10, samp25, samp50 (torch seed recorded per scheme)
  libsamp  POMO checkpoints only, the library's SamplingEval, kept for continuity
         with the earlier single-checkpoint rows: libsamp10, libsamp25
  beam   POMO checkpoints only, beam search keeping the best beam:
         beam10, beam25, beam50

Why samp is not the library's SamplingEval: with samples=1 the library's decoding
strategy discards the 1 (num_starts=1 turns multistart off and num_samples is
unset), falls back to num_starts = n = 50, and returns the best of 50 rollouts
whose first step is sampled from the never-trained step-zero distribution. For
samples >= 2 it draws first nodes with sample_n_random_actions, which can draw
the depot. samp uses one definition at every budget.

A POMO checkpoint has no start head, so for the rule and samp schemes it is loaded
into the SAS policy class with only that one weight missing (asserted); the rule
supplies the scores the head would, and samp uses the class's uniform start draw.
Beam search and libsamp run through the library POMO model loaded strictly, as the
earlier single-checkpoint baseline did.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from rl4co.envs import OPEnv
from rl4co.models.zoo.pomo import POMO
from rl4co.tasks.eval import SamplingEval

from op_eval import COMPASS_REF, load_dataset
from sas_pomo import SASPOMO

CHUNK = dict(one=2000, n10=512, n25=256, n50=128, samp=512, beam=256)
SAMPLE_SEED = 20260708
HEAD_KEY = "policy.start_head.proj.weight"


class RuleHead(torch.nn.Module):
    """Stands in for the learned start head: returns the precomputed rule scores."""

    def __init__(self):
        super().__init__()
        self.cur = None

    def forward(self, h):
        return self.cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=["pomo", "sas"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default=os.path.join(os.environ.get("DATA_ROOT", "data"), "op",
                                                      "op_dist50_test_seed1234.pkl"))
    ap.add_argument("--schemes", default="", help="comma list; default all for the backbone")
    ap.add_argument("--limit", type=int, default=0, help="first N instances (smoke only)")
    ap.add_argument("--device", default="cuda", help="cpu for a mechanics check only")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = 50
    dev = args.device
    env = OPEnv(generator_params=dict(num_loc=n))
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]

    sas_model = SASPOMO(env, sas_arm="sas")
    missing, unexpected = sas_model.load_state_dict(sd, strict=False)
    allowed = {HEAD_KEY} if args.backbone == "pomo" else set()
    assert set(missing) == allowed and not unexpected, (missing, unexpected)
    sas_policy = sas_model.policy.to(dev).eval()
    sas_policy.sas_arm = "sas"
    learned_head = sas_policy.start_head
    rule_head = RuleHead()

    free_policy = None
    if args.backbone == "pomo":
        pomo = POMO(env)
        pomo.load_state_dict(sd, strict=True)
        free_policy = pomo.policy.to(dev).eval()

    td_all, md5 = load_dataset(args.dataset, n)
    if args.limit:
        td_all = td_all[: args.limit]
    B = td_all.batch_size[0]
    d0 = torch.linalg.norm(td_all["locs"] - td_all["depot"][:, None, :], dim=-1)
    rule = torch.full((B, n + 1), -1e9)
    rule[:, 1:] = td_all["prize"] / (d0 + 1e-9)

    def oneshot(td, a, b):
        return sas_policy(td, env, phase="test", decode_type="greedy",
                          num_starts=0, one_shot_start=True)["reward"]

    def multistart(k):
        def f(td, a, b):
            out = sas_policy(td.clone(), env, phase="test",
                             decode_type="multistart_greedy", num_starts=k)
            return out["reward"].reshape(k, td.shape[0]).T.max(-1).values
        return f

    def sampling(k):
        # uniform random feasible start (SAS class, uniform draw), sampled continuation
        if k == 1:  # num_starts=1 would silently turn multistart off; inject instead
            def f(td, a, b):
                return sas_policy(td, env, phase="test", decode_type="sampling",
                                  num_starts=0, one_shot_start=True)["reward"]
            return f

        def f(td, a, b):
            out = sas_policy(td.clone(), env, phase="test",
                             decode_type="multistart_sampling", num_starts=k)
            return out["reward"].reshape(k, td.shape[0]).T.max(-1).values
        return f

    def lib_sampling(k):
        assert k >= 2, "library SamplingEval with samples=1 runs 50 rollouts"
        ev = SamplingEval(env, samples=k, softmax_temp=1.0, progress=False)

        def f(td, a, b):
            return ev._inner(free_policy, td)[1]
        return f

    def beam(k):
        def f(td, a, b):
            return free_policy(td, env, phase="test", decode_type="beam_search",
                               beam_width=k, select_best=True)["reward"]
        return f

    # name -> (start scorer, chunk, rollouts, fn)
    plan = {}
    if args.backbone == "sas":
        plan["head1"] = ("head", CHUNK["one"], 1, oneshot)
        plan["head10"] = ("head", CHUNK["n10"], 10, multistart(10))
        plan["head25"] = ("head", CHUNK["n25"], 25, multistart(25))
    plan["rule1"] = ("rule", CHUNK["one"], 1, oneshot)
    plan["rule10"] = ("rule", CHUNK["n10"], 10, multistart(10))
    plan["rule25"] = ("rule", CHUNK["n25"], 25, multistart(25))
    plan["rule50"] = ("rule", CHUNK["n50"], 50, multistart(50))
    if args.backbone == "pomo":
        for k in (1, 10, 25, 50):
            plan[f"samp{k}"] = ("uniform", CHUNK["samp"], k, sampling(k))
        for k in (10, 25):
            plan[f"libsamp{k}"] = ("free", CHUNK["samp"], k, lib_sampling(k))
        for k in (10, 25, 50):
            plan[f"beam{k}"] = ("free", CHUNK["beam"], k, beam(k))
    if args.schemes:
        keep = set(args.schemes.split(","))
        plan = {k: v for k, v in plan.items() if k in keep}

    ref = COMPASS_REF[n]
    results = dict(backbone=args.backbone, ckpt=args.ckpt, dataset=args.dataset,
                   dataset_md5=md5, n=n, n_inst=B, device=(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"),
                   reference=ref, sample_seed_base=SAMPLE_SEED, schemes={})
    per_inst = {}
    for name, (scorer, chunk, rollouts, fn) in plan.items():
        sas_policy.sas_arm = "uniform" if scorer == "uniform" else "sas"
        if scorer == "rule":
            sas_policy.start_head = rule_head
        elif scorer == "head":
            sas_policy.start_head = learned_head
        if scorer in ("free", "uniform"):
            torch.manual_seed(SAMPLE_SEED + rollouts)
        t0 = time.time()
        vals = []
        with torch.inference_mode():
            for i in range(0, B, chunk):
                sub = td_all[i:i + chunk].to(dev)
                td = env.reset(sub)
                if scorer == "rule":
                    rule_head.cur = rule[i:i + sub.batch_size[0]].to(dev)
                vals.append(fn(td, i, i + sub.batch_size[0]).detach().float().cpu())
        r = torch.cat(vals)
        assert r.shape[0] == B, (name, r.shape)
        mp = float(r.mean())
        results["schemes"][name] = dict(rollouts=rollouts, chunk=chunk,
                                        mean_best_prize=round(mp, 4),
                                        gap_pct=round((ref - mp) / ref * 100, 3),
                                        wall_s=round(time.time() - t0, 1))
        per_inst[name] = r.numpy()
        print(f"[{args.backbone}] {name}: gap={(ref - mp) / ref * 100:.3f}% "
              f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=2)
    np.savez_compressed(args.out.replace(".json", "_perinst.npz"), **per_inst)
    print("done", args.out, flush=True)


if __name__ == "__main__":
    main()
