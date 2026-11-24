"""Generated from the OP evaluation harness (src/sas/op_eval.py): identical
scheme machinery; build_model extended with the pplus arm (PPlusPOMO)."""
"""OP evaluation harness — five decode schemes on the canonical Kool test set.

Decoding-scheme accounting (pinned from rl4co v0.6.0 tasks/eval.py):
  greedy        : plain greedy, 1 rollout (ALL arms incl. SAS — see note)
  sampling1280  : library SamplingEval(samples=1280)  [library-verbatim, both arms]
  multistart    : library GreedyMultiStartEval(num_starts=n) — POMO arm gets the
                  library arange-forced starts; SAS/uniform arms get their own
                  select_start_nodes_fn via SASPolicy.forward (SAS-INSERT)
  aug1280       : library AugmentationEval(num_augment=1280, symmetric)
  msaug         : library GreedyMultiStartAugmentEval(num_starts=n, num_augment=16,
                  symmetric) == E.1 column "Greedy Multistart with Augmentation (N x 16)"

Implementation note: the greedy row
is PLAIN greedy for all arms — num_starts=1 multistart silently degrades in the
library (DecodingStrategy.__init__ reassignment), so "head-argmax one-shot" cannot
be run at exactly-1-rollout budget through the strategy; plain greedy is
budget-identical to every baseline row and strictly fair.

Gap reference (OP50): Compass mean prize 16.17 on this exact dataset
(RL4CO v6 Table E.1 Classical Solvers row; dataset = op_dist50_test_seed1234.pkl).
"""
import argparse
import hashlib
import json
import os
import pickle
import time

import numpy as np
import torch
from tensordict import TensorDict

from rl4co.envs import OPEnv
from rl4co.models.zoo.pomo import POMO
from rl4co.tasks.eval import (
    AugmentationEval,
    GreedyEval,
    GreedyMultiStartAugmentEval,
    GreedyMultiStartEval,
    SamplingEval,
)

import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sas"))

# OP50: RL4CO v6 Table E.1. OP100: Kool 2019 table (dist),
# with OP50 16.17 cross-checked in the same table.
COMPASS_REF = {50: 16.17, 100: 33.19}

# fixed chunk sizes (recorded in the output JSON)
CHUNKS = {"greedy": 2000, "sampling1280": 8, "multistart": 128, "aug1280": 8, "msaug": 12,
          "ms10": 512, "ms25": 256, "greedy_head": 2000}
# ms10/ms25/greedy_head = mechanism-evidence schemes.
# ms10/ms25: the head has real selection room (N < n). greedy_head:
# true 1-rollout head-argmax first step, same budget as plain greedy.


def load_dataset(path, n):
    with open(path, "rb") as f:
        data = pickle.load(f)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    B = len(data)
    depot = torch.tensor([d[0] for d in data], dtype=torch.float32)
    locs = torch.tensor([d[1] for d in data], dtype=torch.float32)
    prize = torch.tensor([d[2] for d in data], dtype=torch.float32)
    max_length = torch.tensor([d[3] for d in data], dtype=torch.float32)
    assert locs.shape == (B, n, 2) and prize.shape == (B, n)
    td = TensorDict(
        dict(depot=depot, locs=locs, prize=prize, max_length=max_length),
        batch_size=[B],
    )
    return td, md5


def build_model(arm, env):
    if arm == "pomo":
        return POMO(env)
    if arm == "pplus":
        from pomoplus_pomo import PPlusPOMO

        return PPlusPOMO(env)
    from sas_pomo import SASPOMO

    return SASPOMO(env, sas_arm=("uniform" if arm == "uniform" else "sas"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pomo", "sas", "uniform", "pplus"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="CPU mechanics check, tiny budgets")
    ap.add_argument("--chunk-div", type=int, default=1,
                    help="divide all chunk sizes (smaller-VRAM GPUs); recorded in output")
    ap.add_argument("--schemes", default="",
                    help="comma list; run only these schemes (greedy_head listed too)")
    args = ap.parse_args()
    if args.chunk_div > 1:
        for k in list(CHUNKS):
            CHUNKS[k] = max(1, CHUNKS[k] // args.chunk_div)

    dev = "cpu" if args.smoke else "cuda"
    env = OPEnv(generator_params=dict(num_loc=args.n))
    model = build_model(args.arm, env)
    if args.ckpt != "NONE":
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(sd, strict=True)  # strict: no silent residue
        print(f"loaded ckpt strict=True ({len(sd)} keys)")
    policy = model.policy.to(dev).eval()

    td_all, md5 = load_dataset(args.dataset, args.n)
    if args.smoke:
        td_all = td_all[:24]

    n_aug_ms, n_samples, n_aug_big = (2, 8, 8) if args.smoke else (16, 1280, 1280)

    # Multistart-family schemes are implemented DIRECTLY because the library
    # evaluators {GreedyMultiStartEval,GreedyMultiStartAugmentEval}._inner
    # call the policy WITHOUT an explicit env: ConstructivePolicy.forward then
    # falls back to get_env(env_name) with the DEFAULT generator (num_loc=20),
    # and select_start_nodes wraps starts modulo 20 (arange %
    # env.generator.num_loc) — at N=25/50 only 20 distinct starts would be
    # forced (ms25==ms50 bit-identical, POMO understated) while mask-based SAS
    # arms would be unaffected: a silently unfair comparison. Direct policy
    # calls WITH env + start-major regrouping avoid this (layout validated by
    # wiring assertions + allclose probes).
    from rl4co.data.transforms import StateAugmentation

    def _multistart_best(td, ns):
        out = policy(td.clone(), env, phase="test",
                     decode_type="multistart_greedy", num_starts=ns)
        return out["reward"].reshape(ns, td.shape[0]).T.max(-1).values

    def _msaug_best(td, ns, naug, _aug=StateAugmentation(num_augment=n_aug_ms,
                                                         augment_fn="symmetric")):
        B0 = td.shape[0]
        td_a = _aug(td.clone())  # copy-major (a b): flat naug*B0
        out = policy(td_a, env, phase="test",
                     decode_type="multistart_greedy", num_starts=ns)
        # flat layout (s (a b)): index s*(naug*B0) + a*B0 + b -> group by b
        return out["reward"].reshape(ns * naug, B0).T.max(-1).values

    evals = {
        "greedy": GreedyEval(env, progress=False),
        "sampling1280": SamplingEval(env, samples=n_samples, softmax_temp=1.0, progress=False),
        "multistart": lambda td: _multistart_best(td, args.n),
        "aug1280": AugmentationEval(env, num_augment=n_aug_big, progress=False),
        "msaug": lambda td: _msaug_best(td, args.n, n_aug_ms),
        # mechanism-evidence schemes (all three arms for attribution)
        "ms10": lambda td: _multistart_best(td, 10),
        "ms25": lambda td: _multistart_best(td, 25),
    }
    if args.n != 50:
        # the OP50-consistent scheme list includes ms50; at n!=50 report BOTH
        # ms50 (literal) and the N=n "multistart" row above (structural).
        # No selection between them.
        evals["ms50"] = lambda td: _multistart_best(td, 50)
        CHUNKS.setdefault("ms50", CHUNKS["ms25"])
    if args.schemes:
        keep = set(args.schemes.split(","))
        evals = {k: v for k, v in evals.items() if k in keep}
    # greedy_head: SAS/uniform arms only (POMO arm's counterpart IS plain greedy)
    run_greedy_head = args.arm in ("sas", "uniform", "pplus")
    if args.schemes and "greedy_head" not in args.schemes.split(","):
        run_greedy_head = False

    results = dict(
        arm=args.arm, ckpt=args.ckpt, dataset=args.dataset, dataset_md5=md5,
        n=args.n, device=dev, chunks=CHUNKS,
        accounting=dict(msaug=f"{args.n}x{n_aug_ms}", sampling=n_samples, aug=n_aug_big),
        schemes={},
    )
    per_instance = {}
    B = td_all.batch_size[0]
    for name, ev in evals.items():
        chunk = max(1, CHUNKS[name] // (50 if args.smoke else 1)) if args.smoke else CHUNKS[name]
        t0 = time.time()
        bests = []
        with torch.inference_mode():
            for i in range(0, B, chunk):
                sub = td_all[i : i + chunk].to(dev)
                td = env.reset(sub)
                if callable(ev) and not hasattr(ev, "_inner"):
                    rewards = ev(td)  # direct multistart-family path
                else:
                    _, rewards = ev._inner(policy, td)
                bests.append(rewards.detach().float().cpu())
        bests = torch.cat(bests)
        assert bests.shape[0] == B, (name, bests.shape)
        wall = time.time() - t0
        mean_prize = float(bests.mean())
        ref = COMPASS_REF.get(args.n)
        gap = (ref - mean_prize) / ref * 100 if ref else None
        results["schemes"][name] = dict(
            mean_best_prize=round(mean_prize, 4),
            gap_pct=(round(gap, 3) if gap is not None else None),
            wall_s=round(wall, 1),
        )
        per_instance[name] = bests.numpy()
        print(f"[{args.arm}] {name}: mean={mean_prize:.4f} gap={gap if gap is None else round(gap,3)}% "
              f"({wall:.0f}s)", flush=True)

    if run_greedy_head:
        t0 = time.time()
        bests = []
        with torch.inference_mode():
            for i in range(0, B, CHUNKS["greedy_head"]):
                sub = td_all[i : i + CHUNKS["greedy_head"]].to(dev)
                td = env.reset(sub)
                out = policy(td, env, phase="test", decode_type="greedy",
                             num_starts=0, one_shot_start=True)
                bests.append(out["reward"].detach().float().cpu())
        bests = torch.cat(bests)
        assert bests.shape[0] == B
        wall = time.time() - t0
        mean_prize = float(bests.mean())
        ref = COMPASS_REF.get(args.n)
        gap = (ref - mean_prize) / ref * 100 if ref else None
        results["schemes"]["greedy_head"] = dict(
            mean_best_prize=round(mean_prize, 4),
            gap_pct=(round(gap, 3) if gap is not None else None),
            wall_s=round(wall, 1),
            note="1-rollout head-argmax first step",
        )
        per_instance["greedy_head"] = bests.numpy()
        print(f"[{args.arm}] greedy_head: mean={mean_prize:.4f} "
              f"gap={gap if gap is None else round(gap,3)}% ({wall:.0f}s)", flush=True)

    # head-distribution snapshot for mechanism visualization (SAS arm)
    if args.arm == "sas":
        with torch.inference_mode():
            sub = td_all[:200].to(dev)
            td = env.reset(sub)
            h, _ = policy.encoder(td)
            mask = td["action_mask"].clone()
            mask[..., 0] = False
            logits = policy.start_head(h).masked_fill(~mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        per_instance["head_probs_first200"] = probs

    if args.arm == "pomo" and args.n == 50 and not args.smoke:
        g = results["schemes"]["greedy"]["gap_pct"]
        results["baseline_check"] = "PASS" if 10.0 <= g <= 20.0 else "FAIL"
        print(f"baseline check (greedy gap in [10,20]): {g}% -> {results['baseline_check']}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(args.out.replace(".json", "_perinst.npz"), **per_instance)
    print("EVAL_DONE", args.out)


if __name__ == "__main__":
    main()
