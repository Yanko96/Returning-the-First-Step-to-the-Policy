"""PCTSP evaluation harness (OP protocol translated to PCTSP50).

Identical structure to the OP harness (op_eval.py): five accounting
schemes + mechanism-evidence schemes, multistart family via DIRECT policy
calls with explicit env (the library evaluators do not pass the environment
explicitly), strict ckpt loading, per-instance
npz, head-probs snapshot for the SAS arm.

PCTSP specifics:
  - dataset: canonical Kool pctsp50_test_seed1234.pkl
    (md5 df23990603d21aa25ed41f87640db4fa), tuple =
    (depot, loc, penalty, deterministic_prize, stochastic_prize).
  - reward in rl4co = NEGATIVE cost (cost = tour length + unvisited
    penalties); best-of-N = max reward = min cost, so the OP reduction
    code carries over unchanged. Reporting converts to cost.
  - reference: OR-Tools 4.48 / 0.00% (RL4CO E.1). gap = (cost-4.48)/4.48.
  - self-check at load: decode one batch greedy, recompute the cost of the
    returned actions from raw coordinates/penalties and compare to
    -reward (guards both the reward-sign convention and the loader's
    field mapping).
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

from rl4co.envs import PCTSPEnv
from rl4co.models.zoo.pomo import POMO
from rl4co.tasks.eval import AugmentationEval, GreedyEval, SamplingEval

ORTOOLS_REF = 4.48  # RL4CO v6 E.1 PCTSP50 reference row (cost, minimization)

CHUNKS = {"greedy": 1000, "sampling1280": 8, "multistart": 64, "aug1280": 8, "msaug": 8,
          "ms10": 256, "ms25": 128, "greedy_head": 1000}


def load_dataset(path, n):
    with open(path, "rb") as f:
        data = pickle.load(f)
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    B = len(data)
    td = TensorDict(dict(
        depot=torch.tensor([d[0] for d in data], dtype=torch.float32),
        locs=torch.tensor([d[1] for d in data], dtype=torch.float32),
        penalty=torch.tensor([d[2] for d in data], dtype=torch.float32),
        deterministic_prize=torch.tensor([d[3] for d in data], dtype=torch.float32),
        stochastic_prize=torch.tensor([d[4] for d in data], dtype=torch.float32),
    ), batch_size=[B])
    assert td["locs"].shape == (B, n, 2)
    return td, md5


def build_model(arm, env):
    if arm == "pomo":
        return POMO(env)
    from sas_pomo import SASPOMO

    return SASPOMO(env, sas_arm=("uniform" if arm == "uniform" else "sas"))


def self_check(policy, env, td_all, dev):
    """Greedy-decode 8 instances; recompute cost from actions; compare.

    NOTE: env.reset mutates its input td in place (locs gain the depot
    row, penalties get padded) -- recompute MUST read from a pre-reset
    clone or every node index is off by one."""
    raw = td_all[:8].clone()
    locs = raw["locs"].numpy()
    depot = raw["depot"].numpy()
    pen = raw["penalty"].numpy()
    td = env.reset(raw.clone().to(dev))
    with torch.inference_mode():
        out = policy(td.clone(), env, phase="test", decode_type="greedy",
                     return_actions=True)
    r = out["reward"].cpu().numpy()
    acts = out["actions"].cpu().numpy()
    for b in range(8):
        route = [a for a in acts[b] if a != 0]
        pts = np.concatenate([depot[b][None], locs[b][np.array(route, int) - 1],
                              depot[b][None]]) if route else np.stack([depot[b], depot[b]])
        length = float(np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(-1)).sum())
        unv = np.ones(locs.shape[1], bool)
        if route:
            unv[np.array(route, int) - 1] = False
        cost = length + float(pen[b][unv].sum())
        assert abs(cost - (-r[b])) < 1e-3, (b, cost, -r[b])
    print("self-check OK: reward == -(length + unvisited penalties)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["pomo", "sas", "uniform"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    dev = "cpu" if args.smoke else "cuda"
    env = PCTSPEnv(generator_params=dict(num_loc=args.n))
    model = build_model(args.arm, env)
    if args.ckpt != "NONE":
        sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(sd, strict=True)
        print(f"loaded ckpt strict=True ({len(sd)} keys)")
    policy = model.policy.to(dev).eval()

    td_all, md5 = load_dataset(args.dataset, args.n)
    if args.smoke:
        td_all = td_all[:24]
    self_check(policy, env, td_all, dev)

    n_aug_ms, n_samples, n_aug_big = (2, 8, 8) if args.smoke else (16, 1280, 1280)

    from rl4co.data.transforms import StateAugmentation

    def _multistart_best(td, ns):
        out = policy(td.clone(), env, phase="test",
                     decode_type="multistart_greedy", num_starts=ns)
        return out["reward"].reshape(ns, td.shape[0]).T.max(-1).values

    def _msaug_best(td, ns, naug, _aug=StateAugmentation(num_augment=n_aug_ms,
                                                         augment_fn="symmetric")):
        B0 = td.shape[0]
        td_a = _aug(td.clone())
        out = policy(td_a, env, phase="test",
                     decode_type="multistart_greedy", num_starts=ns)
        return out["reward"].reshape(ns * naug, B0).T.max(-1).values

    evals = {
        "greedy": GreedyEval(env, progress=False),
        "sampling1280": SamplingEval(env, samples=n_samples, softmax_temp=1.0, progress=False),
        "multistart": lambda td: _multistart_best(td, args.n),
        "aug1280": AugmentationEval(env, num_augment=n_aug_big, progress=False),
        "msaug": lambda td: _msaug_best(td, args.n, n_aug_ms),
        "ms10": lambda td: _multistart_best(td, 10),
        "ms25": lambda td: _multistart_best(td, 25),
    }
    run_greedy_head = args.arm in ("sas", "uniform")

    results = dict(arm=args.arm, ckpt=args.ckpt, dataset=args.dataset, dataset_md5=md5,
                   n=args.n, device=dev, chunks=CHUNKS, reference=ORTOOLS_REF,
                   accounting=dict(msaug=f"{args.n}x{n_aug_ms}", sampling=n_samples,
                                   aug=n_aug_big),
                   schemes={})
    per_instance = {}
    B = td_all.batch_size[0]
    for name, ev in evals.items():
        chunk = max(1, CHUNKS[name] // 50) if args.smoke else CHUNKS[name]
        t0 = time.time()
        bests = []
        with torch.inference_mode():
            for i in range(0, B, chunk):
                sub = td_all[i:i + chunk].to(dev)
                td = env.reset(sub)
                if callable(ev) and not hasattr(ev, "_inner"):
                    rewards = ev(td)
                else:
                    _, rewards = ev._inner(policy, td)
                bests.append(rewards.detach().float().cpu())
        bests = torch.cat(bests)
        assert bests.shape[0] == B, (name, bests.shape)
        cost = float((-bests).mean())
        gap = (cost - ORTOOLS_REF) / ORTOOLS_REF * 100
        results["schemes"][name] = dict(mean_best_cost=round(cost, 4),
                                        gap_pct=round(gap, 3),
                                        wall_s=round(time.time() - t0, 1))
        per_instance[name] = (-bests).numpy()
        print(f"[{args.arm}] {name}: cost={cost:.4f} gap={gap:.3f}% "
              f"({time.time() - t0:.0f}s)", flush=True)

    if run_greedy_head:
        t0 = time.time()
        bests = []
        with torch.inference_mode():
            for i in range(0, B, CHUNKS["greedy_head"]):
                sub = td_all[i:i + CHUNKS["greedy_head"]].to(dev)
                td = env.reset(sub)
                out = policy(td, env, phase="test", decode_type="greedy",
                             num_starts=0, one_shot_start=True)
                bests.append(out["reward"].detach().float().cpu())
        bests = torch.cat(bests)
        assert bests.shape[0] == B
        cost = float((-bests).mean())
        gap = (cost - ORTOOLS_REF) / ORTOOLS_REF * 100
        results["schemes"]["greedy_head"] = dict(mean_best_cost=round(cost, 4),
                                                 gap_pct=round(gap, 3),
                                                 wall_s=round(time.time() - t0, 1))
        per_instance["greedy_head"] = (-bests).numpy()
        print(f"[{args.arm}] greedy_head: cost={cost:.4f} gap={gap:.3f}%", flush=True)

    if args.arm == "sas":
        with torch.inference_mode():
            sub = td_all[:200].to(dev)
            td = env.reset(sub)
            h, _ = policy.encoder(td)
            mask = td["action_mask"].clone()
            mask[..., 0] = False
            logits = policy.start_head(h).masked_fill(~mask, float("-inf"))
            per_instance["head_probs_first200"] = torch.softmax(
                logits, dim=-1).cpu().numpy()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(args.out.replace(".json", "_perinst.npz"), **per_instance)
    print("EVAL_DONE", args.out)


if __name__ == "__main__":
    main()
