"""Fit the 128-weight start head to a frozen POMO checkpoint.

The backbone comes from an unmodified POMO checkpoint and is frozen; only the start
head (one linear map, 128 weights) is trained, with the credit term SAS adds to the
POMO loss: -(R_s - b) log pi_head(s) over all N = n starts, b the shared mean
baseline, rollouts decoded by multistart sampling as in POMO training. The recipe's
optimizer settings are kept (Adam, lr 1e-4, weight decay 1e-6, batch 64, 2,500 steps
per epoch). The head after the last epoch is kept; the validation curve, on
instances disjoint from the test set, is only recorded.

Two things differ from SAS training, both by construction of a frozen fit: the
backbone does not move, and the start temperature is 1 from the first step (SAS
uses 5 and 3 for its first two epochs; at T = 3 every start is taken, so the
temperature rescales the credit gradient without changing its direction).

The fitted model is saved as a SAS checkpoint, so op_eval.py evaluates it exactly
as it evaluates the Table 1 checkpoints.
"""
import argparse
import json
import os
import time

import torch

from rl4co.envs import OPEnv

from sas_pomo import SASPOMO

HEAD_KEY = "policy.start_head.proj.weight"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="POMO checkpoint (frozen backbone)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=2500, help="steps per epoch")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--val-size", type=int, default=2000)
    ap.add_argument("--device", default="cuda", help="cpu for a mechanics check only")
    ap.add_argument("--out-ckpt", required=True)
    args = ap.parse_args()

    n, dev = 50, args.device
    env = OPEnv(generator_params=dict(num_loc=n))
    model = SASPOMO(env, sas_arm="sas")
    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert set(missing) == {HEAD_KEY} and not unexpected, (missing, unexpected)

    policy = model.policy.to(dev).eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    torch.manual_seed(args.seed)
    policy.start_head.proj.reset_parameters()
    w = policy.start_head.proj.weight
    w.requires_grad_(True)
    frozen_before = {k: v.detach().clone() for k, v in policy.state_dict().items()
                     if "start_head" not in k}
    opt = torch.optim.Adam([w], lr=1e-4, weight_decay=1e-6)

    # validation instances: own seed, disjoint from the test file
    torch.manual_seed(999_000 + args.seed)
    val = env.generator(batch_size=[args.val_size])

    def val_oneshot():
        with torch.inference_mode():
            td = env.reset(val.clone().to(dev))
            r = policy(td, env, phase="test", decode_type="greedy",
                       num_starts=0, one_shot_start=True)["reward"]
        return float(r.float().mean())

    torch.manual_seed(args.seed)
    log = dict(ckpt=args.ckpt, seed=args.seed, epochs=args.epochs, steps=args.steps,
               batch=args.batch, val_size=args.val_size, curve=[])
    v0 = val_oneshot()
    log["curve"].append(dict(epoch=0, val_oneshot_prize=v0))
    print(f"epoch 0 (init) val one-shot prize {v0:.4f}", flush=True)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        losses = []
        for it in range(args.steps):
            td = env.reset(env.generator(batch_size=[args.batch]).to(dev))
            out = policy(td, env, phase="train", decode_type="multistart_sampling",
                         num_starts=n)
            R = out["reward"].reshape(n, args.batch).T            # [B, n], start-major
            start_logp, _, pads = policy.pop_start_stats()       # [B, n], grad to head only
            assert start_logp.shape == (args.batch, n) and pads == 0
            adv = (R - R.mean(-1, keepdim=True)).detach()
            loss = -(adv * start_logp).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        v = val_oneshot()
        log["curve"].append(dict(epoch=ep, val_oneshot_prize=v,
                                 mean_loss=sum(losses) / len(losses),
                                 elapsed_s=round(time.time() - t0, 1)))
        print(f"epoch {ep} val one-shot prize {v:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    # the backbone must be bit-identical to the checkpoint it came from
    after = {k: v for k, v in policy.state_dict().items() if "start_head" not in k}
    assert all(torch.equal(frozen_before[k], after[k]) for k in frozen_before), "backbone moved"
    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "frozen_fit": log}, args.out_ckpt)
    json.dump(log, open(args.out_ckpt.replace(".ckpt", "_fit.json"), "w"), indent=2)
    print("done", args.out_ckpt, flush=True)


if __name__ == "__main__":
    main()
