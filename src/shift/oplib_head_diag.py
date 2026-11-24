"""SAS head diagnostics on OPLib.
Forward encoder+head of the 3 SAS checkpoints on all 210 included
instances (same subset/normalization as oplib_eval). Records per
instance: argmax, top-50 (idx, p), IPR, top-1 mass; full softmax for
n<=300. Masking identical to the OP50 head snapshot (env action_mask
after reset, depot forced off)."""
import glob
import json
import os

import numpy as np
import torch
from tensordict import TensorDict

import sys

from rl4co.envs import OPEnv

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "instruments"))
from oplib_parser import parse_oplib, normalize  # noqa: E402
from oplib_eval import build_model  # noqa: E402  (tolerance patch irrelevant: no decode)

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
CKPTS = {s: os.path.join(CHECKPOINT_ROOT, f"sas_op50_seed{s}", "last.ckpt")
         for s in (42, 43, 44)}
ROOT = os.environ.get("OPLIB_ROOT",
                      os.path.join(os.environ.get("DATA_ROOT", "data"),
                                   "OPLib-master"))

paths = []
for gen in (1, 2, 3):
    for p in sorted(glob.glob(os.path.join(ROOT, "instances", f"gen{gen}", "*.oplib"))):
        try:
            raw = parse_oplib(p)
        except NotImplementedError:
            continue
        if raw.edge_weight_type == "EUC_2D":
            paths.append((p, gen))
print(f"{len(paths)} instances")

env0 = OPEnv(generator_params=dict(num_loc=100))
for seed, ck in CKPTS.items():
    model = build_model("sas", env0)
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    policy = model.policy.to("cuda").eval()
    out = {}
    meta = []
    for p, gen in paths:
        raw = parse_oplib(p)
        norm = normalize(raw)
        n = raw.dimension - 1
        td0 = TensorDict(dict(
            depot=torch.tensor(norm["depot"], dtype=torch.float32)[None],
            locs=torch.tensor(norm["loc"], dtype=torch.float32)[None],
            prize=torch.tensor(norm["prize"], dtype=torch.float32)[None],
            max_length=torch.tensor([norm["max_length"]], dtype=torch.float32),
        ), batch_size=[1])
        env = OPEnv(generator_params=dict(num_loc=n))
        with torch.inference_mode():
            td = env.reset(td0).to("cuda")
            h, _ = policy.encoder(td)
            mask = td["action_mask"].clone()
            mask[..., 0] = False
            logits = policy.start_head(h).masked_fill(~mask, float("-inf"))
            probs = torch.softmax(logits, dim=-1)[0].float().cpu().numpy()
        name = os.path.basename(p).replace(".oplib", "")
        ipr = float(1.0 / (probs ** 2).sum())
        top = np.argsort(-probs)[:50]
        out[f"{name}__top_idx"] = top.astype(np.int32)
        out[f"{name}__top_p"] = probs[top].astype(np.float32)
        if n <= 300:
            out[f"{name}__full"] = probs.astype(np.float32)
        meta.append(dict(name=name, gen=gen, n=n, ipr=round(ipr, 4),
                         top1=round(float(probs.max()), 6),
                         argmax=int(probs.argmax())))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    np.savez_compressed(os.path.join(OUTPUT_ROOT,
                                     f"oplib_head_diag_seed{seed}.npz"), **out)
    with open(os.path.join(OUTPUT_ROOT,
                           f"oplib_head_diag_seed{seed}.json"), "w") as f:
        json.dump(meta, f)
    print(f"seed {seed}: mean IPR {np.mean([m['ipr'] for m in meta]):.3f}, "
          f"mean top1 {np.mean([m['top1'] for m in meta]):.4f}", flush=True)
print("HEAD_DIAGNOSTIC_COMPLETE")
