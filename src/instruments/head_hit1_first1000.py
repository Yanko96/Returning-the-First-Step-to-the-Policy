"""hit@1 on 1000 canonical instances, OP50 + PCTSP50, local CPU forwards.

Extends the 200-instance head-vs-nearopt statistic (make_head_overlay_fig.py)
to the first 1000 instances of each canonical test set, against the
1000-instance near-optimal start sets (nearopt_starts_op50_first1000.json /
pctsp_nearopt_first1000.json). Head snapshot convention identical to the
evaluation harnesses: encoder embeddings -> start_head -> depot masked ->
softmax; argmax index shares the full-graph indexing of the ILS start sets
(0 = depot). Forwards run in 200-row chunks.

Per-seed 95% CI: Wilson score interval on the 1000-instance hit count.
"""
import json
import os
import pickle
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sas"))

from rl4co.envs import OPEnv, PCTSPEnv  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from sas_pomo import SASPOMO  # noqa: E402

N_FIRST = 1000
CHUNK = 200

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
OP_PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
PCTSP_PKL = os.path.join(DATA_ROOT, "pctsp", "pctsp50_test_seed1234.pkl")
OP_CKPT = os.path.join(CHECKPOINT_ROOT, "sas_op50_seed{sd}", "last.ckpt")
PCTSP_CKPT = os.path.join(CHECKPOINT_ROOT, "sas_pctsp50_seed{sd}", "last.ckpt")
OP_NEAR = os.path.join(ARTIFACT_ROOT, "nearopt",
                       "nearopt_starts_op50_first1000.json")
PCTSP_NEAR = os.path.join(ARTIFACT_ROOT, "nearopt",
                          "pctsp_nearopt_first1000.json")


def load_op(path):
    with open(path, "rb") as f:
        data = pickle.load(f)[:N_FIRST]
    return TensorDict(dict(
        depot=torch.tensor([d[0] for d in data], dtype=torch.float32),
        locs=torch.tensor([d[1] for d in data], dtype=torch.float32),
        prize=torch.tensor([d[2] for d in data], dtype=torch.float32),
        max_length=torch.tensor([d[3] for d in data], dtype=torch.float32),
    ), batch_size=[len(data)])


def load_pctsp(path):
    with open(path, "rb") as f:
        data = pickle.load(f)[:N_FIRST]
    return TensorDict(dict(
        depot=torch.tensor([d[0] for d in data], dtype=torch.float32),
        locs=torch.tensor([d[1] for d in data], dtype=torch.float32),
        penalty=torch.tensor([d[2] for d in data], dtype=torch.float32),
        deterministic_prize=torch.tensor([d[3] for d in data], dtype=torch.float32),
        stochastic_prize=torch.tensor([d[4] for d in data], dtype=torch.float32),
    ), batch_size=[len(data)])


def head_probs(env, td_all, ckpt):
    model = SASPOMO(env, sas_arm="sas")
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd, strict=True)
    policy = model.policy.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, td_all.shape[0], CHUNK):
            td = env.reset(td_all[i:i + CHUNK].clone())
            h, _ = policy.encoder(td)
            mask = td["action_mask"].clone()
            mask[..., 0] = False
            logits = policy.start_head(h).masked_fill(~mask, float("-inf"))
            out.append(torch.softmax(logits, dim=-1).numpy())
    return np.concatenate(out)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return ctr - hw, ctr + hw


def run(tag, env, td_all, ckpt_tpl, near_path, n_nodes=50):
    near = json.load(open(near_path))
    S = [set(r["nearopt_start_set"]) for r in near["instances"]]
    assert len(S) == N_FIRST
    sizes = np.array([len(s) for s in S])
    res = dict(n_instances=N_FIRST,
               chance_hit1=float(sizes.mean() / n_nodes),
               start_set_size_mean=float(sizes.mean()), per_seed={})
    for sd in ("42", "43", "44"):
        hp = head_probs(env, td_all, ckpt_tpl.format(sd=sd))
        arg = hp.argmax(1)
        hits = np.array([int(arg[i]) in S[i] for i in range(N_FIRST)])
        mass = np.array([hp[i, sorted(S[i])].sum() for i in range(N_FIRST)])
        lo, hi = wilson(int(hits.sum()), N_FIRST)
        res["per_seed"][sd] = dict(
            hit_at_1=float(hits.mean()),
            wilson95=[float(lo), float(hi)],
            hit_at_1_first200=float(hits[:200].mean()),
            head_mass_on_S=float(mass.mean()))
        print(f"[{tag} seed {sd}] hit@1 {hits.mean():.4f} "
              f"(95% CI {lo:.4f}-{hi:.4f}), first-200 subset {hits[:200].mean():.4f}",
              flush=True)
    h = [res["per_seed"][s]["hit_at_1"] for s in res["per_seed"]]
    res["hit_at_1_mean"] = float(np.mean(h))
    res["hit_at_1_std"] = float(np.std(h, ddof=1))
    k_pool = int(round(sum(h) * N_FIRST))
    lo, hi = wilson(k_pool, 3 * N_FIRST)
    res["pooled_wilson95"] = [float(lo), float(hi)]
    print(f"[{tag}] mean {res['hit_at_1_mean']:.4f} +/- {res['hit_at_1_std']:.4f}, "
          f"chance {res['chance_hit1']:.4f}", flush=True)
    return res


def main():
    torch.manual_seed(0)
    out = {}
    out["op50"] = run("OP50", OPEnv(generator_params=dict(num_loc=50)),
                      load_op(OP_PKL), OP_CKPT, OP_NEAR)
    out["pctsp50"] = run("PCTSP50", PCTSPEnv(generator_params=dict(num_loc=50)),
                         load_pctsp(PCTSP_PKL), PCTSP_CKPT, PCTSP_NEAR)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    dst = os.path.join(OUTPUT_ROOT, "head_hit1_first1000.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=2)
    print("saved", dst)
    print("HIT1_1000_DONE")


if __name__ == "__main__":
    main()
