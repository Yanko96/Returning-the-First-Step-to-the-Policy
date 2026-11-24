"""Achievability probe: is the head's argmax start
achievable-near-optimal even though it falls outside the ILS 16-restart
union? lite-ILS forced at the head's argmax vs lite free best,
T=2.0 vs the T=3.0 control, first 200 instances. Diagnostic only.
"""
import os
import pickle
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import budget_sweep_stakes as fs  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
BK = os.path.join(ARTIFACT_ROOT, "op50_sweep_evals")

with open(PKL, "rb") as f:
    DATA = pickle.load(f)

CASES = [("T2", 2.0), ("T3", 3.0)]
ARGS = {}
for tag, T in CASES:
    hp = np.load(os.path.join(BK, f"eval_{tag}_s42_perinst.npz"))["head_probs"]
    ARGS[tag] = hp.argmax(1)


def job(a):
    tag, T, idx = a
    depot, loc, prize_raw, _ = DATA[idx]
    xy = np.vstack([np.asarray(depot, float)[None, :], np.asarray(loc, float)])
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    prize = np.concatenate([[0.0], np.asarray(prize_raw, float)])
    s = int(ARGS[tag][idx])
    seed0 = 900000 + idx * 100
    free = max(fs.lite_solve(D, prize, T, seed0 + r) for r in range(8))
    forced = max(fs.lite_solve(D, prize, T, seed0 + 50 + r, forced=s)
                 for r in range(4))
    return tag, idx, forced / max(free, forced, 1e-9)


def main():
    jobs = [(tag, T, i) for tag, T in CASES for i in range(200)]
    acc = {"T2": [], "T3": []}
    idxs = {"T2": [], "T3": []}
    with Pool(processes=14) as pool:
        for tag, i, ratio in pool.imap_unordered(job, jobs, chunksize=4):
            acc[tag].append(ratio); idxs[tag].append(i)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    np.savez(os.path.join(OUTPUT_ROOT, "achievability_probe_perinst.npz"),
             **{t: np.array(sorted(zip(idxs[t], acc[t]))) for t in acc})
    for tag in ("T2", "T3"):
        r = np.array(acc[tag])
        print(f"{tag}: head-argmax achievable/free = {r.mean():.4f} "
              f"(median {np.median(r):.4f}); frac>=0.995: {(r >= 0.995).mean():.3f}; "
              f"frac>=0.98: {(r >= 0.98).mean():.3f}")
    print("PROBE_DONE")


if __name__ == "__main__":
    main()
