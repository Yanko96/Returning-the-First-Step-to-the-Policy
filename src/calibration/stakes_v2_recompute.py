"""Recompute the v2 start-stakes (stakes_first200_v2.json) from shipped inputs.

Inputs (all under ARTIFACT_ROOT):
  stakes/stakes_first200.json                  v1 instrument output: per-instance
                                               mean_forced / best_free per tier
  nearopt/nearopt_T{T}_first1000.json          T in {1.5, 2, 4, 4.5}
  nearopt/nearopt_starts_op50_first1000.json   anchor tier T=3.0

Definition (the paper's stakes definition): per instance i,
  stakes_i = clip(1 - mean_forced_i / ref_best_i, 0, 1)
with ref_best_i the per-instance 16-restart reference best from the
near-optimal start-set files; per-tier stakes = mean over the first 200
instances. The recomputation is verified against the shipped
stakes/stakes_first200_v2.json to < 1e-9 and the result is written to
OUTPUT_ROOT/stakes_first200_v2.json.
"""
import json
import os

import numpy as np

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
TIERS = [1.5, 2.0, 3.0, 4.0, 4.5]
N_INST = 200


def nearopt_path(T):
    if T == 3.0:
        return os.path.join(ARTIFACT_ROOT, "nearopt",
                            "nearopt_starts_op50_first1000.json")
    return os.path.join(ARTIFACT_ROOT, "nearopt", f"nearopt_T{T:g}_first1000.json")


v1 = json.load(open(os.path.join(ARTIFACT_ROOT, "stakes", "stakes_first200.json")))
shipped = json.load(open(os.path.join(ARTIFACT_ROOT, "stakes",
                                      "stakes_first200_v2.json")))

tiers_out = {}
max_err = 0.0
for T in TIERS:
    near = json.load(open(nearopt_path(T)))
    ils_ref = {r["idx"]: r["best_prize"] for r in near["instances"]}
    pi = v1["tiers"][str(T)]["per_instance"]
    mean_forced = np.array([pi[i]["mean_forced"] for i in range(N_INST)])
    best_free = np.array([pi[i]["best_free"] for i in range(N_INST)])
    ref_best = np.array([ils_ref[i] for i in range(N_INST)])
    raw = 1.0 - mean_forced / ref_best
    st = np.clip(raw, 0.0, 1.0)
    tiers_out[str(T)] = dict(
        stakes_mean=float(st.mean()), stakes_std=float(st.std(ddof=1)),
        lite_free_over_ref_mean=float((best_free / ref_best).mean()),
        clipped=int((raw < 0).sum()),
        per_instance_stakes=[float(v) for v in st])
    ship = shipped["tiers"][str(T)]
    err = abs(tiers_out[str(T)]["stakes_mean"] - ship["stakes_mean"])
    err_pi = float(np.max(np.abs(st - np.array(ship["per_instance_stakes"]))))
    max_err = max(max_err, err, err_pi)
    print(f"T={T:g}: stakes_mean recomputed {st.mean():.12f} "
          f"shipped {ship['stakes_mean']:.12f} |diff| {err:.3e} "
          f"(per-instance max |diff| {err_pi:.3e})")

assert max_err < 1e-9, f"recomputation does not match shipped v2 (max err {max_err})"
os.makedirs(OUTPUT_ROOT, exist_ok=True)
out = {k: shipped[k] for k in shipped if k != "tiers"}
out["tiers"] = tiers_out
dst = os.path.join(OUTPUT_ROOT, "stakes_first200_v2.json")
json.dump(out, open(dst, "w"))
print(f"V2_RECOMPUTE_OK max_err {max_err:.3e}; wrote {dst}")
