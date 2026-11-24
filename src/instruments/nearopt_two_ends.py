"""How often the near-optimal start set is exactly the two ends of one tour (Section 3.5).

Section 3.5 reports that the near-optimal start set S has 2.2 nodes per instance on
average, never more than 5, and that on 83% of the first 1,000 OP50 test instances
it is exactly the two ends of the best tour. S is the union, over the restart tours
within 99.5% of the best of 16, of the two depot-adjacent nodes {t[1], t[-1]}, as
shipped for the canonical T = 3 in artifacts/nearopt/. This reads those sets and
counts the instances whose S equals {best_tour[1], best_tour[-1]}; no solver is
rerun (CPU, seconds).

    python src/instruments/nearopt_two_ends.py

Expected: n_instances 1000, start_set_size_mean 2.215, start_set_size_max 5,
n_two_ends_of_best_tour 830 (0.83). Writes outputs/nearopt_two_ends.json.
"""
import json
import os

import numpy as np

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
SRC = os.path.join(ARTIFACT_ROOT, "nearopt", "nearopt_starts_op50_first1000.json")

d = json.load(open(SRC))
inst = d["instances"]
sizes = np.array([len(r["nearopt_start_set"]) for r in inst])
two_ends = [set(r["nearopt_start_set"]) == {r["best_tour"][1], r["best_tour"][-1]}
            for r in inst]

out = {
    "source": SRC.replace(os.sep, "/"),
    "pkl_md5": d["pkl_md5"],
    "n_restarts": d["n_restarts"],
    "nearopt_frac": d["nearopt_frac"],
    "n_instances": len(inst),
    "start_set_size_mean": round(float(sizes.mean()), 4),
    "start_set_size_max": int(sizes.max()),
    "n_size_two": int((sizes == 2).sum()),
    "n_two_ends_of_best_tour": int(sum(two_ends)),
    "share_two_ends_of_best_tour": round(float(np.mean(two_ends)), 4),
}
os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(os.path.join(OUTPUT_ROOT, "nearopt_two_ends.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
