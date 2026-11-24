"""Feasible first moves at the five length budgets (Appendix B).

Appendix B reports, on the first 1,000 canonical OP50 test instances, that at T = 1.5 a mean
of 39.4 of the 50 nodes are feasible starts and fewer than 50 are feasible on 90.7% of
instances, at T = 2.0 48.6 and 45.3%, and at T = 3.0, 4.0 and 4.5 every start is feasible on
every instance. A node s is a feasible first move iff the depot round trip fits the length
budget, 2 * d(depot, s) <= T, which is the first-step action mask of the OP environment.
The output field pct_instances_guard is the share of instances with fewer than N = 50
feasible starts. On such an instance the start sampler (src/sas/sas_pomo.py, the
`n_feas < k` branch) takes each feasible start once and draws the remaining starts with
replacement from the head's softmax over the feasible nodes. CPU, seconds; no model involved.

    python src/instruments/guard_feasibility.py

Writes outputs/guard_feasibility_five_tiers.json (the shipped copy is
artifacts/guard_feasibility_five_tiers.json).
"""
import hashlib
import json
import os
import pickle

import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")
N_INST, N_STARTS = 1000, 50
BUDGETS = (1.5, 2.0, 3.0, 4.0, 4.5)

raw = open(PKL, "rb").read()
data = pickle.loads(raw)[:N_INST]
depot = np.array([d[0] for d in data], dtype=np.float64)
loc = np.array([d[1] for d in data], dtype=np.float64)
round_trip = 2.0 * np.linalg.norm(loc - depot[:, None, :], axis=-1)  # [N_INST, 50]

rows = []
for T in BUDGETS:
    feas = (round_trip <= T).sum(-1)
    rows.append(dict(T=T, n_inst=N_INST,
                     mean_feasible=round(float(feas.mean()), 2),
                     pct_starts_infeasible=round(100.0 * float((round_trip > T).mean()), 2),
                     pct_instances_guard=round(100.0 * float((feas < N_STARTS).mean()), 1),
                     min_feasible=int(feas.min())))
out = dict(what="budget-feasible first moves and share of instances with fewer than 50 feasible starts, N=n=50 starts requested",
           rule="node s is feasible iff 2*d(depot,s) <= T ; pct_instances_guard counts instances with n_feas < 50",
           pkl="op/op_dist50_test_seed1234.pkl (under DATA_ROOT)", pkl_md5=hashlib.md5(raw).hexdigest(),
           n_instances=N_INST, num_starts=N_STARTS,
           max_observed_round_trip=round(float(round_trip.max()), 4),
           theoretical_max_round_trip=round(2.0 * np.sqrt(2.0), 4),
           rows=rows)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(os.path.join(OUTPUT_ROOT, "guard_feasibility_five_tiers.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
