"""SAS head start distribution vs near-optimal start subset.

Two outputs:
1. head_vs_nearopt_stats.json — aggregate over the first 1000 canonical
   instances x 3 seeds (upgraded from 200; per-seed probs from
   head_hit1_first1000.py CPU forwards, sets from
   nearopt_starts_op50_first1000.json):
   hit@1 = P(head argmax in near-opt start set S_i), vs chance = mean(|S_i|)/50
   (= expected hit rate of a uniform-random start, and of the enumerated
   arm's fixed node-1 start by symmetry). S_i is a 16-restart SAMPLE of the near-optimal
   space -> hit@1 is a LOWER bound on the true rate.
2. fig_head_vs_nearopt.png — geometry small multiples, instances 0/1/2
   (fixed rule: first three, no selection), seed 42 head:
   dot size ~ prize, aqua ring = near-opt start set, blue fill = head argmax,
   square = depot. Colors #1baf7a (near-opt set) / #2a78d6 (head),
   direct labels, no legend box (roles named in subtitle), recessive frame.
"""
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
PKL = os.path.join(DATA_ROOT, "op", "op_dist50_test_seed1234.pkl")

INK1, INK2 = "#0b0b0b", "#52514e"
SURF, GRID = "#fcfcfb", "#e7e6e2"
AQUA, BLUE = "#1baf7a", "#2a78d6"  # near-opt set / learned head (same roles as fig_variance_contrast)

near = json.load(open(os.path.join(ARTIFACT_ROOT, "nearopt",
                                   "nearopt_starts_op50_first1000.json")))
S = [set(r["nearopt_start_set"]) for r in near["instances"]]
sizes = np.array([len(s) for s in S])

# aggregate stat: 1000-instance x 3-seed run (head_hit1_first1000.py)
big = json.load(open(os.path.join(ARTIFACT_ROOT, "head_diag",
                                  "head_hit1_first1000.json")))["op50"]
stats = {"chance_hit1": big["chance_hit1"],
         "start_set_size_mean": big["start_set_size_mean"],
         "n_instances": big["n_instances"],
         "per_seed": big["per_seed"],
         "hit_at_1_mean": big["hit_at_1_mean"],
         "hit_at_1_std": big["hit_at_1_std"]}
os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(os.path.join(OUTPUT_ROOT, "head_vs_nearopt_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print(json.dumps(stats, indent=2))

# ---------- geometry small multiples ----------
data = pickle.load(open(PKL, "rb"))
hp42 = np.load(os.path.join(ARTIFACT_ROOT, "main_evals", "op50",
               "sas_op50_seed42_eval_perinst.npz"))["head_probs_first200"]

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.5), dpi=200)
fig.patch.set_facecolor(SURF)
for ax, idx in zip(axes, (0, 1, 2)):
    depot, loc, prize, _T = data[idx]
    loc = np.asarray(loc)
    prize = np.asarray(prize)
    ax.set_facecolor(SURF)
    ax.scatter(loc[:, 0], loc[:, 1], s=8 + 52 * prize, color="#c9c7c2", zorder=2)
    ss = sorted(S[idx])
    ax.scatter(loc[[i - 1 for i in ss], 0], loc[[i - 1 for i in ss], 1],
               s=[46 + 52 * prize[i - 1] for i in ss], facecolor="none",
               edgecolor=AQUA, lw=1.8, zorder=3)
    arg = int(hp42[idx].argmax())
    ax.scatter(loc[arg - 1, 0], loc[arg - 1, 1], s=14 + 52 * prize[arg - 1],
               color=BLUE, zorder=4)
    ax.scatter(*np.asarray(depot), s=70, marker="s", color=INK1, zorder=4)
    hit = "in set" if arg in S[idx] else "outside set"
    ax.set_title(f"instance {idx} (head pick {hit})", fontsize=9, color=INK2, loc="left")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)

handles = [
    mlines.Line2D([], [], marker="s", ls="none", color=INK1, markersize=7, label="depot"),
    mlines.Line2D([], [], marker="o", ls="none", color="#c9c7c2", markersize=7,
                  label="node (size = prize)"),
    mlines.Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                  markeredgecolor=AQUA, markeredgewidth=1.8, markersize=9,
                  label="near-optimal start set (ILS, 16 restarts)"),
    mlines.Line2D([], [], marker="o", ls="none", color=BLUE, markersize=7,
                  label="head argmax (ours, seed 42)"),
]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
           fontsize=8.5, labelcolor=INK2, bbox_to_anchor=(0.5, -0.02))
# No in-image title: the caption gives the aggregate hit@1 and chance rate.
plt.tight_layout(rect=(0, 0.05, 1, 1))
out = os.path.join(OUTPUT_ROOT, "fig_head_vs_nearopt.png")
plt.savefig(out, facecolor=SURF, bbox_inches="tight")
print("saved", out)
