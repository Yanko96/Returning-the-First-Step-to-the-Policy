"""Report — full 'SAS-uniform advantage vs rollout budget' curve, per seed.

Five points: greedy_head(1) / ms10(10) / ms25(25) / multistart(50) / msaug(800).
Reports advantage magnitude (pp of gap), paired Wilcoxon p, net wins, and the
decay shape / zero-crossing, for each of the five seed pairs (42-46). Reads
artifacts/main_evals/five_seed/ (set ARTIFACT_ROOT to read regenerated outputs).
"""
import json
import os

import numpy as np
from scipy import stats

REF = 16.17
R = os.path.join(os.environ.get("ARTIFACT_ROOT", "artifacts"), "main_evals", "five_seed")
OUT = os.environ.get("OUTPUT_ROOT", "outputs")
BUDGET = {"greedy_head": 1, "ms10": 10, "ms25": 25, "multistart": 50, "msaug": 800}

curves = {}
for seed in (42, 43, 44, 45, 46):
    sas = np.load(f"{R}/op50_sas_seed{seed}_perinst.npz")
    unif = np.load(f"{R}/op50_uniform_seed{seed}_perinst.npz")

    print(f"\n=== SAS-uniform advantage vs rollout budget (seed-{seed} pair, 10k) ===")
    print(f"{'scheme':<12}{'rollouts':>9}{'unif gap%':>11}{'SAS gap%':>10}{'adv pp':>9}"
          f"{'p(1-sided)':>12}{'net wins':>10}")
    curve = {}
    for k, b in BUDGET.items():
        u, s = unif[k], sas[k]
        gu, gs = (REF - u.mean()) / REF * 100, (REF - s.mean()) / REF * 100
        d = s.astype(np.float64) - u.astype(np.float64)
        p = stats.wilcoxon(d, alternative="greater")[1] if not np.allclose(d, 0) else 1.0
        nw = int((d > 0).sum()) - int((d < 0).sum())
        adv = gu - gs
        curve[k] = dict(rollouts=b, unif_gap=round(float(gu), 3), sas_gap=round(float(gs), 3),
                        adv_pp=round(float(adv), 3), p=float(p), net_wins=nw)
        print(f"{k:<12}{b:>9}{gu:>11.3f}{gs:>10.3f}{adv:>+9.3f}{p:>12.2e}{nw:>+10d}")

    advs = [curve[k]["adv_pp"] for k in BUDGET]
    sig = [curve[k]["p"] < 0.01 and curve[k]["adv_pp"] > 0 for k in BUDGET]
    names = list(BUDGET)
    zero = next((names[i] for i, s_ in enumerate(sig) if not s_), None)
    print(f"\nshape: {' -> '.join(f'{a:+.2f}' for a in advs)} (pp at 1/10/25/50/800 rollouts)")
    print("zero-crossing: " + (f"first non-significant/non-positive at '{zero}'" if zero
          else "advantage stays significantly positive through ALL five budgets"))
    curves[seed] = curve
os.makedirs(OUT, exist_ok=True)
with open(os.path.join(OUT, "advantage_curve_per_seed.json"), "w") as f:
    json.dump(curves, f, indent=2)
print("\nwritten", os.path.join(OUT, "advantage_curve_per_seed.json"))
