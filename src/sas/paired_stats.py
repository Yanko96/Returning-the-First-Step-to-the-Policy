# -*- coding: utf-8 -*-
"""Per-seed paired Wilcoxon (statistic + p) and paired effect sizes
(median paired improvement + IQR, in gap points) for SAS vs uniform,
OP50 and PCTSP50, over the five seeds the two configurations share (42-46).
Per-seed pairing, never pooled. Reads artifacts/main_evals/five_seed/."""
import glob
import os

import numpy as np
from scipy.stats import wilcoxon

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
FIVE = os.path.join(ARTIFACT_ROOT, "main_evals", "five_seed")


def one(problem, ref, load_u, load_s, to_gap_points):
    print(f"== {problem} ==")
    for k in ("greedy_head", "ms10", "ms25"):
        rows = []
        for s in ("42", "43", "44", "45", "46"):
            u = load_u(s)[k]
            v = load_s(s)[k]
            d = to_gap_points(u, v)  # >0 = SAS better, in gap points
            stat, p = wilcoxon(d, alternative="greater")
            q1, med, q3 = np.percentile(d, [25, 50, 75])
            rows.append((s, stat, p, med, q1, q3, (d > 0).sum(), (d < 0).sum()))
        for s, stat, p, med, q1, q3, w, l in rows:
            print(f"  {k:12s} seed{s}: W={stat:.3e} p={'<1e-300' if p == 0 else f'{p:.2e}'} "
                  f"median +{med:.3f}pp IQR [{q1:.3f}, {q3:.3f}] wins {w} losses {l}")


one("OP50 (uniform - SAS, prize -> gap points, ref 16.17)", 16.17,
    lambda s: np.load(os.path.join(FIVE, f"op50_uniform_seed{s}_perinst.npz")),
    lambda s: np.load(os.path.join(FIVE, f"op50_sas_seed{s}_perinst.npz")),
    lambda u, v: (v - u) / 16.17 * 100.0)

one("PCTSP50 (costs, ref 4.48)", 4.48,
    lambda s: np.load(os.path.join(FIVE, f"pctsp50_uniform_seed{s}_perinst.npz")),
    lambda s: np.load(os.path.join(FIVE, f"pctsp50_sas_seed{s}_perinst.npz")),
    lambda u, v: (u - v) / 4.48 * 100.0)
