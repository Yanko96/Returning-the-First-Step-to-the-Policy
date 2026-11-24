"""Advantage-vs-budget curve (SAS - uniform, paired by seed), five seeds.
Both configurations carry seeds 42-46, so the pairing runs over all five; the
source is the five-seed evaluation in artifacts/main_evals/five_seed/.

Advantage at budget N = uniform gap%(N) - SAS gap%(N), same seed; mean +- std
over the five seed pairs. Budgets: 1 (one-shot: greedy_head for both arms)
/ 10 / 25 / 50 (= n) / 800 (msaug, the full-budget scheme).
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
R = os.path.join(ARTIFACT_ROOT, "main_evals", "five_seed")

INK1, INK2 = "#0b0b0b", "#52514e"
SURF, GRID = "#fcfcfb", "#e7e6e2"
BLUE = "#2a78d6"
BAND = "#eaf1fb"

BUDGETS = [("greedy_head", 1), ("ms10", 10), ("ms25", 25),
           ("multistart", 50), ("msaug", 800)]
SEEDS = ["42", "43", "44", "45", "46"]


def gaps(arm, seed):
    f = os.path.join(R, f"op50_{arm}_seed{seed}.json")
    return json.load(open(f))["schemes"]


adv = {k: [] for k, _ in BUDGETS}
for s in SEEDS:
    u = gaps("uniform", s)
    v = gaps("sas", s)
    for k, _ in BUDGETS:
        adv[k].append(u[k]["gap_pct"] - v[k]["gap_pct"])

xs = np.array([n for _, n in BUDGETS], dtype=float)
means = np.array([np.mean(adv[k]) for k, _ in BUDGETS])
stds = np.array([np.std(adv[k], ddof=1) for k, _ in BUDGETS])

out_json = {k: dict(rollouts=n, per_seed=[round(a, 4) for a in adv[k]],
                    mean=round(float(np.mean(adv[k])), 4),
                    std=round(float(np.std(adv[k], ddof=1)), 4),
                    two_se=round(float(2 * np.std(adv[k], ddof=1)
                                       / np.sqrt(len(SEEDS))), 4))
            for k, n in BUDGETS}
os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(os.path.join(OUTPUT_ROOT, "advantage_curve_5seed.json"), "w",
          encoding="utf-8") as f:
    json.dump(out_json, f, indent=2)
print(json.dumps(out_json, indent=2))

fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)
ax.set_xscale("log")

ax.axvspan(1, 25, color=BAND, zorder=0)
ax.annotate("planning-loop regime\n(1 to 25 rollouts)",
            (np.sqrt(1 * 25), 6.4), ha="center", fontsize=9, color=INK2)

ax.axhline(0, color=GRID, lw=1.2, zorder=1)
for y in (2, 4, 6, 8):
    ax.axhline(y, color=GRID, lw=0.8, zorder=0)

ax.errorbar(xs, means, yerr=stds, color=BLUE, lw=2, marker="o", markersize=6.5,
            capsize=3, zorder=3)

offsets = [(10, 4), (8, 6), (8, 6), (0, -18), (0, 11)]
for (k, n), m, s, off in zip(BUDGETS, means, stds, offsets):
    ax.annotate(f"+{m:.2f}" if m >= 0.005 else f"{m:+.3f}",
                (n, m), xytext=off, textcoords="offset points",
                fontsize=9, color=INK1, ha="left" if off[0] > 0 else "center")

ax.annotate("reaches zero at $N = n = 50$\n(not significant in any seed)",
            (50, 0.05), xytext=(110, 2.9), textcoords="data",
            fontsize=9, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8,
                            connectionstyle="arc3,rad=-0.15"))

ax.set_xticks(xs)
ax.set_xticklabels(["1", "10", "25", "50", "800"])
ax.tick_params(axis="x", which="minor", bottom=False)
ax.set_xlabel("decoding budget (rollouts per instance, log scale)",
              fontsize=9.5, color=INK2)
ax.set_ylabel("advantage over the uniform control\n(gap points, higher = better)",
              fontsize=9.5, color=INK2)
# No in-image title: the caption carries it.
ax.set_ylim(-0.6, 8.6)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
plt.tight_layout()
out = os.path.join(OUTPUT_ROOT, "fig_advantage_curve.png")
plt.savefig(out, facecolor=SURF, bbox_inches="tight")
print("saved", os.path.normpath(out))
