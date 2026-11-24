"""Calibration curve (start-stakes -> realized benefit), from budget_sweep_curve.json.
Two series (realized = blue filled, oracle = aqua rings), OLS fit on the
realized benefit (dashed), budget labels per point, canonical budget marked.
Error bars are 3-seed standard deviations (mostly smaller than the markers).

Three off-distribution / cross-problem points from ood_points.json (PCTSP50,
OP100, OPLib) are overlaid as amber diamonds and excluded from the fit, which
uses the five OP50 budgets only. A zero line is added since OPLib is negative.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
CURVE = os.path.join(ARTIFACT_ROOT, "budget_sweep_curve.json")
OOD = os.path.join(ARTIFACT_ROOT, "ood_points.json")

INK1, INK2 = "#0b0b0b", "#52514e"
SURF, GRID = "#fcfcfb", "#e7e6e2"
AQUA, BLUE = "#1baf7a", "#2a78d6"
AMBER = "#c76a04"

rows = sorted(json.load(open(CURVE))["rows"], key=lambda r: r["stakes"])
x = np.array([r["stakes"] for r in rows])
y = np.array([r["realized_mean"] for r in rows])
ye = np.array([r["realized_std"] for r in rows])
o = np.array([r["oracle_mean"] for r in rows])
oe = np.array([r["oracle_std"] for r in rows])
T = [r["T"] for r in rows]

sl, ic = np.polyfit(x, y, 1)
r2 = np.corrcoef(x, y)[0, 1] ** 2

ood = json.load(open(OOD))
ood_pts = [
    ("PCTSP50", ood["pctsp50"]["x_stakes"], ood["pctsp50"]["realized_mean"],
     ood["pctsp50"]["realized_std"], (-8, -3), "right"),
    ("OP100", ood["op100"]["x_stakes"], ood["op100"]["realized_mean"],
     ood["op100"]["realized_std"], (6, -14), "left"),
    ("OPLib", ood["oplib"]["x_stakes"], ood["oplib"]["realized_mean"],
     ood["oplib"]["realized_std"], (6, 8), "left"),
]

fig, ax = plt.subplots(figsize=(6.4, 4.3), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)

ax.axhline(0.0, color=GRID, lw=1.0, zorder=1)
xs = np.linspace(x.min() - 0.012, x.max() + 0.012, 10)
# y = 100 * sigma: the stakes measure expressed in the same units as the
# y axis (both are shares of the same per-budget reference). Points below
# this line are settings where the measure exceeds the model's
# available start value.
ax.plot(xs, 100.0 * xs, ls=":", lw=1.3, color=INK2, alpha=0.8, zorder=2)
ax.annotate("$y = 100\\,\\sigma$", (xs[-1], 100.0 * xs[-1]),
            textcoords="offset points", xytext=(-6, 4), fontsize=8,
            color=INK2, ha="right")
ax.plot(xs, sl * xs + ic, ls="--", lw=1.2, color=BLUE, alpha=0.45, zorder=2)

for name, xo, yo, eo, off, ha in ood_pts:
    ax.errorbar([xo], [yo], yerr=[eo], ls="none", marker="D", markersize=7,
                color=AMBER, capsize=2, zorder=5)
    ax.annotate(name, (xo, yo), textcoords="offset points", xytext=off,
                fontsize=8.5, color=AMBER, ha=ha)

# PCTSP50's oracle start value, the one measured setting that sits on the
# dotted line rather than below it (-0.16 points against a standard error
# of 0.20).
_p = ood["pctsp50"]
ax.errorbar([_p["x_stakes"]], [_p["oracle_mean"]], yerr=[_p["oracle_std"]],
            ls="none", marker="D", markersize=7, markerfacecolor="none",
            markeredgewidth=1.8, color=AMBER, capsize=2, zorder=5)
ax.annotate("PCTSP50 oracle", (_p["x_stakes"], _p["oracle_mean"]),
            textcoords="offset points", xytext=(9, 2), fontsize=8.5,
            color=AMBER)

ax.errorbar(x, o, yerr=oe, ls="-", lw=1.4, color=AQUA, marker="o",
            markersize=8, markerfacecolor="none", markeredgewidth=1.8,
            capsize=2, zorder=3)
ax.errorbar(x, y, yerr=ye, ls="-", lw=1.6, color=BLUE, marker="o",
            markersize=7, capsize=2, zorder=4)

for xi, yi, t in zip(x, y, T):
    lab = f"$T{{=}}{t:g}$"
    dy = -1.7 if t != 3.0 else -2.1
    ax.annotate(lab, (xi, yi), textcoords="offset points",
                xytext=(4, -12 if t != 3.0 else -15),
                fontsize=8.5, color=INK2)
ax.annotate("canonical\nbenchmark", (x[2], y[2]), textcoords="offset points",
            xytext=(-12, 12), fontsize=7.5, color=INK2, ha="right")

ax.text(0.035, 0.96,
        f"OP50 fit: realized benefit $= {sl:.1f}\\times$stakes$ - {abs(ic):.1f}$   "
        f"($R^2 = {r2:.2f}$)",
        transform=ax.transAxes, fontsize=9, color=BLUE, va="top")
ax.text(0.035, 0.885,
        "capture $= 0.72$–$0.75$ at every OP50 length budget"
        "\nand $0.75$ on PCTSP50; $0.51$ at $n{=}500$",
        transform=ax.transAxes, fontsize=9, color=INK2, va="top")

ax.set_xlabel("start-stakes $\\sigma$ (model-free, measured by local search)",
              fontsize=9.5, color=INK1)
ax.set_ylabel("one-shot benefit over a random start (points)", fontsize=9.5,
              color=INK1)
# No in-image title: the caption carries it.

handles = [
    mlines.Line2D([], [], marker="o", ls="-", color=BLUE, markersize=7,
                  label="realized by the learned head (SAS)"),
    mlines.Line2D([], [], marker="o", ls="-", color=AQUA, markersize=8,
                  markerfacecolor="none", markeredgewidth=1.8,
                  label="oracle start value (best feasible start)"),
    mlines.Line2D([], [], marker="D", ls="none", color=AMBER, markersize=7,
                  label="off-distribution / cross-problem (not in fit)"),
]
# Lower LEFT, not lower right: the OPLib diamond and its label sit at
# (0.242, -0.9), i.e. inside the lower-right corner, and the legend collided
# with them. The lower-left wedge below the T=4.5 point is the only empty
# region large enough for three rows.
ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5,
          labelcolor=INK2)
ax.grid(True, color=GRID, lw=0.7)
ax.set_axisbelow(True)
for sp in ax.spines.values():
    sp.set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8.5)

plt.tight_layout()
os.makedirs(OUTPUT_ROOT, exist_ok=True)
out = os.path.join(OUTPUT_ROOT, "fig_calibration.png")
plt.savefig(out, facecolor=SURF, bbox_inches="tight")
print("saved", out)
