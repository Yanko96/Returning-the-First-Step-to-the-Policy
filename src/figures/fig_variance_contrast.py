"""One-shot gap of the five SAS OP50 checkpoints, two ways to take the first step.
Data: artifacts/main_evals/five_seed/op50_sas_seed{42..46}.json. The plain-greedy
row bypasses the start head, the trained-head row decodes through it. Seed
identities are not drawn; the run-to-run spread is the point, not which run is
which.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

INK1, INK2 = "#0b0b0b", "#52514e"
SURF, GRID = "#fcfcfb", "#e7e6e2"
AQUA, BLUE = "#1baf7a", "#2a78d6"

# five SAS checkpoints, one-shot gap % to Compass 16.17, 10k instances
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
_runs = [json.load(open(os.path.join(ARTIFACT_ROOT, "main_evals", "five_seed",
                                     f"op50_sas_seed{s}.json")))["schemes"]
         for s in (42, 43, 44, 45, 46)]
plain = np.array([r["greedy"]["gap_pct"] for r in _runs])       # step zero bypassed
head = np.array([r["greedy_head"]["gap_pct"] for r in _runs])   # step zero learned

p_m, p_s, p_span = plain.mean(), plain.std(ddof=1), np.ptp(plain)
h_m, h_s, h_span = head.mean(), head.std(ddof=1), np.ptp(head)
paired = plain - head
ratio = p_s / h_s

fig, ax = plt.subplots(figsize=(7.4, 2.9), dpi=200)
fig.patch.set_facecolor(SURF)
ax.set_facecolor(SURF)

Y_PLAIN, Y_HEAD = 1.0, 0.0
ax.set_ylim(-0.62, 1.62)
ax.set_xlim(0, 18)

for x in range(0, 19, 2):
    ax.axvline(x, color=GRID, lw=0.8, zorder=0)

# top row: unlearned step zero
jitter_p = (-0.10, 0.0, 0.10, -0.05, 0.05)
for v, dy in zip(plain, jitter_p):
    ax.scatter(v, Y_PLAIN + dy, s=90, color=AQUA, zorder=3)
ax.plot([p_m, p_m], [Y_PLAIN - 0.20, Y_PLAIN + 0.20], color=INK1, lw=2, zorder=4)
ax.annotate(f"5 runs: {p_m:.2f} ± {p_s:.2f} points   (span {p_span:.2f})",
            (17.8, Y_PLAIN + 0.36), ha="right", fontsize=9.5, color=INK1)

# bottom row: learned step zero
jitter_h = (-0.10, 0.0, 0.10, -0.05, 0.05)
for v, dy in zip(head, jitter_h):
    ax.scatter(v, Y_HEAD + dy, s=90, color=BLUE, zorder=3)
ax.plot([h_m, h_m], [Y_HEAD - 0.20, Y_HEAD + 0.20], color=INK1, lw=2, zorder=4)
ax.annotate(f"5 runs: {h_m:.3f} ± {h_s:.3f} points   (span {h_span:.2f})",
            (h_m + 0.45, Y_HEAD + 0.34), ha="left", fontsize=9.5, color=INK1)
ax.annotate(f"paired gain {paired.mean():.2f} points "
            f"({paired.min():.2f} to {paired.max():.2f})",
            (h_m + 0.45, Y_HEAD - 0.42), ha="left", fontsize=8, color=INK2)

ax.set_yticks([Y_PLAIN, Y_HEAD])
ax.set_yticklabels(["free first step\n(plain greedy)",
                    "trained head\n(argmax start, ours)"],
                   fontsize=10, color=INK1)
ax.set_xlabel("OP50 gap to Compass (%), one-shot decoding, 10,000 instances",
              fontsize=9.5, color=INK2)
# No in-image title: the caption carries it.
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)
plt.tight_layout()

os.makedirs(OUTPUT_ROOT, exist_ok=True)
out = os.path.join(OUTPUT_ROOT, "fig_variance_contrast.png")
plt.savefig(out, facecolor=SURF, bbox_inches="tight")
print(f"saved {os.path.normpath(out)}")
print(f"  plain  mean={p_m:.4f} std={p_s:.4f} span={p_span:.4f}")
print(f"  head   mean={h_m:.4f} std={h_s:.4f} span={h_span:.4f}")
print(f"  ratio={ratio:.2f}  paired mean={paired.mean():.4f} "
      f"min={paired.min():.4f} max={paired.max():.4f}")
