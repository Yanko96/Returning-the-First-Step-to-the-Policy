"""Regime study: measure the post-normalization
geometry of the OPLib target bracket (101<=n<=300), so the synthetic generator's
geometry mixture can be fixed to COVER the target — matching WIDTH, not mean.
No model, no treatment.

Two complementary scales (median R=1.0 means ~half the bracket is NN-indistinguishable
from uniform, so R alone may miss large-scale density gradients typical of TSPLIB
cities — hence VMR):
  - R  = Clark-Evans nearest-neighbour index = mean-NN / (0.5*sqrt(A/n)).
         R<1 clustered, ~1 random(CSR), >1 regular. NN scale.
  - VMR = variance-to-mean of 8x8 quadrat counts on the normalized bbox.
         VMR>1 large-scale clumping, ~1 CSR, <1 regular. Coarse scale.
Both on the SAME normalized coords the model sees (oplib_parser.normalize).
gen1/2/3 share coordinates -> geometry measured once on the 22 distinct instances.
n-dependence of R is checked (Spearman R vs n); if a trend exists, the n~200 subset
is the primary calibration reference.
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oplib_parser import normalize, parse_oplib  # noqa: E402


def clark_evans_R(pts):
    n = len(pts)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    d_obs = d.min(1).mean()
    span = pts.max(0) - pts.min(0)
    A = float(span[0] * span[1])
    return float(d_obs / (0.5 / np.sqrt(n / A)))


def vmr(pts, grid=8):
    lo = pts.min(0)
    span = pts.max(0) - lo
    span = np.where(span > 0, span, 1.0)
    u = (pts - lo) / span
    ix = np.clip((u[:, 0] * grid).astype(int), 0, grid - 1)
    iy = np.clip((u[:, 1] * grid).astype(int), 0, grid - 1)
    counts = np.bincount(ix * grid + iy, minlength=grid * grid).astype(float)
    return float(counts.var() / counts.mean())


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def qs(a):
    a = np.asarray(a, float)
    return {q: round(float(np.percentile(a, q)), 3) for q in (10, 25, 50, 75, 90)}


def bracket_rows():
    """Per-instance (name, n, R, VMR) for the 101<=n<=300 OPLib bracket (gen1 = distinct
    geometries). Helper so the joint-coverage check reuses the SAME R/VMR
    on the SAME normalized coords; does not change main()'s output."""
    base = os.path.join(ROOT, "data", "oplib_full", "OPLib-master", "instances", "gen1")
    rows, skipped = [], 0
    for f in sorted(glob.glob(os.path.join(base, "*.oplib"))):
        try:
            raw = parse_oplib(f)
        except NotImplementedError:
            skipped += 1
            continue
        if not (101 <= raw.dimension <= 300):
            continue
        norm = normalize(raw)
        pts = np.concatenate([np.asarray(norm["depot"], float).reshape(1, 2),
                              np.asarray(norm["loc"], float)], 0)
        rows.append((raw.name, raw.dimension, round(clark_evans_R(pts), 3), round(vmr(pts), 3)))
    return rows, skipped


def main():
    rows, skipped = bracket_rows()
    Rs = [r[2] for r in rows]; Vs = [r[3] for r in rows]; ns = [r[1] for r in rows]
    print(f"{len(rows)} distinct bracket geometries (x3 prize = {3*len(rows)} target instances); "
          f"skipped {skipped} unsupported")
    print(f"R   quantiles {qs(Rs)}  range [{min(Rs):.3f},{max(Rs):.3f}]")
    print(f"VMR quantiles {qs(Vs)}  range [{min(Vs):.3f},{max(Vs):.3f}]  (CSR~1.0)")
    print(f"n-trend: Spearman(R,n)={spearman(Rs, ns):+.3f}  Spearman(VMR,n)={spearman(Vs, ns):+.3f}")
    # R~1 subset (near-CSR at NN scale): is it clumped at coarse scale?
    near = [(nm, n, R, V) for (nm, n, R, V) in rows if 0.9 <= R <= 1.1]
    if near:
        print(f"\nR~1 subset (0.9<=R<=1.1): {len(near)} instances | VMR quantiles {qs([x[3] for x in near])}")
        print("  -> if this VMR is materially >1, R alone misses large-scale structure, so VMR is matched as well")
        for nm, n, R, V in near:
            print(f"     {nm:14s} n={n:4d} R={R} VMR={V}")
    print("\n--- all (name,n,R,VMR) ---")
    for nm, n, R, V in rows:
        print(f"{nm:14s} n={n:4d} R={R:5.3f} VMR={V:6.3f}")


if __name__ == "__main__":
    main()
