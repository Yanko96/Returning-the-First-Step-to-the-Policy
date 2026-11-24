"""Clustered OP geometries for the controlled pool (regime_pool.py --geometry mix).

Per-instance 4-regime geometry to COVER the OPLib bracket's post-normalization
R and VMR distributions (match width, not mean):
  uniform      (R~1,       VMR~1)   = Kool native
  gmm          (R<1,       VMR>1)   = C centers ~U[cmin,cmax], intra-cluster sigma ~U[smin,smax]
  hardcore     (1<R<~1.4,  VMR~1)   = Poisson-disk / dart-throw, min sep dmin=c/sqrt(m)
  jittered_grid(R>1.4,     VMR<1)   = regular grid + uniform jitter (kept, unused by default)
Regime sampled per instance with {p_u,p_c,p_h,p_g}. The hard-core regime
decouples "raise the R high tail toward 1.2-1.3" from "crash VMR below 1": a
min-separation dart process is NN-regular (R>1) yet coarse-random (VMR~1), so it
covers the R 75/90 target WITHOUT the jittered-grid's VMR undershoot. All geometry
pushed through the SAME normalization as oplib_parser (bbox-min->0, divide by
longest side) prior to any measurement or output -- else clustered instances
crammed in a box corner != box-filling GMM (the silent failure mode here).

This file (GEOMETRY + --measure) is used for coverage tuning (geometry
only, no budget). Prize (gen1/2/3 equal) and budget (max_length = alpha*L, L =
greedy+2-opt TSP, alpha~U[.45,.55]) are added when the fixed training POOL is
generated (the regime study uses a pre-generated fixed pool, not fresh-per-epoch,
because per-instance 2-opt L on-the-fly would starve the GPU).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# NOTE: clark_evans_R/qs/vmr imported LAZILY inside measure() only — that module pulls
# oplib_parser (same directory), which only the coverage measurement needs. gen_geometry
# (used by the pool generator) has no such dependency, so pool gen imports cleanly everywhere.

# fixed target (oplib_bracket_geom.txt) + fixed tolerances
TGT_R = {10: 0.458, 25: 0.591, 50: 0.999, 75: 1.211, 90: 1.314}
TGT_VMR = {10: 0.924, 25: 0.992, 50: 1.962, 75: 4.896, 90: 8.722}
R_TOL = 0.10                       # abs, per percentile
VMR_TOL = {25: 1.41, 50: 1.41, 75: 1.41, 10: 2.0, 90: 2.0}  # factor (log)


def normalize(pts):
    lo = pts.min(0)
    scale = float((pts.max(0) - lo).max())
    return (pts - lo) / (scale if scale > 0 else 1.0)


def gen_geometry(n, regime, rng, C=(2, 8), sig=(0.04, 0.16), jit=(0.35, 0.75),
                 hc=(0.55, 1.05), dpP=(2, 4), dpconc=(0.4, 1.5), dpjit=(0.10, 0.55)):
    m = n + 1  # depot + n locs, depot shares geometry (OPLib convention)
    if regime == "uniform":
        pts = rng.uniform(0, 1, (m, 2))
    elif regime == "gmm":
        c = rng.integers(C[0], C[1] + 1)
        centers = rng.uniform(0, 1, (c, 2))
        s = float(np.exp(rng.uniform(np.log(sig[0]), np.log(sig[1]))))  # log-uniform sigma
        pts = centers[rng.integers(0, c, m)] + rng.normal(0, s, (m, 2))
        pts = np.clip(pts, 0, 1)
    elif regime == "hardcore":
        # Poisson-disk (dart-throwing): min pairwise separation dmin = coef/sqrt(m).
        # coef small -> weak exclusion (R~1.05, VMR~1); coef large -> near-saturation
        # (R~1.3-1.4, VMR~0.9). Coarse structure stays random => VMR~1 across the range.
        coef = float(rng.uniform(hc[0], hc[1]))
        d2 = (coef / np.sqrt(m)) ** 2
        pts = np.empty((m, 2))
        k, tries, maxtries = 0, 0, m * 60
        while k < m and tries < maxtries:
            p = rng.uniform(0, 1, 2)
            tries += 1
            if k == 0 or ((pts[:k] - p) ** 2).sum(1).min() >= d2:
                pts[k] = p
                k += 1
        while k < m:  # saturated before filling m: top up uniformly (rare, coef high)
            pts[k] = rng.uniform(0, 1, 2)
            k += 1
    elif regime == "density_partition":
        # density-partition regime: fills the R>1 & VMR>1.66 quadrant the other 3 regimes
        # STRUCTURALLY cannot (hardcore R>1=>VMR<1; GMM R<1=>VMR>1). PxP box tiling with
        # UNEQUAL per-box counts (Dirichlet, small conc => coarse density gradient => VMR>1)
        # and a LOCAL jittered mini-grid inside each box (NN-regular => R>1). NOT hardcore-in-
        # a-blob (concentration would pull R<1) -- the structure tiles space, only density varies.
        P = int(rng.integers(dpP[0], dpP[1] + 1))
        conc = float(rng.uniform(dpconc[0], dpconc[1]))
        counts = rng.multinomial(m, rng.dirichlet(np.full(P * P, conc)))
        jfrac = float(rng.uniform(dpjit[0], dpjit[1]))
        out = []
        for b in range(P * P):
            c = int(counts[b])
            if c == 0:
                continue
            bx, by = b // P, b % P
            side = int(np.ceil(np.sqrt(c)))
            g1 = (np.arange(side) + 0.5) / side
            gx, gy = np.meshgrid(g1, g1)
            local = np.stack([gx.ravel(), gy.ravel()], 1)[:c]
            local = local + rng.uniform(-jfrac / side, jfrac / side, local.shape)
            out.append((np.array([bx, by]) + np.clip(local, 0, 1)) / P)
        pts = np.concatenate(out, 0)
    elif regime == "jittered_grid":
        side = int(np.ceil(np.sqrt(m)))
        gx, gy = np.meshgrid(np.linspace(0, 1, side), np.linspace(0, 1, side))
        grid = np.stack([gx.ravel(), gy.ravel()], 1)[:m]
        j = rng.uniform(*jit) / side
        pts = np.clip(grid + rng.uniform(-j, j, grid.shape), 0, 1)
    else:
        raise ValueError(regime)
    return normalize(pts)


# Fixed geometry config (outcome of coverage tuning).
# 4-regime (uniform/gmm/hardcore/density_partition) — dp added to fill the R>1 & VMR>1.66
# quadrant the other 3 STRUCTURALLY cannot. Verified at n=200: at
# fixed-pool scale (M=40000 single draw) ALL 10 R & VMR quantiles inside the fixed tolerances
# (oplib_bracket_geom.txt), binding VMR q50 margin +0.044; joint improved (quadrant ~9% vs target
# ~14% vs 3-regime ~0%; Spearman -0.86 vs target -0.78 vs 3-regime -0.97). See oplib_geom_coverage.txt.
# GMM C=(3,14) x log-U sigma [0.022,0.20]; hardcore coef [0.24,0.48]; dp P=2-4/conc=0.8-2.0/jit=0.10-0.45.
# Mix (regime_pool): pu=0.10 / pc(gmm)=0.54 / ph(hardcore)=0.22 / pdp(density_partition)=0.14.
GMM = dict(C=(3, 14), sig=(0.022, 0.20))
HC = (0.24, 0.48)   # hard-core dmin coefficient range


def measure(n, M, probs, seed=7, jit=(0.20, 0.60), gmm=None, hc=HC):
    """probs = (p_u, p_gmm, p_hardcore, p_grid); routed in that order."""
    from oplib_geometry import clark_evans_R, qs, vmr  # lazy: pulls oplib_parser (coverage only)
    gmm = gmm or GMM
    rng = np.random.default_rng(seed)
    p_u, p_c, p_h, p_g = probs
    Rs, Vs, cnt = [], [], {"uniform": 0, "gmm": 0, "hardcore": 0, "grid": 0}
    for _ in range(M):
        r = rng.random()
        if r < p_u:
            regime, kw, lab = "uniform", {}, "uniform"
        elif r < p_u + p_c:
            regime, kw, lab = "gmm", gmm, "gmm"
        elif r < p_u + p_c + p_h:
            regime, kw, lab = "hardcore", dict(hc=hc), "hardcore"
        else:
            regime, kw, lab = "jittered_grid", dict(jit=jit), "grid"
        cnt[lab] += 1
        pts = gen_geometry(n, regime, rng, **kw)
        Rs.append(clark_evans_R(pts)); Vs.append(vmr(pts))
    qR, qV = qs(Rs), qs(Vs)
    okR = all(abs(qR[q] - TGT_R[q]) <= R_TOL for q in (10, 25, 50, 75, 90))
    okV = all(1 / VMR_TOL[q] <= (qV[q] / TGT_VMR[q]) <= VMR_TOL[q] for q in (10, 25, 50, 75, 90))
    print(f"probs u/c/h/g={probs} counts={cnt} gmm={gmm} hc={hc}")
    print(f"  R   syn {qR}")
    print(f"      tgt {TGT_R}  -> {'PASS' if okR else 'FAIL'} (|dR|<= {R_TOL})")
    print(f"  VMR syn {qV}")
    print(f"      tgt {TGT_VMR}  -> {'PASS' if okV else 'FAIL'} (factor tol)")
    for q in (10, 25, 50, 75, 90):
        print(f"      q{q}: R syn {qR[q]} vs {TGT_R[q]} (d={qR[q]-TGT_R[q]:+.2f}) | "
              f"VMR syn {qV[q]} vs {TGT_VMR[q]} (x{qV[q]/TGT_VMR[q]:.2f})")
    print(f"RESULT {'PASS' if (okR and okV) else 'FAIL'}")
    return okR and okV


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--M", type=int, default=600)
    ap.add_argument("--pu", type=float, default=0.10)   # fixed mix (see GMM/HC above)
    ap.add_argument("--pc", type=float, default=0.57)
    ap.add_argument("--ph", type=float, default=0.33)
    ap.add_argument("--pg", type=float, default=0.0)
    ap.add_argument("--jmin", type=float, default=0.20)
    ap.add_argument("--jmax", type=float, default=0.60)
    ap.add_argument("--smin", type=float, default=None, help="GMM sigma floor override")
    ap.add_argument("--smax", type=float, default=None, help="GMM sigma ceil override")
    ap.add_argument("--cmin", type=int, default=None, help="GMM min clusters override")
    ap.add_argument("--cmax", type=int, default=None, help="GMM max clusters override")
    ap.add_argument("--hcmin", type=float, default=None, help="hard-core coef floor override")
    ap.add_argument("--hcmax", type=float, default=None, help="hard-core coef ceil override")
    a = ap.parse_args()
    if a.measure:
        gmm = dict(GMM)
        if a.smin is not None or a.smax is not None:
            gmm["sig"] = (a.smin if a.smin is not None else GMM["sig"][0],
                          a.smax if a.smax is not None else GMM["sig"][1])
        if a.cmin is not None or a.cmax is not None:
            gmm["C"] = (a.cmin if a.cmin is not None else GMM["C"][0],
                        a.cmax if a.cmax is not None else GMM["C"][1])
        hc = (a.hcmin if a.hcmin is not None else HC[0],
              a.hcmax if a.hcmax is not None else HC[1])
        measure(a.n, a.M, (a.pu, a.pc, a.ph, a.pg), jit=(a.jmin, a.jmax), gmm=gmm, hc=hc)
