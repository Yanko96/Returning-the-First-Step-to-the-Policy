"""Fixed-pool generator for the controlled prize-structure pool of Appendix G.

Emits a pre-generated fixed pool as FLOAT32 SHARDED .npy (memmap-friendly), NOT a monolithic
pickle: 5 concurrent trainings memmap-share the OS page cache (not 5x RAM), random access,
version-robust, fast to checksum. Sharding also structurally avoids multi-process memmap
page-sharing corruption (each worker owns its shard files).

Per instance:
  geometry : --geometry mix (fixed regime mix, clustered_op_gen) or uniform.
  prize    : gen1(const=1)/gen2(unif)/gen3(dist-from-depot), equal 1/3 (Kool schemes).
  budget   : max_length = alpha*L, alpha~U[.45,.55], L = NN+2-opt TSP over all n+1 nodes.
Shard files per shard k: coords_{k}.npy (S,n+1,2) f32 [row0=depot], prize_{k}.npy (S,n) f32,
maxlen_{k}.npy (S,) f32, meta_{k}.npz (regime,scheme int8; alpha,L f32).

DISJOINT RNG: namespaces {pool,holdout,uniform} use SeedSequence(entropy=[seed, ns_id]).
The training pool, the same-distribution holdout, and the uniform set therefore draw
from INDEPENDENT streams (spawn tree) with NON-OVERLAPPING seed bases.
"""
import argparse
import glob
import hashlib
import os
import time
from multiprocessing import Pool

import numpy as np

import clustered_op_gen as g  # fixed geometry: GMM, HC, gen_geometry

# Fixed 4-regime mix: the density-partition regime fills the R>1 & VMR>1.66 quadrant. Verified at
# n=200 fixed-pool scale (M=40000): all 10 marginal tolerances hold (binding VMR q50 +0.044);
# joint quadrant ~9% (target ~14%, 3-regime ~0%), Spearman -0.86 (target -0.78).
PU, PC, PH, PDP = 0.10, 0.54, 0.22, 0.14
DP = ((2, 4), (0.8, 2.0), (0.10, 0.45))
PRIZE_SCHEMES = ("const", "unif", "dist")
ALPHA = (0.45, 0.55)
MAX_2OPT_PASSES = 25
NAMESPACES = {"pool": 1, "holdout": 2, "uniform": 3}
REG_CODE = {"uniform": 0, "gmm": 1, "hardcore": 2, "density_partition": 3}


def tsp_len(pts):
    n = len(pts)
    D = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(D, np.inf)
    tour, cur, seen = [0], 0, {0}
    for _ in range(n - 1):
        row = D[cur].copy()
        row[list(seen)] = np.inf
        cur = int(row.argmin()); tour.append(cur); seen.add(cur)
    np.fill_diagonal(D, 0.0)
    improved, passes = True, 0
    while improved and passes < MAX_2OPT_PASSES:
        improved = False; passes += 1
        for i in range(1, n - 1):
            a, b = tour[i - 1], tour[i]
            for j in range(i + 1, n):
                c = tour[j]; d = tour[j + 1] if j + 1 < n else tour[0]
                if D[a, c] + D[b, d] + 1e-12 < D[a, b] + D[c, d]:
                    tour[i:j + 1] = tour[i:j + 1][::-1]; improved = True; b = tour[i]
    idx = np.asarray(tour + [tour[0]])
    return float(D[idx[:-1], idx[1:]].sum())


def make_instance(rng, nloc, geometry):
    if geometry == "uniform":
        regime, kw = "uniform", {}
    else:
        r = rng.random()
        if r < PU:
            regime, kw = "uniform", {}
        elif r < PU + PC:
            regime, kw = "gmm", dict(C=g.GMM["C"], sig=g.GMM["sig"])
        elif r < PU + PC + PH:
            regime, kw = "hardcore", dict(hc=g.HC)
        else:
            regime, kw = "density_partition", dict(dpP=DP[0], dpconc=DP[1], dpjit=DP[2])
    pts = g.gen_geometry(nloc, regime, rng, **kw)          # (n+1,2) normalized; [0]=depot
    depot, loc = pts[0], pts[1:]
    scheme = PRIZE_SCHEMES[rng.integers(0, 3)]
    if scheme == "const":
        prize = np.ones(nloc)
    elif scheme == "unif":
        prize = (1 + rng.integers(0, 100, size=nloc)) / 100.0
    else:
        dd = np.linalg.norm(loc - depot[None, :], axis=-1)
        prize = (1 + (dd / dd.max() * 99).astype(int)) / 100.0
    L = tsp_len(pts)
    alpha = float(rng.uniform(*ALPHA))
    return (pts.astype(np.float32), prize.astype(np.float32), np.float32(alpha * L),
            REG_CODE[regime], PRIZE_SCHEMES.index(scheme), np.float32(alpha), np.float32(L))


def gen_shard(arg):
    k, ss, cnt, nloc, geometry, outdir = arg
    kids = ss.spawn(cnt)
    m = nloc + 1
    co = np.empty((cnt, m, 2), np.float32); pr = np.empty((cnt, nloc), np.float32)
    ml = np.empty(cnt, np.float32); reg = np.empty(cnt, np.int8); sch = np.empty(cnt, np.int8)
    al = np.empty(cnt, np.float32); Ls = np.empty(cnt, np.float32)
    for j, kid in enumerate(kids):
        pts, prize, mlen, rc, sc, a, L = make_instance(np.random.default_rng(kid), nloc, geometry)
        co[j] = pts; pr[j] = prize; ml[j] = mlen; reg[j] = rc; sch[j] = sc; al[j] = a; Ls[j] = L
    np.save(f"{outdir}/coords_{k:04d}.npy", co)
    np.save(f"{outdir}/prize_{k:04d}.npy", pr)
    np.save(f"{outdir}/maxlen_{k:04d}.npy", ml)
    np.savez(f"{outdir}/meta_{k:04d}.npz", regime=reg, scheme=sch, alpha=al, L=Ls)
    return k, cnt, reg, sch, al, Ls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--count", type=int, required=True)
    ap.add_argument("--namespace", choices=list(NAMESPACES), default="pool")
    ap.add_argument("--geometry", choices=["mix", "uniform"], default="mix")
    ap.add_argument("--seed", type=int, default=20260808, help="namespace seed base")
    ap.add_argument("--pool-seed", type=int, default=None, dest="pool_seed",
                    help="per-seed pool: fold training seed k into entropy so pool_k is an "
                         "INDEPENDENT stream (cross-seed std then includes DATA variation, not just "
                         "init/shuffle -> SE comparable to fresh-per-epoch). "
                         "pool_k <-> seed k; SAS_k & POMO_k share pool_k.")
    ap.add_argument("--shard", type=int, default=100000, help="instances per shard")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    nshards = (a.count + a.shard - 1) // a.shard
    entropy = [a.seed, NAMESPACES[a.namespace]]
    if a.pool_seed is not None:                     # per-seed independent pool stream
        entropy.append(a.pool_seed)
    parent = np.random.SeedSequence(entropy=entropy)
    shard_ss = parent.spawn(nshards)
    jobs = [(k, shard_ss[k], min(a.shard, a.count - k * a.shard), a.n, a.geometry, a.outdir)
            for k in range(nshards)]
    t0 = time.time()
    reg_all, sch_all, al_all, L_all, tot = [], [], [], [], 0
    with Pool(min(a.procs, nshards)) as pool:
        for k, cnt, reg, sch, al, Ls in pool.imap_unordered(gen_shard, jobs):
            reg_all.append(reg); sch_all.append(sch); al_all.append(al); L_all.append(Ls); tot += cnt
            print(f"  shard {k} ({cnt}) done, {time.time()-t0:.0f}s", flush=True)
    reg = np.concatenate(reg_all); sch = np.concatenate(sch_all)
    al = np.concatenate(al_all); Ls = np.concatenate(L_all)
    inv = {v: k for k, v in REG_CODE.items()}
    manifest = dict(n=a.n, count=int(tot), nshards=nshards, shard=a.shard, namespace=a.namespace,
                    geometry=a.geometry, seed=a.seed, pool_seed=a.pool_seed,
                    entropy=entropy, ns_id=NAMESPACES[a.namespace],
                    mix=dict(pu=PU, pc=PC, ph=PH, pdp=PDP), dp=DP,
                    regime_frac={inv[c]: round(float((reg == c).mean()), 4) for c in np.unique(reg)},
                    scheme_frac={PRIZE_SCHEMES[c]: round(float((sch == c).mean()), 4) for c in np.unique(sch)},
                    alpha_mean=round(float(al.mean()), 4), L_mean=round(float(Ls.mean()), 4),
                    wall_s=round(time.time() - t0, 1))
    import json
    with open(f"{a.outdir}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print("POOL_MANIFEST " + json.dumps(manifest))
    print(f"  {1000*(time.time()-t0)/max(tot,1):.1f} ms/inst  -> {a.outdir}")


def load_pool(outdir):
    """Memmap-load a sharded pool (read-only). Returns dict of lists of per-shard memmaps +
    a total count. Random access: shard = idx // shard_size (from manifest)."""
    import json
    manifest = json.load(open(f"{outdir}/manifest.json"))
    g_ = lambda p: [np.load(f, mmap_mode="r") for f in sorted(glob.glob(f"{outdir}/{p}_*.npy"))]
    return dict(manifest=manifest, coords=g_("coords"), prize=g_("prize"), maxlen=g_("maxlen"))


if __name__ == "__main__":
    main()
