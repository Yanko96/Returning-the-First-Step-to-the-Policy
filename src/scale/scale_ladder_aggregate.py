"""Summarise the scale-ladder evaluations into the numbers the paper reports.

Reads the per-cell JSONs written by scale_ladder_eval.py from an eval_out/ directory,
aggregates them, writes a summary JSON and prints the report.

Estimators (per SAS checkpoint; both start policies decode the same weights):
  realized benefit(i) = sas_oneshot(i) - uniform_oneshot(i)
  capture(cell)       = (sas_at_50 - unif) / (best_of_50 - unif)
POMO one-shot is an independently trained baseline, reported alongside rather
than used for the within-checkpoint realized benefit.

Conventions:
  - the largest scale reported is the highest one at which every scale up to it
    clears its threshold; any non-monotonicity is singled out rather than smoothed;
  - per-scale completed-seed counts are shown, and a scale with fewer than three
    seeds is reported as a trend only;
  - with three seeds the two-standard-error interval is a t(2 df) interval, not a
    95 % one; the threshold is a literal 2 SE;
  - capture is pooled over (scale, seed) cells with equal weight, and the per-cell
    range is reported alongside the pooled value;
  - an effect smaller than 2 SE is reported as inconclusive rather than as zero;
  - no multiplicity correction is applied and every quantity is reported;
  - missing or failed runs are reported as the seed count actually obtained, with
    no substitution.
"""
import argparse
import glob
import json
import os

import numpy as np

SCALES = [100, 200, 500]          # main ladder
SEEDS = [42, 43, 44]
SEEDS_REGIME = [42, 43, 44, 45, 46]
# With three seeds the "2*SE" interval is a t(2 df) interval, not a 95% one; the
# t(2 df) value is reported alongside and the >2*SE threshold is literal.
from scipy import stats  # noqa: E402


def load(evdir, tag):
    p = os.path.join(evdir, f"{tag}.json")
    if not os.path.exists(p):
        return None
    j = json.load(open(p))
    z = np.load(os.path.join(evdir, f"{tag}_perinst.npz"))
    return j, {k: z[k] for k in z.files}


def realized_benefit_perinst(pi):
    return pi["sas_oneshot"] - pi["uniform_oneshot"]


def conversion_cell(pi):
    num = pi["sas_at_50"].mean() - pi["uniform_oneshot"].mean()
    den = pi["best_of_50"].mean() - pi["uniform_oneshot"].mean()
    return float(num / den) if abs(den) > 1e-9 else float("nan")


def se_band(vals):
    """mean, SE over seeds, and the t(2 df) value for the 2*SE reading."""
    v = np.asarray(vals, float)
    k = len(v)
    m = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(k)) if k >= 2 else float("nan")
    t2 = float(stats.t.ppf(1 - (2 * stats.norm.sf(2)) / 2, df=k - 1)) if k >= 2 else float("nan")
    return m, se, k, t2


def branch(effect, se):
    if not np.isfinite(se):
        return "single-seed"
    if effect > 2 * se:
        return "positive"
    if effect < -2 * se:
        return "negative"
    return "inconclusive"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evdir", default="eval_out")
    ap.add_argument("--out", default="scale_ladder_summary.json")
    args = ap.parse_args()
    V = dict(scale_advantage={}, capture={}, regime_advantage={}, pomo_xval={})

    # ---- scale ladder: realized benefit and capture ----
    rb_by_scale, conv_cells, seedcount = {}, [], {}
    for n in SCALES:
        rbs, convs, present = [], [], []
        for s in SEEDS:
            r = load(args.evdir, f"sas_op{n}_seed{s}")
            if r is None:
                continue
            _, pi = r
            rbs.append(float(realized_benefit_perinst(pi).mean()))
            convs.append(conversion_cell(pi))
            conv_cells.append((n, s, conversion_cell(pi)))
            present.append(s)
        seedcount[n] = len(present)
        if rbs:
            m, se, k, t2 = se_band(rbs)
            rb_by_scale[n] = dict(realized_mean=m, se=se, seeds=present, k=k,
                                  t2df=t2, branch=branch(m, se),
                                  exceeds_2se=bool(np.isfinite(se) and m > 2 * se))

    # largest scale reported = highest one with every scale below it clearing;
    # fewer than three seeds means trend only
    eligible = [n for n in SCALES if seedcount.get(n, 0) >= 3]
    passing_through, nstar = [], None
    for n in sorted(eligible):
        if rb_by_scale.get(n, {}).get("exceeds_2se"):
            passing_through.append(n)
            if passing_through == sorted(eligible[:len(passing_through)]):
                nstar = n
        else:
            break
    non_monotone = [n for n in eligible
                    if rb_by_scale.get(n, {}).get("exceeds_2se") and
                    (nstar is None or n > nstar)]      # passed above a break
    V["scale_advantage"] = dict(seed_counts=seedcount, per_scale=rb_by_scale, n_star=nstar,
                                trend_only=[n for n in SCALES if 0 < seedcount.get(n, 0) < 3],
                                non_monotone_singled=non_monotone,
                                statement=(f"benefit persists through trained scale n={nstar}"
                                           if nstar else "no through-scale established"))

    # capture: pool by (scale, seed) with equal weight
    cvals = [c for (_, _, c) in conv_cells if np.isfinite(c)]
    pooled = float(np.mean(cvals)) if cvals else float("nan")
    V["capture"] = dict(pooled_equal_weight=pooled, n_cells=len(cvals),
                        per_cell=[dict(scale=n, seed=s, conversion=c) for (n, s, c) in conv_cells],
                        cell_range=[float(np.min(cvals)), float(np.max(cvals))] if cvals else None)


    # ---- Geometry/prize regimes: advantage vs 2*SE over five seeds. The main layer
    # is the holdout's distance-prize layer; the other two are descriptive.
    # Per-instance layering uses the 'scheme' field (0/1/2 = constant / pseudo-random
    # / distance prize) emitted by the holdout eval.
    GEN = {0: "gen1_const", 1: "gen2_unif", 2: "gen3_dist"}
    def layer_means(scheme_filter):
        vals = []
        for s in SEEDS_REGIME:
            r = load(args.evdir, f"sas_regime_seed{s}")
            if r is None:
                continue
            _, pi = r
            rb = realized_benefit_perinst(pi)
            if scheme_filter is None:
                vals.append(float(rb.mean()))
            elif "scheme" in pi:
                mask = pi["scheme"] == scheme_filter
                if mask.any():
                    vals.append(float(rb[mask].mean()))
        return vals

    def regime_block(vals):
        if not vals:
            return None
        m, se, k, t2 = se_band(vals)
        ci95 = 1.96 * (np.std(vals, ddof=1) / np.sqrt(k)) if k >= 2 else float("nan")
        return dict(advantage_mean=m, se=se, seeds_present=len(vals), t2df=t2,
                    ci95=float(ci95), branch=branch(m, se),
                    exceeds_2se=bool(np.isfinite(se) and abs(m) > 2 * se))

    regime_pooled = regime_block(layer_means(None))
    if regime_pooled is None:
        V["regime_advantage"] = dict(status="no regime-study eval outputs found")
    else:
        per_gen = {name: regime_block(layer_means(code)) for code, name in GEN.items()}
        main = per_gen.get("gen3_dist")
        V["regime_advantage"] = dict(main_layer="regime holdout, distance-prize layer",
                                     main=main if main else "gen3 layer absent (no 'scheme' in npz); pooled shown as fallback",
                                     pooled_all_layers=regime_pooled,
                                     per_gen_layer=per_gen,
                                     note="main layer: distance-prize; others descriptive")
    json.dump(V, open(args.out, "w"), indent=2)

    # ---- report ----
    print("=== per-scale completed seeds ===", V["scale_advantage"]["seed_counts"])
    print("\n=== realized benefit vs 2*SE, by scale ===")
    for n in SCALES:
        d = rb_by_scale.get(n)
        if d:
            print(f"  n={n}: realized={d['realized_mean']:+.5f} SE={d['se']:.5f} "
                  f"seeds={d['seeds']} branch={d['branch']} exceeds_2se={d['exceeds_2se']}")
    print(f"  n* (through) = {nstar}  non-monotone={non_monotone}  -> {V['scale_advantage']['statement']}")
    print("\n=== capture, pooled over (scale, seed) cells ===")
    print(f"  pooled={pooled:.4f}  "
          f"cells={len(cvals)}  cell_range={V['capture']['cell_range']}")
    print("\n=== geometry/prize regimes: advantage vs 2*SE, 5 seeds ===")
    print(f"  {V['regime_advantage']}")
    print("\nsaved", args.out)


if __name__ == "__main__":
    main()
