"""Uncertainty + robustness for the OP50 calibration, recomputed from the
shipped artifacts only (no new measurement, no predictor change):

Uncertainty
  - OLS realized~stakes: slope/intercept with 95% t-CI (df=3), R^2;
  - R^2/slope stability: (i) seed bootstrap (resample the 3 seeds per tier
    with replacement, 10k draws), (ii) stakes instance bootstrap (resample
    the 200 per-instance stakes per tier, 10k draws), (iii) both combined,
    (iv) leave-one-tier-out range;
  - stakes sampling error per tier: sd/sqrt(200) from per-instance stakes;
  - conversion ratio: per-(tier,seed) ratios, per-tier mean+-sd, pooled
    mean with 95% t-CI over the 15 (tier,seed) values.

Robustness (to the shared-reference artifact)
  y and stakes both divide by an ILS-reference quantity, so a ratio-correlation
  artifact is conceivable. Rerun the regression with:
  - y_raw   = sas_oneshot - uniform_oneshot in RAW prize units (no
              reference in the numerator or denominator);
  - y_rel   = (sas_oneshot - uniform_oneshot) / uniform_oneshot (per-tier
              means; normalizes by the model's own uniform-start prize,
              no ILS quantity anywhere).
  The headline regression is untouched; these are sensitivity rows.
"""
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
TIERS = [1.5, 2.0, 3.0, 4.0, 4.5]
SEEDS = (42, 43, 44)
RNG = np.random.default_rng(20260712)
NBOOT = 10000


def nearopt_path(T):
    if T == 3.0:
        return os.path.join(ARTIFACT_ROOT, "nearopt",
                            "nearopt_starts_op50_first1000.json")
    return os.path.join(ARTIFACT_ROOT, "nearopt", f"nearopt_T{T:g}_first1000.json")


def eval_path(T, s):
    return os.path.join(ARTIFACT_ROOT, "op50_sweep_evals",
                        f"eval_T{T:g}_s{s}.json" if T != 3.0 else f"eval_T3_s{s}.json")


stakes_v2 = json.load(open(os.path.join(ARTIFACT_ROOT, "stakes",
                                        "stakes_first200_v2.json")))["tiers"]

refs, sas, uni, ora = {}, {}, {}, {}
for T in TIERS:
    near = json.load(open(nearopt_path(T)))
    refs[T] = float(np.mean([r["best_prize"] for r in near["instances"]]))
    sas[T], uni[T], ora[T] = [], [], []
    for s in SEEDS:
        j = json.load(open(eval_path(T, s)))
        sas[T].append(j["sas_oneshot_mean"])
        uni[T].append(j["uniform_oneshot_mean"])
        ora[T].append(j["oracle_mean"])

x = np.array([stakes_v2[str(T)]["stakes_mean"] for T in TIERS])
stakes_pi = {T: np.array(stakes_v2[str(T)]["per_instance_stakes"]) for T in TIERS}

# per-(tier,seed) realized/oracle in pp of tier ref (fixed formula)
rb = np.array([[(sas[T][k] - uni[T][k]) / refs[T] * 100 for k in range(3)] for T in TIERS])
og = np.array([[(ora[T][k] - uni[T][k]) / refs[T] * 100 for k in range(3)] for T in TIERS])
y = rb.mean(1)

out = {}

# ---------- headline OLS with parametric CI ----------
sl, ic, r, _, se_sl = stats.linregress(x, y)
tcrit = stats.t.ppf(0.975, df=3)
n = 5
xm = x.mean()
sxx = ((x - xm) ** 2).sum()
resid = y - (sl * x + ic)
s2 = (resid ** 2).sum() / (n - 2)
se_ic = np.sqrt(s2 * (1 / n + xm ** 2 / sxx))
out["ols"] = dict(slope=sl, intercept=ic, r2=r ** 2,
                  slope_ci95=[sl - tcrit * se_sl, sl + tcrit * se_sl],
                  intercept_ci95=[ic - tcrit * se_ic, ic + tcrit * se_ic])

# stakes sampling error per tier
out["stakes_se"] = {str(T): dict(
    mean=float(stakes_pi[T].mean()),
    sd=float(stakes_pi[T].std(ddof=1)),
    se=float(stakes_pi[T].std(ddof=1) / np.sqrt(len(stakes_pi[T]))),
    n=int(len(stakes_pi[T]))) for T in TIERS}

# bootstrap distributions of R^2 and slope
def boot(resample_seeds, resample_stakes):
    r2s, sls = np.empty(NBOOT), np.empty(NBOOT)
    for b in range(NBOOT):
        yb = np.array([rb[t][RNG.integers(0, 3, 3)].mean() for t in range(5)]) \
            if resample_seeds else y.copy()
        xb = np.array([stakes_pi[T][RNG.integers(0, 200, 200)].mean() for T in TIERS]) \
            if resample_stakes else x.copy()
        s_, _, r_, _, _ = stats.linregress(xb, yb)
        r2s[b], sls[b] = r_ ** 2, s_
    return dict(r2_q025=float(np.quantile(r2s, .025)), r2_q975=float(np.quantile(r2s, .975)),
                r2_median=float(np.median(r2s)),
                slope_q025=float(np.quantile(sls, .025)), slope_q975=float(np.quantile(sls, .975)))

out["boot_seeds"] = boot(True, False)
out["boot_stakes"] = boot(False, True)
out["boot_both"] = boot(True, True)

# leave-one-tier-out
loto = []
for drop in range(5):
    keep = [i for i in range(5) if i != drop]
    s_, i_, r_, _, _ = stats.linregress(x[keep], y[keep])
    loto.append(dict(dropped=TIERS[drop], slope=s_, r2=r_ ** 2))
out["leave_one_out"] = loto
out["loto_r2_range"] = [min(d["r2"] for d in loto), max(d["r2"] for d in loto)]
out["loto_slope_range"] = [min(d["slope"] for d in loto), max(d["slope"] for d in loto)]

# conversion ratio uncertainty
ratios = rb / og  # [5,3]
pooled = ratios.ravel()
out["ratio"] = dict(
    per_tier={str(T): dict(mean=float(ratios[t].mean()), sd=float(ratios[t].std(ddof=1)))
              for t, T in enumerate(TIERS)},
    pooled_mean=float(pooled.mean()), pooled_sd=float(pooled.std(ddof=1)),
    pooled_ci95=[float(pooled.mean() - stats.t.ppf(0.975, 14) * pooled.std(ddof=1) / np.sqrt(15)),
                 float(pooled.mean() + stats.t.ppf(0.975, 14) * pooled.std(ddof=1) / np.sqrt(15))],
    min=float(pooled.min()), max=float(pooled.max()))

# ---------- oracle-on-stakes companion regression ----------
yo = og.mean(1)
so_, io_, ro_, _, _ = stats.linregress(x, yo)
out["oracle_regression"] = dict(slope=float(so_), intercept=float(io_),
                                r2=float(ro_ ** 2),
                                slope_ratio_realized_over_oracle=float(sl / so_))

# ---------- robustness regressions ----------
y_raw = np.array([np.mean([sas[T][k] - uni[T][k] for k in range(3)]) for T in TIERS])
y_rel = np.array([np.mean([(sas[T][k] - uni[T][k]) / uni[T][k] for k in range(3)]) for T in TIERS])
rows = {}
for name, yy in (("headline_pp_of_ref", y), ("raw_prize_units", y_raw),
                 ("relative_to_uniform", y_rel * 100)):
    s_, i_, r_, _, _ = stats.linregress(x, yy)
    rho_, p_ = stats.spearmanr(x, yy)
    rows[name] = dict(slope=float(s_), r2=float(r_ ** 2),
                      spearman=float(rho_), y=[float(v) for v in yy])
out["robustness"] = rows

os.makedirs(OUTPUT_ROOT, exist_ok=True)
json.dump(out, open(os.path.join(OUTPUT_ROOT, "calibration_uncertainty.json"), "w"), indent=2)

print("=== uncertainty ===")
o = out["ols"]
print(f"OLS: slope {o['slope']:.2f} CI95 [{o['slope_ci95'][0]:.2f},{o['slope_ci95'][1]:.2f}], "
      f"intercept {o['intercept']:.2f} CI95 [{o['intercept_ci95'][0]:.2f},{o['intercept_ci95'][1]:.2f}], "
      f"R2 {o['r2']:.4f}")
for k in ("boot_seeds", "boot_stakes", "boot_both"):
    b = out[k]
    print(f"{k}: R2 [{b['r2_q025']:.4f},{b['r2_q975']:.4f}] med {b['r2_median']:.4f}; "
          f"slope [{b['slope_q025']:.2f},{b['slope_q975']:.2f}]")
print(f"LOTO: R2 range {out['loto_r2_range'][0]:.4f}-{out['loto_r2_range'][1]:.4f}, "
      f"slope range {out['loto_slope_range'][0]:.2f}-{out['loto_slope_range'][1]:.2f}")
print("stakes SE per tier:", {k: round(v['se'], 5) for k, v in out["stakes_se"].items()})
r = out["ratio"]
print(f"ratio pooled {r['pooled_mean']:.4f} sd {r['pooled_sd']:.4f} "
      f"CI95 [{r['pooled_ci95'][0]:.4f},{r['pooled_ci95'][1]:.4f}] "
      f"range [{r['min']:.4f},{r['max']:.4f}]")
oreg = out["oracle_regression"]
print(f"oracle~stakes: R2 {oreg['r2']:.4f} slope {oreg['slope']:.2f} "
      f"intercept {oreg['intercept']:.2f} slope-ratio {oreg['slope_ratio_realized_over_oracle']:.3f}")
print("=== robustness ===")
for k, v in rows.items():
    print(f"{k}: R2 {v['r2']:.4f} slope {v['slope']:.3g} spearman {v['spearman']:.2f}")
print("UNCERTAINTY_ANALYSIS_COMPLETE")
