"""Budget-sweep calibration aggregation: the calibration checks of the paper's
appendix (density monotonicity, R^2 thresholds, loosest tier, ratio spread,
within-tier Spearman, stakes as secondary predictor). Outputs
budget_sweep_curve.json.
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


def nearopt_path(T):
    if T == 3.0:
        return os.path.join(ARTIFACT_ROOT, "nearopt",
                            "nearopt_starts_op50_first1000.json")
    return os.path.join(ARTIFACT_ROOT, "nearopt", f"nearopt_T{T:g}_first1000.json")


def eval_path(T, s):
    return os.path.join(ARTIFACT_ROOT, "op50_sweep_evals",
                        f"eval_T{T:g}_s{s}.json")


def npz_path(T, s):
    cand = os.path.join(ARTIFACT_ROOT, "op50_sweep_evals",
                        f"eval_T{T:g}_s{s}_perinst.npz")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError((T, s))


stakes = json.load(open(os.path.join(ARTIFACT_ROOT, "stakes",
                                     "stakes_first200_v2.json")))["tiers"]

rows = []
for ti, T in enumerate(TIERS):
    near = json.load(open(nearopt_path(T)))
    S = [set(r["nearopt_start_set"]) for r in near["instances"]]
    ref = float(np.mean([r["best_prize"] for r in near["instances"]]))
    density = float(np.mean([len(s) for s in S]) / 50.0)
    rb, og, hit = [], [], []
    sp_rho, sp_p = [], []
    for s in SEEDS:
        j = json.load(open(eval_path(T, s)))
        rb.append((j["sas_oneshot_mean"] - j["uniform_oneshot_mean"]) / ref * 100)
        og.append((j["oracle_mean"] - j["uniform_oneshot_mean"]) / ref * 100)
        z = np.load(npz_path(T, s))
        hp = z["head_probs"]
        arg = hp.argmax(1)
        hit.append(float(np.mean([int(arg[i]) in S[i] for i in range(len(S))])))
        # within-tier per-instance Spearman (|S_i| vs benefit_i)
        ben_i = z["sas_oneshot"] - z["uniform_oneshot_draws"].mean(0)
        s_i = np.array([len(x) for x in S])
        rho, p = stats.spearmanr(s_i, ben_i / ref * 100)
        sp_rho.append(float(rho)); sp_p.append(float(p))
    rb, og = np.array(rb), np.array(og)
    rows.append(dict(
        T=T, ref=ref, density=density,
        stakes=stakes[str(T)]["stakes_mean"],
        realized_mean=float(rb.mean()), realized_std=float(rb.std(ddof=1)),
        oracle_mean=float(og.mean()), oracle_std=float(og.std(ddof=1)),
        ratio_mean=float((rb / og).mean()),
        hit1_mean=float(np.mean(hit)), hit1_std=float(np.std(hit, ddof=1)),
        within_tier_spearman_rho=sp_rho, within_tier_spearman_p=sp_p,
    ))

print(f"{'T':>4} {'dens':>6} {'stakes':>7} {'realized':>16} {'oracle':>8} "
      f"{'ratio':>6} {'hit@1':>12}")
for r in rows:
    print(f"{r['T']:4g} {r['density']:6.4f} {r['stakes']:7.4f} "
          f"{r['realized_mean']:8.3f}±{r['realized_std']:<6.3f} "
          f"{r['oracle_mean']:8.3f} {r['ratio_mean']:6.3f} "
          f"{r['hit1_mean']:6.3f}±{r['hit1_std']:<5.3f}")

y = np.array([r["realized_mean"] for r in rows])
se = np.array([r["realized_std"] for r in rows]) / np.sqrt(3)
x_dens = np.array([r["density"] for r in rows])
x_stakes = np.array([r["stakes"] for r in rows])

print("\n=== Density monotonicity: y strictly decreasing in density, one adjacent")
print("    tie within 2*SE allowed ===")
order = np.argsort(x_dens)
yd = y[order]
sed = se[order]
ties, violations = [], []
for a in range(4):
    diff = yd[a] - yd[a + 1]  # expect > 0 (decreasing)
    lim = 2 * np.sqrt(sed[a] ** 2 + sed[a + 1] ** 2)
    if diff > lim:
        pass
    elif abs(diff) <= lim:
        ties.append(a)
    else:
        violations.append(a)
print(f"pairwise decreasing: ties={ties} violations={violations}")
density_monotone = "PASS" if (len(violations) == 0 and len(ties) <= 1) else "FAIL"
print("density monotone:", density_monotone)

print("\n=== OLS y~density R2 thresholds 0.85 / 0.60 ===")
sl, ic, r_d, _, _ = stats.linregress(x_dens, y)
r2_d = r_d ** 2
r2_status = ("tight, predictable" if r2_d >= 0.85 else
            "organized spectrum but loose" if r2_d >= 0.60
            else "density alone insufficient")
print(f"R2(density) = {r2_d:.4f} -> {r2_status}")
sl2, ic2, r_s, _, _ = stats.linregress(x_stakes, y)
print(f"[descriptive] R2(stakes) = {r_s**2:.4f}, "
      f"slope {sl2:.2f}, intercept {ic2:.2f}")

print("\n=== Loosest tier realized within +/-0.5pp ===")
loosest = [r for r in rows if r["T"] == 4.5][0]
loosest_within = "PASS" if abs(loosest["realized_mean"]) <= 0.5 else "FAIL"
print(f"T4.5 realized = {loosest['realized_mean']:.3f} pp -> loosest tier within 0.5 pp of zero: {loosest_within}")

print("\n=== Ratio spread (descriptive): ratio std <= 0.15 ===")
ratios = np.array([r["ratio_mean"] for r in rows])
print(f"ratios {np.round(ratios, 3)} std {ratios.std(ddof=1):.4f} -> "
      f"{'approximately constant' if ratios.std(ddof=1) <= 0.15 else 'not constant (reported as measured)'}")

print("\n=== Within-tier Spearman, median rho<0 &")
print("    p<0.01 in >=4/5 tiers ===")
ok = 0
for r in rows:
    med_rho = float(np.median(r["within_tier_spearman_rho"]))
    all_sig = all(p < 0.01 for p in r["within_tier_spearman_p"])
    good = med_rho < 0 and all_sig
    ok += int(good)
    print(f"T{r['T']:g}: median rho {med_rho:+.3f}, "
          f"p {['%.1e' % p for p in r['within_tier_spearman_p']]}, {'OK' if good else 'no'}")
print(f"within-tier Spearman: {ok}/5 tiers -> {'per-instance claim holds' if ok >= 4 else 'per-distribution only'}")

print("\n=== Stakes as secondary predictor: dynamic range & direction ===")
rho_s, p_s = stats.spearmanr(x_stakes, y)
print(f"stakes range {x_stakes.max()/x_stakes.min():.2f}x, "
      f"Spearman(stakes, y) rho={rho_s:.3f} p={p_s:.4f}")
print("stakes predictor:", "wide range, monotone" if (x_stakes.max() / x_stakes.min() > 1.5 and rho_s > 0.9)
      else "mixed")

os.makedirs(OUTPUT_ROOT, exist_ok=True)
json.dump(dict(rows=rows, density_monotone=density_monotone, r2_density=float(r2_d),
               r2_stakes=float(r_s**2), loosest_tier_within_half_point=loosest_within,
               ratio_std=float(ratios.std(ddof=1)), within_tier_spearman_ok_tiers=ok,
               spearman_stakes_y=[float(rho_s), float(p_s)]),
          open(os.path.join(OUTPUT_ROOT, "budget_sweep_curve.json"), "w"), indent=2)
print("\nsaved", os.path.join(OUTPUT_ROOT, "budget_sweep_curve.json"))
