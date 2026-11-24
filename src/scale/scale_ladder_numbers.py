"""Every number Section 4.4 and Appendix H report for the scale ladder, from the
shipped per-instance files.

    python src/scale/scale_ladder_numbers.py

Chain:

    src/scale/scale_ladder_calibrate_budget.py   budget grid per scale (T = 4.0 / 5.5 / 9.0)
    src/scale/scale_ladder_train.py              one run per (scale, seed)
    src/scale/scale_ladder_eval.py               -> artifacts/scale_ladder/eval_out/*.json + *_perinst.npz
    src/scale/scale_ladder_numbers.py            -> the printed numbers            (this file)

Section 4.5 also compares, on the n = 100 held-out set and its fifty-draw pool, the n = 100
models with the section-2 n = 50 checkpoints applied zero-shot (scale_ladder_eval.py with
--ckpt <the section-2 SAS checkpoint of the seed> --budget 4.0 --dataset <the n = 100 held-out
set>, shipped as eval_out/zeroshot_sas_op50_on_op100_seed*.json). It needs only means: kappa50
and the pool start value (mean best_of_50 - mean uniform, prize units) per seed, and the
seed-paired difference of kappa50 with two standard errors over the three seeds.

Per (scale n, seed), on the held-out set of that scale (n = 50: all 10,000 test
instances; n = 100/200/500: 1,000 instances):
  realized  = mean(sas_oneshot - uniform_oneshot)                       prize units
  norm_adv  = realized / mean(uniform_oneshot)
  avail50   = (mean best_of_50 - mean uniform) / mean uniform           fifty-draw pool
  kappa50   = (mean sas_at_50 - mean uniform) / (mean best_of_50 - mean uniform)
  availF    = (mean best_of_full_oracle - mean uniform) / mean uniform  every feasible start
  kappaF    = (mean sas_oneshot - mean uniform) / (mean best_of_full_oracle - mean uniform)
uniform = the mean of eight fixed-seed random feasible starts on the same checkpoint.
At n = 50 the fifty-draw pool is every feasible start, so the two estimators
coincide. The n = 200 oracle over every start comes from the file of the same
model with the fulloracle_ prefix (asserted to be the same model).
"""
import json
import os
import sys

import numpy as np

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
EVDIR = os.path.join(ARTIFACT_ROOT, "scale_ladder", "eval_out")
SCALES, TRAINED, SEEDS = (50, 100, 200, 500), (100, 200, 500), (42, 43, 44)


def load(tag):
    p = os.path.join(EVDIR, f"{tag}_perinst.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


def cell(n, s):
    pi = load(f"sas_op{n}_seed{s}")
    full = pi if "best_of_full_oracle" in pi else load(f"fulloracle_sas_op{n}_seed{s}")
    u = float(pi["uniform_oneshot"].mean())
    c = dict(realized=float((pi["sas_oneshot"] - pi["uniform_oneshot"]).mean()), unif=u)
    c["norm_adv"] = c["realized"] / u
    b = pi["best_of_50"].mean()
    c["avail50"] = float((b - u) / u)
    c["kappa50"] = float((pi["sas_at_50"].mean() - u) / (b - u))
    if n == 50:                       # the fifty-draw pool is every feasible start
        bf = b
    else:
        assert full is not None, f"no all-starts oracle for n={n} seed {s}"
        assert np.allclose(full["sas_oneshot"], pi["sas_oneshot"]), "oracle file is a different model"
        bf = full["best_of_full_oracle"].mean()
    c["availF"] = float((bf - u) / u)
    c["kappaF"] = float((pi["sas_oneshot"].mean() - u) / (bf - u))
    return c


def zeroshot(rows):
    """Section 4.5: the n = 50 checkpoints on the n = 100 set, against the n = 100 models."""
    zk, zv = [], []
    for s in SEEDS:
        p = os.path.join(EVDIR, f"zeroshot_sas_op50_on_op100_seed{s}.json")
        if not os.path.exists(p):
            return None
        sch = json.load(open(p))["schemes"]
        u, b = sch["uniform_oneshot"]["mean"], sch["best_of_50"]["mean"]
        zk.append((sch["sas_at_50"]["mean"] - u) / (b - u))
        zv.append(b - u)
    per = {c["seed"]: c for c in rows[100]["per_seed"]}
    tk = [per[s]["kappa50"] for s in SEEDS]
    tv = [per[s]["avail50"] * per[s]["unif"] for s in SEEDS]
    d = np.array(tk) - np.array(zk)
    return dict(kappa50_zeroshot=float(np.mean(zk)), kappa50_trained=float(np.mean(tk)),
                paired_diff=float(d.mean()), paired_2se=float(2 * d.std(ddof=1) / np.sqrt(len(d))),
                pool_value_zeroshot=float(np.mean(zv)), pool_value_trained=float(np.mean(tv)),
                per_seed=[dict(seed=s, kappa50_zeroshot=float(a), kappa50_trained=float(c))
                          for s, a, c in zip(SEEDS, zk, tk)])


def main():
    rows, cells = {}, []
    for n in SCALES:
        per = [dict(seed=s, **cell(n, s)) for s in SEEDS]
        keys = [k for k in per[0] if k != "seed"]
        m = {k: float(np.mean([c[k] for c in per])) for k in keys}
        m["realized_se"] = float(np.std([c["realized"] for c in per], ddof=1) / np.sqrt(len(per)))
        rows[n] = dict(mean=m, per_seed=per)
        if n in TRAINED:
            cells += [(n, c) for c in per]

    x = np.log(np.array(TRAINED, float))
    y = np.log([rows[n]["mean"]["norm_adv"] for n in TRAINED])
    slope, icpt = np.polyfit(x, y, 1)
    kF = [c["kappaF"] for _, c in cells]
    k50 = [c["kappa50"] for _, c in cells]
    lo, hi = rows[50]["mean"], rows[500]["mean"]
    R = dict(rows={str(n): v for n, v in rows.items()},
             fit_over_trained_scales=dict(exponent=float(slope), multiplier=float(np.exp(icpt))),
             pooled_kappaF=float(np.mean(kF)), kappaF_cell_range=[float(min(kF)), float(max(kF))],
             pooled_kappa50=float(np.mean(k50)),
             decomposition_50_to_500=dict(norm_adv=lo["norm_adv"] / hi["norm_adv"],
                                          availF=lo["availF"] / hi["availF"],
                                          kappaF=lo["kappaF"] / hi["kappaF"],
                                          avail50=lo["avail50"] / hi["avail50"],
                                          kappa50=lo["kappa50"] / hi["kappa50"]))

    print(f"{'n':>4s} {'realized':>9s} {'(se)':>6s} {'norm_adv':>9s} {'avail50':>8s} {'kappa50':>8s} "
          f"{'availF':>8s} {'kappaF':>8s}")
    for n in SCALES:
        m = rows[n]["mean"]
        print(f"{n:4d} {m['realized']:9.3f} {m['realized_se']:6.3f} {m['norm_adv']:9.4f} "
              f"{m['avail50']:8.4f} {m['kappa50']:8.4f} {m['availF']:8.4f} {m['kappaF']:8.4f}")
    print(f"\nlog-log fit of the normalized advantage over the three trained scales: "
          f"exponent {slope:+.3f}, multiplier {np.exp(icpt):.2f}")
    print("  (n = 50 is listed for comparison and is not part of the fit; three points "
          "support no goodness-of-fit statistic)")
    print(f"capture, oracle over every feasible start: pooled {np.mean(kF):.3f} over {len(kF)} cells, "
          f"cells {min(kF):.3f} to {max(kF):.3f}")
    print(f"capture, fifty-draw pool: pooled {np.mean(k50):.3f}")
    d = R["decomposition_50_to_500"]
    print(f"n = 50 -> 500: normalized advantage falls {d['norm_adv']:.2f}x; every-start oracle: "
          f"available value {d['availF']:.2f}x, capture {d['kappaF']:.2f}x; fifty-draw pool: "
          f"{d['avail50']:.2f}x and {d['kappa50']:.2f}x")
    zs = zeroshot(rows)
    if zs is not None:
        R["zeroshot_n50_on_n100"] = zs
        print(f"zero-shot at n = 100, fifty-draw pool: capture {zs['kappa50_zeroshot']:.3f} for the "
              f"n = 50 model against {zs['kappa50_trained']:.3f} for the n = 100 model, paired "
              f"{zs['paired_diff']:+.3f} +- {zs['paired_2se']:.3f} (two standard errors over three seeds); "
              f"pool start value {zs['pool_value_zeroshot']:.2f} against {zs['pool_value_trained']:.2f} "
              f"prize units")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    with open(os.path.join(OUTPUT_ROOT, "scale_ladder_numbers.json"), "w") as f:
        json.dump(R, f, indent=1)
    print("written", os.path.join(OUTPUT_ROOT, "scale_ladder_numbers.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
