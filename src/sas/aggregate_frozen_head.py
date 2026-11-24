"""Head fitted to a frozen POMO checkpoint (frozen_head_fit.py), against SAS and the rule.

    python src/sas/aggregate_frozen_head.py

Inputs: artifacts/main_evals/op50/frozen_head/ (op_eval.py evaluations of the fitted
models + the fit logs), artifacts/main_evals/op50/baselines/ (rule and SAS head on the
same instances) and artifacts/main_evals/five_seed/ (Table 1).
Output: outputs/frozen_head_summary.json (the shipped copy is
artifacts/main_evals/op50/frozen_head_summary.json).

Consistency check: at N = n every start is taken and the backbone is the POMO
backbone unchanged, so the fitted model's multistart-50 row must equal the POMO
checkpoint's own Table 1 row.
Comparisons:
  frozen head vs the rule on the same POMO backbone (same checkpoint, paired per instance)
  frozen head vs SAS, seed-paired on the shared seeds 42-45 (different models, same instances)
"""
import json
import os
import sys

import numpy as np
from scipy import stats

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
OP50 = os.path.join(ARTIFACT_ROOT, "main_evals", "op50")
FROZEN = os.path.join(OP50, "frozen_head")
BASE = os.path.join(OP50, "baselines")
TABLE = os.path.join(ARTIFACT_ROOT, "main_evals", "five_seed")

REF = 16.17
POMO_SEEDS = [1234, 42, 43, 44, 45]
SHARED = [42, 43, 44, 45]
SCHEMES = (("greedy_head", "head1", 1), ("ms10", "head10", 10), ("ms25", "head25", 25))


def main():
    gap = lambda p: (REF - np.asarray(p, float)) / REF * 100
    fj = {s: json.load(open(os.path.join(FROZEN, f"eval_frozen_pomo_seed{s}.json"))) for s in POMO_SEEDS}
    fz = {s: np.load(os.path.join(FROZEN, f"eval_frozen_pomo_seed{s}_perinst.npz")) for s in POMO_SEEDS}
    fit = {s: json.load(open(os.path.join(FROZEN, f"frozen_pomo_seed{s}_fit.json"))) for s in POMO_SEEDS}
    pub = {s: json.load(open(os.path.join(TABLE, f"op50_pomo_seed{s}.json"))) for s in POMO_SEEDS}
    bz = {("pomo", s): np.load(os.path.join(BASE, f"pomo_seed{s}_perinst.npz")) for s in POMO_SEEDS}
    bz.update({("sas", s): np.load(os.path.join(BASE, f"sas_seed{s}_perinst.npz")) for s in SHARED})
    R = dict(check={}, frozen={}, paired={}, curves={})

    bad = []
    for s in POMO_SEEDS:
        a, b = fj[s]["schemes"]["multistart"]["gap_pct"], pub[s]["schemes"]["multistart"]["gap_pct"]
        R["check"][f"pomo{s} multistart-50"] = [a, b]
        if abs(a - b) > 1e-3:
            bad.append(f"seed {s}: frozen-head multistart {a} vs POMO Table 1 {b}")
    if bad:
        print("consistency check failed:\n  " + "\n  ".join(bad))
        return 1
    print("check passes: frozen-head multistart-50 == POMO's Table 1 row on all five seeds")

    for eval_name, base_name, k in SCHEMES:
        g = [fj[s]["schemes"][eval_name]["gap_pct"] for s in POMO_SEEDS]
        R["frozen"][base_name] = dict(rollouts=k, mean=float(np.mean(g)), std=float(np.std(g, ddof=1)),
                                      per_seed=dict(zip(map(str, POMO_SEEDS), g)))

    def paired(name, a_of, b_of, seeds):
        diffs, wil = [], {}
        for s in seeds:
            d = gap(b_of(s)) - gap(a_of(s))
            diffs.append(float(d.mean()))
            wil[str(s)] = float(stats.wilcoxon(d[d != 0]).pvalue) if np.any(d != 0) else 1.0
        diffs = np.array(diffs)
        R["paired"][name] = dict(mean=float(diffs.mean()),
                                 two_se=float(2 * diffs.std(ddof=1) / np.sqrt(len(diffs))),
                                 per_seed=dict(zip(map(str, seeds), diffs.tolist())), wilcoxon_p=wil)

    for eval_name, base_name, k in SCHEMES:
        paired(f"pomo rule{k} - frozen head{k} (same POMO ckpt)",
               lambda s, n=eval_name: fz[s][n], lambda s, kk=k: bz[("pomo", s)][f"rule{kk}"], POMO_SEEDS)
        paired(f"frozen head{k} - sas head{k} (seed-paired)",
               lambda s, g=base_name: bz[("sas", s)][g], lambda s, n=eval_name: fz[s][n], SHARED)

    for s in POMO_SEEDS:
        R["curves"][str(s)] = [round(c["val_oneshot_prize"], 4) for c in fit[s]["curve"]]

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    out = os.path.join(OUTPUT_ROOT, "frozen_head_summary.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"\n{'frozen POMO + fitted head':28s} {'mean':>7s} {'std':>6s}   per seed")
    for k, v in R["frozen"].items():
        print(f"{k:28s} {v['mean']:7.3f} {v['std']:6.3f}   {list(v['per_seed'].values())}")
    print(f"\n{'paired (positive = first named is worse)':52s} {'mean':>7s} {'2SE':>6s}  max p")
    for k, v in R["paired"].items():
        print(f"{k:52s} {v['mean']:7.3f} {v['two_se']:6.3f}  {max(v['wilcoxon_p'].values()):.1e}")
    print("\nvalidation one-shot prize by epoch (0 = initial head):")
    for s, c in R["curves"].items():
        print(f"  seed {s:>4s}: {c}")
    print("\nwritten", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
