"""Aggregate the start-rule / sampling / beam baselines (start_rule_baselines.py) over
the five Table 1 seeds, after checking that the harness reproduces the table.

    python src/sas/aggregate_baselines.py

Inputs: artifacts/main_evals/op50/baselines/ (one JSON + per-instance npz per
checkpoint) and artifacts/main_evals/five_seed/ (the Table 1 evaluations).
Output: outputs/baselines_summary.json (the shipped copy is
artifacts/main_evals/op50/baselines_summary.json).

Consistency checks (nothing is reported if one fails):
  the SAS head at 1/10/25 rollouts equals Table 1's one-shot / multistart 10 / multistart 25 rows;
  the rule at 50 rollouts takes every start, so it equals Table 1's multistart-50 row.
Paired statistics follow the paper: configurations trained with the same seed are
paired; per-seed mean differences, then mean and two standard errors over seeds;
per-seed paired Wilcoxon over the 10,000 instances.
"""
import json
import os
import sys

import numpy as np
from scipy import stats

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
BASE = os.path.join(ARTIFACT_ROOT, "main_evals", "op50", "baselines")
TABLE = os.path.join(ARTIFACT_ROOT, "main_evals", "five_seed")

REF = 16.17
POMO_SEEDS = [1234, 42, 43, 44, 45]
SAS_SEEDS = [42, 43, 44, 45, 46]
SHARED = [42, 43, 44, 45]          # seeds both POMO and SAS were trained with


def gap(prize):
    return (REF - np.asarray(prize, float)) / REF * 100


def load(d, tag):
    j = json.load(open(os.path.join(d, f"{tag}.json")))
    z = np.load(os.path.join(d, f"{tag}_perinst.npz"))
    return j, {k: z[k] for k in z.files}


def main():
    R = {"checks": {}, "schemes": {}, "paired": {}}
    runs = {("pomo", s): load(BASE, f"pomo_seed{s}") for s in POMO_SEEDS}
    runs.update({("sas", s): load(BASE, f"sas_seed{s}") for s in SAS_SEEDS})
    pub = {(a, s): json.load(open(os.path.join(TABLE, f"op50_{a}_seed{s}.json")))
           for a, seeds in (("pomo", POMO_SEEDS), ("sas", SAS_SEEDS)) for s in seeds}

    fails = []
    for s in SAS_SEEDS:
        j, _ = runs[("sas", s)]
        for mine, theirs in (("head1", "greedy_head"), ("head10", "ms10"), ("head25", "ms25")):
            a, b = j["schemes"][mine]["gap_pct"], pub[("sas", s)]["schemes"][theirs]["gap_pct"]
            R["checks"][f"sas{s} {mine} vs Table 1"] = [a, b]
            if abs(a - b) > 1e-3:
                fails.append(f"sas{s} {mine}: {a} vs Table 1 {b}")
    for key in runs:
        j, _ = runs[key]
        a, b = j["schemes"]["rule50"]["gap_pct"], pub[key]["schemes"]["multistart"]["gap_pct"]
        R["checks"][f"{key[0]}{key[1]} rule50 vs Table 1 multistart"] = [a, b]
        if abs(a - b) > 1e-3:
            fails.append(f"{key}: rule50 {a} vs multistart {b}")
    if fails:
        print("consistency check failed, nothing reported:\n  " + "\n  ".join(fails))
        return 1
    print("checks pass: the head reproduces Table 1 and the rule at N=n equals multistart-50")

    # per-scheme mean +- std over seeds (gap of means per seed, as Table 1)
    for arm, seeds in (("pomo", POMO_SEEDS), ("sas", SAS_SEEDS)):
        for name in runs[(arm, seeds[0])][0]["schemes"]:
            g = [runs[(arm, s)][0]["schemes"][name]["gap_pct"] for s in seeds]
            R["schemes"][f"{arm}:{name}"] = dict(mean=float(np.mean(g)), std=float(np.std(g, ddof=1)),
                                                 per_seed=dict(zip(map(str, seeds), g)),
                                                 rollouts=runs[(arm, seeds[0])][0]["schemes"][name]["rollouts"])

    def paired(name, a_of, b_of, seeds):
        """b minus a in gap points (positive = a better), per seed and pooled."""
        diffs, wil = [], {}
        for s in seeds:
            d = gap(b_of(s)) - gap(a_of(s))
            diffs.append(float(d.mean()))
            wil[str(s)] = float(stats.wilcoxon(d[d != 0]).pvalue) if np.any(d != 0) else 1.0
        diffs = np.array(diffs)
        R["paired"][name] = dict(mean=float(diffs.mean()),
                                 two_se=float(2 * diffs.std(ddof=1) / np.sqrt(len(diffs))),
                                 per_seed=dict(zip(map(str, seeds), diffs.tolist())),
                                 wilcoxon_p=wil, seeds=seeds)

    P = lambda arm, name: (lambda s: runs[(arm, s)][1][name])
    # learned head vs the rule on the same SAS checkpoint (isolates the selector)
    for k in (1, 10, 25):
        paired(f"rule{k} - head{k} (same SAS ckpt)", P("sas", f"head{k}"), P("sas", f"rule{k}"), SAS_SEEDS)
    # SAS head vs POMO + rule, seed-paired
    for k in (1, 10, 25):
        paired(f"pomo rule{k} - sas head{k}", P("sas", f"head{k}"), P("pomo", f"rule{k}"), SHARED)
    # SAS head vs sampling and beam search on POMO at matched rollouts, seed-paired
    for k in (1, 10, 25):
        paired(f"pomo samp{k} - sas head{k}", P("sas", f"head{k}"), P("pomo", f"samp{k}"), SHARED)
    for k in (10, 25):
        paired(f"pomo beam{k} - sas head{k}", P("sas", f"head{k}"), P("pomo", f"beam{k}"), SHARED)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    out = os.path.join(OUTPUT_ROOT, "baselines_summary.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"\n{'scheme':28s} {'rollouts':>8s} {'gap mean':>9s} {'std':>6s}")
    for k, v in R["schemes"].items():
        print(f"{k:28s} {v['rollouts']:8d} {v['mean']:9.3f} {v['std']:6.3f}")
    print(f"\n{'paired (positive = first named is worse)':48s} {'mean':>7s} {'2SE':>6s}  max p")
    for k, v in R["paired"].items():
        print(f"{k:48s} {v['mean']:7.3f} {v['two_se']:6.3f}  {max(v['wilcoxon_p'].values()):.1e}")
    print("\nwritten", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
