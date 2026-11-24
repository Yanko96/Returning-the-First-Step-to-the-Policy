"""Assemble the three OOD/cross-problem calibration points.
The OP50 fit is computed once; these points are overlays.

PCTSP50 (cross-problem, same caliber as the budget-sweep calibration):
  realized = (uniform_oneshot_cost - sas_oneshot_cost) / ILS_ref * 100
  oracle   = (uniform_oneshot_cost - oracle_cost) / ILS_ref * 100
  computed on the FIRST 200 instances (the ILS reference's coverage),
  same checkpoint, 8 fixed uniform draws, 3 seeds. Secondary: the same
  quantities over all 1000 eval instances against the OR-Tools 4.48
  reference, reported for context.
  x = stakes_pctsp50.json stakes_mean.

OP100 / OPLib (distribution shift): y is the zero-shot one-shot advantage of
SAS over the separately trained uniform configuration, paired by seed over the
five seeds the two share (42-46), mean +- std, in gap points (the paper's decay
table: OP100 from shift_evals/op100_*_eval.json, OPLib from the 210-instance
mean gap in shift_evals/oplib_*_eval.json). This caliber differs from the
same-checkpoint one of the budget sweep, which is recorded in the output and
stated in the paper.
  x = stakes_op100.json / stakes_oplib.json stakes_mean.

Also: oracle persistence on OP100/OPLib (SAS, five seeds): one-shot gap minus
full-multistart gap = how much start value the oracle still finds
off-distribution (OPLib on the n <= 300 subset the stakes cover).
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
SHIFT_EVALS = os.path.join(ARTIFACT_ROOT, "shift_evals")

ils_ref = json.load(open(os.path.join(ARTIFACT_ROOT, "nearopt",
                                      "pctsp_nearopt.json")))
ref_mean = float(np.mean([r["best_obj"] for r in ils_ref["instances"]]))  # first 200

rows = []
for s in (42, 43, 44):
    z = np.load(os.path.join(ARTIFACT_ROOT, "pctsp_point",
                             f"pctsp_cal_s{s}_perinst.npz"))
    sas_c = -z["sas_oneshot"]          # cost = -reward
    uni_c = -z["uniform_oneshot_draws"].mean(0)
    ora_c = -z["oracle_beststart"]
    r200 = dict(
        realized=float((uni_c[:200] - sas_c[:200]).mean() / ref_mean * 100),
        oracle=float((uni_c[:200] - ora_c[:200]).mean() / ref_mean * 100))
    r1000 = dict(
        realized=float((uni_c - sas_c).mean() / 4.48 * 100),
        oracle=float((uni_c - ora_c).mean() / 4.48 * 100))
    rows.append(dict(seed=s, first200_ref=r200, all1000_ortools=r1000))
    print(f"s{s}: first200 realized {r200['realized']:.2f} oracle {r200['oracle']:.2f} "
          f"| all1000 realized {r1000['realized']:.2f} oracle {r1000['oracle']:.2f}")

rb = np.array([r["first200_ref"]["realized"] for r in rows])
og = np.array([r["first200_ref"]["oracle"] for r in rows])
STAKES = os.path.join(ARTIFACT_ROOT, "stakes")
stakes_p = json.load(open(os.path.join(STAKES, "stakes_pctsp50.json")))
stakes_100 = json.load(open(os.path.join(STAKES, "stakes_op100.json")))
stakes_lib = json.load(open(os.path.join(STAKES, "stakes_oplib.json")))

pctsp_point = dict(
    problem="PCTSP50", x_stakes=stakes_p["stakes_mean"],
    realized_mean=float(rb.mean()), realized_std=float(rb.std(ddof=1)),
    oracle_mean=float(og.mean()), oracle_std=float(og.std(ddof=1)),
    ratio_mean=float((rb / og).mean()),
    caliber="same-checkpoint, ILS ref, first 200",
    per_seed=rows, ref_mean=ref_mean)

# ---- shifted points and oracle persistence (five-seed zero-shot evaluations) ----
SEEDS = (42, 43, 44, 45, 46)


def op100_points():
    ev = {(a, s): json.load(open(os.path.join(SHIFT_EVALS, f"op100_{a}_seed{s}_eval.json")))["schemes"]
          for a in ("sas", "uniform") for s in SEEDS}
    adv = np.array([ev[("uniform", s)]["greedy_head"]["gap_pct"] - ev[("sas", s)]["greedy_head"]["gap_pct"]
                    for s in SEEDS])
    one = float(np.mean([ev[("sas", s)]["greedy_head"]["gap_pct"] for s in SEEDS]))
    ms = float(np.mean([ev[("sas", s)]["multistart"]["gap_pct"] for s in SEEDS]))
    return adv, one, ms


def oplib_points():
    ev = {(a, s): json.load(open(os.path.join(SHIFT_EVALS, f"oplib_{a}_seed{s}_eval.json")))
          for a in ("sas", "uniform") for s in SEEDS}
    head = lambda a, s: ev[(a, s)]["aggregates"]["greedy_head"]["overall"]["mean_gap"]
    adv = np.array([head("uniform", s) - head("sas", s) for s in SEEDS])
    one, ms = [], []
    for s in SEEDS:
        sub = [r for r in ev[("sas", s)]["instances"] if r["n"] <= 300]
        one.append(np.mean([r["schemes"]["greedy_head"]["gap_pct"] for r in sub]))
        ms.append(np.mean([r["schemes"]["ms50"]["gap_pct"] for r in sub]))
    return adv, float(np.mean(one)), float(np.mean(ms))


a100, o1, om = op100_points()
alib, l1, lm = oplib_points()
CALIBER = "paper decay table, five seeds (separately trained uniform configuration)"

out = dict(
    pctsp50=pctsp_point,
    op100=dict(problem="OP100", x_stakes=stakes_100["stakes_mean"],
               realized_mean=float(a100.mean()), realized_std=float(a100.std(ddof=1)),
               caliber=CALIBER,
               oracle_persistence=dict(one_shot_gap=o1, multistart_gap=om,
                                       start_value_pp=o1 - om)),
    oplib=dict(problem="OPLib", x_stakes=stakes_lib["stakes_mean"],
               realized_mean=float(alib.mean()), realized_std=float(alib.std(ddof=1)),
               caliber=CALIBER,
               oracle_persistence=dict(one_shot_gap=l1, ms50_gap=lm,
                                       start_value_pp=l1 - lm,
                                       subset="n<=300")),
    op50_fit=dict(slope=71.77, intercept=-2.62, note="fit computed on the five-tier sweep"),
)
os.makedirs(OUTPUT_ROOT, exist_ok=True)
json.dump(out, open(os.path.join(OUTPUT_ROOT, "ood_points.json"), "w"), indent=2)

print(f"\nPCTSP50 point: x={pctsp_point['x_stakes']:.4f} "
      f"realized {rb.mean():.2f}+/-{rb.std(ddof=1):.2f} "
      f"oracle {og.mean():.2f} ratio {(rb/og).mean():.3f} "
      f"| line predicts {71.77*pctsp_point['x_stakes']-2.62:.2f}")
print(f"OP100 point:   x={stakes_100['stakes_mean']:.4f} "
      f"realized {a100.mean():.3f}+/-{a100.std(ddof=1):.3f} "
      f"| line predicts {71.77*stakes_100['stakes_mean']-2.62:.2f} "
      f"| oracle persistence {o1-om:.2f} pp (one-shot {o1:.2f} -> ms {om:.2f})")
print(f"OPLib point:   x={stakes_lib['stakes_mean']:.4f} "
      f"realized {alib.mean():.3f}+/-{alib.std(ddof=1):.3f} "
      f"| line predicts {71.77*stakes_lib['stakes_mean']-2.62:.2f} "
      f"| oracle persistence {l1-lm:.2f} pp (one-shot {l1:.2f} -> ms50 {lm:.2f})")
print("written", os.path.join(OUTPUT_ROOT, "ood_points.json"))
