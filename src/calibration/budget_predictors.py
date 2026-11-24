"""Start-stakes against budget-only predictors of the realized one-shot benefit
(budget sweep, the leave-one-budget-out table of Section 4.3), and the start-stakes
recomputed with the light solver on both sides.

    python src/calibration/budget_predictors.py

Inputs: artifacts/budget_sweep_curve.json (realized benefit and oracle start value
per budget), artifacts/stakes/stakes_first200_v2.json (the paper's stakes: light
forced-start solver against the 16-restart reference) and
artifacts/stakes/stakes_first200.json (light solver on both sides).
Output: outputs/budget_predictors.json (the shipped copy is artifacts/budget_predictors.json).

Check: the stakes rows must reproduce the leave-one-budget-out predictions of the
paper's table (14.75 / 15.12 / 9.91 / 5.32 / 2.54, RMSE 1.80).
"""
import json
import os

import numpy as np

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
curve = json.load(open(os.path.join(ARTIFACT_ROOT, "budget_sweep_curve.json")))["rows"]
light = json.load(open(os.path.join(ARTIFACT_ROOT, "stakes", "stakes_first200.json")))["tiers"]
paper = json.load(open(os.path.join(ARTIFACT_ROOT, "stakes", "stakes_first200_v2.json")))["tiers"]

T = np.array([r["T"] for r in curve], float)
y = np.array([r["realized_mean"] for r in curve])
oracle = np.array([r["oracle_mean"] for r in curve])
s_paper = np.array([paper[f"{t:.1f}"]["stakes_mean"] for t in T])
s_light = np.array([light[f"{t:.1f}"]["stakes_mean"] for t in T])
assert np.allclose(s_paper, [r["stakes"] for r in curve]), "stakes files disagree with the sweep"


def lobo(x):
    pred = []
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        pred.append(np.polyval(np.polyfit(x[m], y[m], 1), x[i]))
    pred = np.array(pred)
    return pred, y - pred, float(np.sqrt(((y - pred) ** 2).mean()))


def r2(x):
    p = np.polyval(np.polyfit(x, y, 1), x)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


out = dict(T=T.tolist(), realized=y.tolist(), predictors={})
for name, x in (("stakes", s_paper), ("1/T", 1 / T), ("log T", np.log(T)),
                ("stakes, light solver both sides", s_light)):
    pred, err, rmse = lobo(x)
    out["predictors"][name] = dict(r2=r2(x), lobo_pred=pred.tolist(), lobo_err=err.tolist(),
                                   rmse=rmse, rmse_share_of_range=rmse / (y.max() - y.min()))
    print(f"{name:32s} R2 {r2(x):.3f}  LOBO RMSE {rmse:.2f} ({100 * rmse / (y.max() - y.min()):.1f}% of range)"
          f"  errors {np.round(err, 2)}")
assert np.allclose(np.round(out["predictors"]["stakes"]["lobo_pred"], 2), [14.75, 15.12, 9.91, 5.32, 2.54]), \
    "check failed: the stakes rows do not reproduce the paper's table"
print("check passes: the stakes rows reproduce the leave-one-budget-out table")

out["stakes_points"] = dict(paper=(100 * s_paper).tolist(), light_both=(100 * s_light).tolist(),
                            drop=(100 * (s_paper - s_light)).tolist(),
                            margin_paper_over_oracle=(100 * s_paper - oracle).tolist(),
                            margin_light_over_oracle=(100 * s_light - oracle).tolist(),
                            light_free_over_anchor=[paper[f"{t:.1f}"]["lite_free_over_ref_mean"] for t in T])
for k, v in out["stakes_points"].items():
    print(f"{k:28s} {np.round(v, 2)}")
os.makedirs(OUTPUT_ROOT, exist_ok=True)
json.dump(out, open(os.path.join(OUTPUT_ROOT, "budget_predictors.json"), "w"), indent=1)
print("written", os.path.join(OUTPUT_ROOT, "budget_predictors.json"))
