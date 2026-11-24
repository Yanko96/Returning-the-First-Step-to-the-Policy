"""Tables 1 and 2 and the OP100 zero-shot table, from the five-seed evaluations.

    python src/sas/five_seed_tables.py

Inputs: artifacts/main_evals/five_seed/{op50,pctsp50}_<arm>_seed<s>.json and
artifacts/shift_evals/op100_<arm>_seed<s>_eval.json, written by op_eval.py /
pctsp_eval.py (RUN.md, section 2). A table cell is the mean +- std (ddof=1) over
the seeds of schemes[<scheme>]["gap_pct"]. Seeds: POMO 1234/42/43/44/45 (the
released recipe seed in place of one of the five), uniform and SAS 42-46.

Seed-paired differences pair configurations trained with the same seed (42-46 for
SAS and uniform, 42-45 for SAS and POMO): per-seed difference, then mean and two
standard errors over the seeds.
Output: outputs/five_seed_tables.json.
"""
import json
import os

import numpy as np

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")
FIVE = os.path.join(ARTIFACT_ROOT, "main_evals", "five_seed")
SHIFT = os.path.join(ARTIFACT_ROOT, "shift_evals")

SEEDS = {"pomo": [1234, 42, 43, 44, 45], "uniform": [42, 43, 44, 45, 46], "sas": [42, 43, 44, 45, 46]}
ROWS = [("greedy_head", "one-shot via start distribution (1)"), ("ms10", "multistart 10"),
        ("ms25", "multistart 25"), ("ms50", "multistart 50"), ("multistart", "multistart N=n"),
        ("msaug", "multistart N=n x aug 16"), ("greedy", "one-shot, plain greedy (1)")]


def path(table, arm, s):
    if table == "op100":
        return os.path.join(SHIFT, f"op100_{arm}_seed{s}_eval.json")
    return os.path.join(FIVE, f"{table}_{arm}_seed{s}.json")


def gaps(table, arm):
    return {s: json.load(open(path(table, arm, s)))["schemes"] for s in SEEDS[arm]}


def paired(a, b, scheme, seeds):
    """a minus b in gap points over the shared seeds: mean, two standard errors."""
    d = np.array([a[s][scheme]["gap_pct"] - b[s][scheme]["gap_pct"] for s in seeds])
    return float(d.mean()), float(2 * d.std(ddof=1) / np.sqrt(len(d)))


out = {}
for table, title in (("op50", "Table 1: OP50, gap % to Compass 16.17"),
                     ("pctsp50", "Table 2: PCTSP50, gap % to OR-Tools 4.48"),
                     ("op100", "OP100 zero-shot, gap % to Compass 33.19")):
    G = {arm: gaps(table, arm) for arm in SEEDS}
    print(f"\n== {title} (mean +- std over five seeds per configuration)")
    print(f"{'scheme':38s}" + "".join(f"{a:>18s}" for a in ("pomo", "uniform", "sas")))
    out[table] = {}
    for key, label in ROWS:
        if key not in G["sas"][42]:
            continue
        cells = []
        for arm in ("pomo", "uniform", "sas"):
            v = [G[arm][s][key]["gap_pct"] for s in SEEDS[arm] if key in G[arm][s]]
            if len(v) == len(SEEDS[arm]):
                out[table].setdefault(key, {})[arm] = dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)))
                cells.append(f"{np.mean(v):7.3f} +- {np.std(v, ddof=1):5.3f}")
            else:
                cells.append("---")
        print(f"{label:38s}" + "".join(f"{c:>18s}" for c in cells))

    print("seed-paired differences (gap points; negative = SAS better)")
    for key, label in ROWS:
        if key not in G["sas"][42] or key == "greedy":
            continue
        su = paired(G["sas"], G["uniform"], key, SEEDS["sas"])
        line = f"  {label:36s} SAS - uniform {su[0]:+7.3f} +- {su[1]:.3f}"
        if key != "greedy_head":
            sp = paired(G["sas"], G["pomo"], key, [42, 43, 44, 45])
            line += f"   SAS - POMO {sp[0]:+7.3f} +- {sp[1]:.3f}"
            out[table].setdefault(key, {})["sas_minus_pomo"] = sp
        out[table].setdefault(key, {})["sas_minus_uniform"] = su
        print(line)

os.makedirs(OUTPUT_ROOT, exist_ok=True)
with open(os.path.join(OUTPUT_ROOT, "five_seed_tables.json"), "w") as f:
    json.dump(out, f, indent=1)
print("\nwritten", os.path.join(OUTPUT_ROOT, "five_seed_tables.json"))
