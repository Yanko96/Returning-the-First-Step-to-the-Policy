"""Head x near-opt cross-analysis: does the head hit the near-optimal
start set on OPLib? (presentation only)"""
import glob
import json
import os

import numpy as np

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
DIAG = os.path.join(ARTIFACT_ROOT, "head_diag")

near = json.load(open(os.path.join(ARTIFACT_ROOT, "nearopt",
                                   "oplib_nearopt.json")))
S = {r["name"]: set(r["nearopt_start_set"]) for r in near["instances"]}
gen_of = {r["name"]: r["gen"] for r in near["instances"]}
n_of = {r["name"]: r["n"] for r in near["instances"]}

chance = float(np.mean([len(S[nm]) / n_of[nm] for nm in S]))
print(f"near-opt sets: {len(S)} instances (n<=300); start-set size mean "
      f"{np.mean([len(s) for s in S.values()]):.2f}; chance hit@1 = {chance:.2%}")

heads = {}
for f in sorted(glob.glob(os.path.join(DIAG, "oplib_head_diag_seed*.json"))):
    seed = f[-7:-5]
    heads[seed] = {m["name"]: m for m in json.load(open(f))}

print("\n== hit@1 (head argmax in near-opt start set), n<=300 subset ==")
per_seed_hits = {}
for seed, H in heads.items():
    hits = np.array([H[nm]["argmax"] in S[nm] for nm in S])
    conf_miss = np.array([(H[nm]["top1"] > 0.9) and (H[nm]["argmax"] not in S[nm]) for nm in S])
    per_seed_hits[seed] = hits
    print(f"seed {seed}: hit@1 {hits.mean():.1%} | confident-miss (top1>0.9 & miss) "
          f"{conf_miss.mean():.1%} | mean top1 {np.mean([H[nm]['top1'] for nm in S]):.3f}")
hm = [per_seed_hits[s].mean() for s in sorted(per_seed_hits)]
print(f"3-seed hit@1: {np.mean(hm):.1%} ± {np.std(hm, ddof=1):.1%}  "
      f"(OP50 in-distribution: 52.8% ± 5.6% on the first 1000, chance 4.4%)")

print("\nby gen (3-seed mean hit@1):")
for g in (1, 2, 3):
    nms = [nm for nm in S if gen_of[nm] == g]
    v = [np.mean([heads[s][nm]["argmax"] in S[nm] for nm in nms]) for s in sorted(heads)]
    print(f"  gen{g}: {np.mean(v):.1%} ± {np.std(v, ddof=1):.1%}  ({len(nms)} inst)")

print("\n== cross-seed agreement of the argmax start (seed 42 against 43 and 44) ==")
# Top-10 sets are not used, for coverage of the near-optimal set or across seeds. The head is near
# one-hot, so its float32 softmax leaves most of a top-10 set at exactly zero probability, and the
# set would depend on argsort's tie-breaking.
names_all = [m["name"] for m in json.load(open(os.path.join(DIAG, "oplib_head_diag_seed42.json")))]
small = [nm for nm in names_all if heads["42"][nm]["n"] <= 300]
OP50 = os.path.join(ARTIFACT_ROOT, "main_evals", "op50")
am50 = {s: np.load(os.path.join(OP50, f"sas_op50_seed{s}_eval_perinst.npz"))["head_probs_first200"].argmax(1)
        for s in ("42", "43", "44")}
for a, b in (("42", "43"), ("42", "44")):
    def agree(nms):
        return np.mean([heads[a][nm]["argmax"] == heads[b][nm]["argmax"] for nm in nms])
    print(f"seed{a} vs seed{b}: OP50 first 200 {np.mean(am50[a] == am50[b]):.3f} | "
          f"OPLib all {len(names_all)} {agree(names_all):.3f} | n<=300 ({len(small)}) {agree(small):.3f}")

print("\n== IPR / top1 across all 210 (collapse check) ==")
for seed, H in heads.items():
    print(f"seed {seed}: mean IPR {np.mean([m['ipr'] for m in H.values()]):.3f}, "
          f"mean top1 {np.mean([m['top1'] for m in H.values()]):.4f}")
