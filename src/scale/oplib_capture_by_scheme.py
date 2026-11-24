"""Per-prize-scheme capture on OPLib -> the three values quoted in Section 4.5.

The paper reports capture of -0.059 +- 0.082 (constant prizes), +0.033 +- 0.055
(pseudo-random) and +0.340 +- 0.072 (distance-based). Those are not fields in any
JSON: the artifact holds per-instance quantities and the reported numbers are the
seed-wise aggregation below. This script is that aggregation.

Producing the artifact (one GPU, about four minutes):

    python src/scale/oplib_regime_eval.py \
        --root <OPLIB_ROOT>/instances --no-equiv --n300 \
        --seeds 42,43,44 \
        --sas-fmt  <CHECKPOINT_ROOT>/sas_op50_seed{s}/last.ckpt \
        --pomo-fmt <CHECKPOINT_ROOT>/pomo_op50_seed1234/last.ckpt \
        --out artifacts/scale_ladder/oplib_regime_zeroshot_n300.json

Reading the reported numbers out of it:

    python src/scale/oplib_capture_by_scheme.py

Estimator (identical to conversion() in scale_ladder_aggregate.py): both ends of the
ratio are taken over the SAME fifty-start pool --- the numerator is the head's best
start within that pool (`sas_at_50`), not its unrestricted one-shot pick, and the
denominator is the pool maximum (`best_of_50`). Mixing the two is not merely a
different convention: on three of these instances the unrestricted pick scores above
the pool maximum, which would put the ratio above 1.

    capture(gen, seed) = sum_i [sas_at_50 - uniform_oneshot] / sum_i [best_of_50 - uniform_oneshot]

aggregated as the mean over the three seeds, with two standard errors over the same
three. `--n300` selects the 34 EUC_2D base instances with n <= 300 (102 with the
three prize schemes), the same set as the hit rates and the constructor comparison.
"""
import argparse
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT = os.path.join(ROOT, "artifacts", "scale_ladder",
                       "oplib_regime_zeroshot_n300.json")
GEN = {1: "gen1 constant", 2: "gen2 pseudo-random", 3: "gen3 distance-based"}


def capture(records, gen=None):
    rows = [r for r in records if gen is None or r["gen"] == gen]
    num = sum(r["sas_at_50"] - r["uniform_oneshot"] for r in rows)
    den = sum(r["best_of_50"] - r["uniform_oneshot"] for r in rows)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=DEFAULT)
    a = ap.parse_args()
    if not os.path.exists(a.artifact):
        print(f"artifact not found: {a.artifact}", file=sys.stderr)
        return 1
    d = json.load(open(a.artifact, encoding="utf-8"))
    seeds = sorted(k for k in d if k.startswith("sas_seed"))
    n = len(d[seeds[0]])
    bases = len({r["name"] for r in d[seeds[0]]})
    ns = sorted(r["n"] for r in d[seeds[0]])
    print(f"{n} instances ({bases} bases x 3 prize schemes), n = {ns[0]}..{ns[-1]}, "
          f"{len(seeds)} seeds")
    print(f"fifty-start pool covers {50/ns[-1]*100:.0f}% to "
          f"{min(100.0, 50/ns[0]*100):.0f}% of the nodes\n")
    print(f"{'prize scheme':<24}{'capture':>10}{'2 SE':>9}   per seed")
    for g, label in GEN.items():
        v = [capture(d[s], g) for s in seeds]
        m, se2 = st.mean(v), 2 * st.stdev(v) / len(v) ** 0.5
        flag = "" if abs(m) > se2 else "   (covers zero)"
        print(f"  {label:<22}{m:+10.4f}{se2:>9.3f}   "
              f"{', '.join(f'{x:+.4f}' for x in v)}{flag}")
    v = [capture(d[s]) for s in seeds]
    print(f"  {'pooled':<22}{st.mean(v):+10.4f}"
          f"{2*st.stdev(v)/len(v)**0.5:>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
