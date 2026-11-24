"""Print the start-stakes-across-scales numbers of Appendix H from the shipped artifact.

    python src/calibration/stakes_across_scales_report.py

Reads artifacts/stakes/stakes_across_scales.json (the three series, which
src/calibration/stakes_across_scales.py reproduces with the seeds it documents). For each
solver configuration it prints the mean stakes at n = 50, 100, 200, 500 with standard
errors, the ratio of the n = 50 to the n = 500 value, and a log-log slope over the four
scales (descriptive). It first recomputes every stored mean and standard error from the
shipped per-instance values and stops if one does not reproduce.
"""
import json
import os

import numpy as np

ARTIFACT_ROOT = os.environ.get("ARTIFACT_ROOT", "artifacts")
PATH = os.path.join(ARTIFACT_ROOT, "stakes", "stakes_across_scales.json")
SCALES = ["50", "100", "200", "500"]


def main():
    with open(PATH) as f:
        art = json.load(f)
    print(art["definition"], "\n")
    for name in ("as_defined", "light_both", "reference_both"):
        v = art["variants"][name]
        sc = v["scales"]
        y = np.array([sc[n]["stakes_mean"] for n in SCALES])
        se = np.array([sc[n]["stakes_se"] for n in SCALES])
        for n in SCALES:
            vals = np.array(list(sc[n]["per_instance"].values()))
            assert len(vals) == sc[n]["n_inst"]
            assert abs(vals.mean() - sc[n]["stakes_mean"]) < 1e-12, (name, n)
            assert abs(vals.std(ddof=1) / np.sqrt(len(vals)) - sc[n]["stakes_se"]) < 1e-12
            if name == "as_defined":
                q = np.array(list(sc[n]["per_instance_light_free_over_reference"].values()))
                assert abs(q.mean() - sc[n]["light_free_over_reference"]) < 1e-12, (name, n)
        ns = np.array([int(n) for n in SCALES], float)
        slope, icpt = np.polyfit(np.log(ns), np.log(y), 1)
        fit = icpt + slope * np.log(ns)
        r2 = 1 - ((np.log(y) - fit) ** 2).sum() / ((np.log(y) - np.log(y).mean()) ** 2).sum()
        print("%-15s forced: %s; reference: %s" % (name, v["forced"], v["reference"]))
        print("   stakes at n = 50/100/200/500: %s  (SE %s)"
              % (" / ".join("%.3f" % x for x in y), " / ".join("%.4f" % x for x in se)))
        print("   n=50 over n=500: %.2f   log-log slope over four scales: %+.3f (R^2 %.3f)"
              % (y[0] / y[-1], slope, r2))
        if name == "as_defined":
            q = [sc[n]["light_free_over_reference"] for n in SCALES]
            print("   light solver, unconstrained, over the reference: %s"
                  % " / ".join("%.3f" % x for x in q))
        print()


if __name__ == "__main__":
    main()
