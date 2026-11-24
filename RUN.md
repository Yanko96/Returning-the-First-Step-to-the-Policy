# RUN.md — reproducing the paper

All commands are executed from the release root (the directory containing
this file). Paths are controlled by the five environment variables in
`README.md` ("Path configuration"); with none set, inputs are read from
`data/`, `checkpoints/`, `artifacts/`, and everything the scripts produce
lands in `outputs/`. Wall times are rough guides (one mid-range NVIDIA GPU
for the GPU stages; 8-14 CPU workers for the multiprocessing instruments).

Every expected number quoted below is read from the shipped artifacts: the
per-run evaluations under `artifacts/` and the aggregated
`artifacts/budget_sweep_curve.json`, `artifacts/calibration_uncertainty.json`,
`artifacts/budget_predictors.json`, `artifacts/ood_points.json`,
`artifacts/constructor_baseline.json`, `artifacts/constructor_op50_full.json`,
`artifacts/main_evals/op50/baselines_summary.json` and
`artifacts/main_evals/op50/frozen_head_summary.json`.

## 1. Smoke test (CPU only, ~5 minutes)

Byte-compile everything, then run the artifact-backed analyses directly on
the shipped `artifacts/` tree:

```bash
python -m compileall -q src
python src/sas/five_seed_tables.py                  # Tables 1, 2 and the OP100 table
python src/sas/aggregate_baselines.py               # start rule, sampling, beam search
python src/sas/aggregate_frozen_head.py             # head fitted to frozen POMO
python src/calibration/budget_sweep_aggregate.py
python src/calibration/calibration_uncertainty.py   # ~2-3 min (10k-draw bootstraps)
python src/calibration/budget_predictors.py
python src/calibration/ood_points.py
python src/calibration/stakes_v2_recompute.py
python src/scale/scale_ladder_numbers.py
```

Expected:

- `five_seed_tables.py` prints Table 1 (OP50 SAS one-shot `3.769 +- 0.183`,
  POMO plain greedy `11.855 +- 1.198`, uniform one-shot `11.431 +- 0.266`),
  Table 2 (PCTSP50 SAS one-shot `4.732 +- 0.148`) and the OP100 table (SAS
  one-shot `14.046 +- 2.379`, uniform `17.765 +- 1.368`), each cell mean +-
  std over five seeds, followed by the seed-paired differences quoted in
  Appendix A and Section 3.2 (OP50 at ten rollouts, Appendix A: SAS - uniform
  `-1.565 +- 0.026`, SAS - POMO `-1.591 +- 0.036`; at N = n, Section 3.2:
  SAS - POMO `-0.030 +- 0.035`, and `+0.009 +- 0.024` at 800 rollouts).
- `aggregate_baselines.py` first checks that the learned head reproduces
  Table 1 and that the start rule at 50 rollouts equals multistart-50 on
  every checkpoint, then prints the rows: POMO + start rule `4.730 +- 0.147`
  / `1.565 +- 0.082` / `1.265 +- 0.076` at 1 / 10 / 25 rollouts, POMO
  sampling `11.815` / `2.963` / `1.700`, beam search `2.883` / `1.626` at
  10 / 25, and the paired comparisons (rule minus head on the same SAS
  checkpoint `+0.934 +- 0.034`, `+0.296 +- 0.010`, `+0.187 +- 0.003`).
- `aggregate_frozen_head.py` checks that the fitted model's multistart-50 row
  equals POMO's own Table 1 row, then prints the frozen POMO + fitted head
  rows (`3.783 +- 0.043` / `1.256 +- 0.077` / `1.074 +- 0.076`), the
  seed-paired difference to SAS (`+0.027 +- 0.170` at one rollout) and the
  validation curves (flat after the first epoch).
- `budget_sweep_aggregate.py` prints the five-tier curve and writes
  `outputs/budget_sweep_curve.json` matching `artifacts/budget_sweep_curve.json`:

  | T | stakes | realized (pp) | oracle (pp) | ratio | hit@1 |
  |---|---|---|---|---|---|
  | 1.5 | 0.2609 | 17.384 +/- 0.376 | 23.086 | 0.753 | 0.569 |
  | 2.0 | 0.2423 | 14.243 +/- 0.274 | 19.253 | 0.740 | 0.178 |
  | 3.0 | 0.1685 |  7.778 +/- 0.057 | 10.535 | 0.738 | 0.528 |
  | 4.0 | 0.1107 |  5.336 +/- 0.028 |  7.393 | 0.722 | 0.632 |
  | 4.5 | 0.0862 |  4.518 +/- 0.079 |  6.291 | 0.718 | 0.633 |

  and the stakes regression `R2(stakes) = 0.9559, slope 71.77,
  intercept -2.62`.
- `calibration_uncertainty.py` prints `OLS: slope 71.77 CI95 [43.44,100.09]`,
  `R2 0.9559`, and the pooled realized/oracle conversion ratio
  `0.7342 sd 0.0164`; writes `outputs/calibration_uncertainty.json`.
- `budget_predictors.py` checks that the stakes rows reproduce the
  leave-one-budget-out predictions of the paper's table, then prints
  in-sample R^2 and leave-one-budget-out RMSE for each predictor: stakes
  `0.956` / `1.80`, 1/T `0.984` / `1.38`, log T `0.987` / `0.85`, stakes with
  the light solver on both sides `0.976` / `1.29`; and the stakes with one
  solver on both sides (`24.24 / 21.48 / 14.68 / 9.63 / 7.80` points, margin
  over the oracle `1.16 / 2.23 / 4.15 / 2.24 / 1.51`).
- `ood_points.py` prints the PCTSP50 point `realized 7.17+/-0.24`
  (x = 0.0944, oracle 9.61, ratio 0.746), the OP100 point at x = 0.1936
  (realized `3.719+/-1.145` over five seeds; oracle persistence 7.10 pp,
  one-shot 14.05 -> multistart 6.95), and the OPLib point at x = 0.2422
  (realized `-0.895+/-3.415`; oracle persistence 14.17 pp on n <= 300);
  writes `outputs/ood_points.json`.
- `stakes_v2_recompute.py` recomputes the per-tier stakes means
  0.2609 / 0.2423 / 0.1685 / 0.1107 / 0.0862 from `artifacts/stakes/` +
  `artifacts/nearopt/` and asserts agreement with the shipped
  `artifacts/stakes/stakes_first200_v2.json` to < 1e-9
  (prints `V2_RECOMPUTE_OK`).
- `scale_ladder_numbers.py` prints the scale-ladder numbers of Section 4.4
  (section 8 below).

## 2. reproduce_op50_table (GPU training + GPU eval)

Train the three configurations at five seeds; each run is 100 epochs x 160k
instances, roughly 8-20 h on one GPU (inputs: none — on-the-fly generator;
outputs: `checkpoints/<name>/`):

```bash
for s in 42 43 44 45 46; do python src/sas/train_sas_op50.py --seed $s --arm sas; done
for s in 42 43 44 45 46; do python src/sas/train_sas_op50.py --seed $s --arm uniform; done
for s in 1234 42 43 44 45; do python src/sas/pomo_train_op50.py --seed $s; done   # 1234 = recipe seed
```

Evaluate each run on the canonical 10k-instance test set (a few minutes per
run on one GPU for the table schemes). Load the highest-numbered epoch file the run
kept, else `last.ckpt` (README, "Checkpoint layout"); the commands below
write `last.ckpt` for brevity. Outputs: scheme-table JSON + per-instance npz
per run:

```bash
mkdir -p outputs/five_seed
S=greedy,greedy_head,ms10,ms25,multistart,msaug
for s in 42 43 44 45 46; do for a in sas uniform; do
  python src/sas/op_eval.py --arm $a --ckpt checkpoints/${a}_op50_seed$s/last.ckpt \
      --dataset data/op/op_dist50_test_seed1234.pkl --schemes $S \
      --out outputs/five_seed/op50_${a}_seed$s.json
done; done
for s in 1234 42 43 44 45; do
  python src/sas/op_eval.py --arm pomo --ckpt checkpoints/pomo_op50_seed$s/last.ckpt \
      --dataset data/op/op_dist50_test_seed1234.pkl --schemes $S \
      --out outputs/five_seed/op50_pomo_seed$s.json
done
```

The shipped per-run outputs are in `artifacts/main_evals/five_seed/`; a
table cell is the mean +- std (ddof=1) over seeds of
`schemes[<scheme>]["gap_pct"]`, which `src/sas/five_seed_tables.py`
prints (to tabulate regenerated runs, place them in the same layout and
point `ARTIFACT_ROOT` at it). The per-seed Wilcoxon statistics of the paper's appendix come from
`python src/sas/paired_stats.py` (CPU, seconds, reads the shipped npz), and
`python src/sas/advantage_curve.py` reports the advantage curve per seed.
The sampling-1280 and augmentation-1280 rows of the appendix tables were
evaluated at three seeds for uniform and SAS (seeds 42/43/44) and in a single
run for POMO (the recipe seed), with all schemes of `op_eval.py` and
`pctsp_eval.py`; those per-run outputs are in
`artifacts/main_evals/op50/` and `artifacts/main_evals/pctsp50/`.

Anchor check (same checkpoints, same caliber, from
`artifacts/budget_sweep_curve.json`, T=3.0 row): the SAS one-shot benefit over
uniform starts at the canonical budget is 7.778 +/- 0.057 pp of the ILS
reference over the sweep's three seeds, with head hit@1 0.528.

### 2b. Start rule, sampling and beam search at matched budgets (GPU)

`src/sas/start_rule_baselines.py` decodes every Table 1 checkpoint with the
schemes of its docstring (a few minutes per checkpoint on one GPU): on SAS checkpoints, the
learned head and the start rule (prize / distance-to-depot) in its place; on
POMO checkpoints, the start rule, uniformly random feasible starts with
sampled continuation, the library's sampling decoder, and beam search.

```bash
mkdir -p outputs/baselines
for s in 1234 42 43 44 45; do
  python src/sas/start_rule_baselines.py --backbone pomo \
      --ckpt checkpoints/pomo_op50_seed$s/last.ckpt --out outputs/baselines/pomo_seed$s.json
done
for s in 42 43 44 45 46; do
  python src/sas/start_rule_baselines.py --backbone sas \
      --ckpt checkpoints/sas_op50_seed$s/last.ckpt --out outputs/baselines/sas_seed$s.json
done
python src/sas/aggregate_baselines.py    # reads artifacts/main_evals/op50/baselines/
```

### 2c. Head fitted to a frozen POMO checkpoint (GPU)

`src/sas/frozen_head_fit.py` freezes a finished POMO checkpoint and trains only
the 128 head weights with SAS's credit term (10 epochs x 2,500 steps, ~30 min
per seed; the backbone is asserted bit-identical afterwards). The fitted model
is a SAS checkpoint and is evaluated by the Table 1 harness:

```bash
mkdir -p outputs/frozen_head
for s in 1234 42 43 44 45; do
  python src/sas/frozen_head_fit.py --ckpt checkpoints/pomo_op50_seed$s/last.ckpt \
      --seed $s --out-ckpt outputs/frozen_head/frozen_pomo_seed$s.ckpt
  python src/sas/op_eval.py --arm sas --ckpt outputs/frozen_head/frozen_pomo_seed$s.ckpt \
      --dataset data/op/op_dist50_test_seed1234.pkl --schemes greedy_head,ms10,ms25,multistart \
      --out outputs/frozen_head/eval_frozen_pomo_seed$s.json
done
python src/sas/aggregate_frozen_head.py  # reads artifacts/main_evals/op50/frozen_head/
```

## 3. reproduce_pctsp_table (GPU training + GPU eval)

Same shape as section 2 (~8-20 h per training run, minutes per eval):

```bash
for s in 42 43 44 45 46; do python src/sas/train_sas_pctsp50.py --seed $s --arm sas; done
for s in 42 43 44 45 46; do python src/sas/train_sas_pctsp50.py --seed $s --arm uniform; done
for s in 1234 42 43 44 45; do python src/sas/pomo_train_pctsp50.py --seed $s; done

# pctsp_eval.py has no --schemes option and always runs every scheme of Tables 2 and 7
for s in 42 43 44 45 46; do for a in sas uniform; do
  python src/sas/pctsp_eval.py --arm $a --ckpt checkpoints/${a}_pctsp50_seed$s/last.ckpt \
      --dataset data/pctsp/pctsp50_test_seed1234.pkl \
      --out outputs/five_seed/pctsp50_${a}_seed$s.json
done; done
for s in 1234 42 43 44 45; do
  python src/sas/pctsp_eval.py --arm pomo --ckpt checkpoints/pomo_pctsp50_seed$s/last.ckpt \
      --dataset data/pctsp/pctsp50_test_seed1234.pkl \
      --out outputs/five_seed/pctsp50_pomo_seed$s.json
done
```

Anchor check (SAS checkpoints of seeds 42/43/44, from
`artifacts/ood_points.json`): the PCTSP50 calibration point produced from
them (section 5) is realized 7.17 +/- 0.24 pp, oracle 9.61 pp, ratio 0.746.

The POMO+ head-to-head uses the same datasets
(~8-20 h per training run, ~1-4 h per eval):

```bash
for p in op pctsp; do for s in 42 43 44; do
  python src/pomoplus/pplus_train.py --seed $s --problem $p
done; done
mkdir -p outputs/pomoplus
for s in 42 43 44; do
  python src/pomoplus/pplus_eval_op.py --arm pplus \
      --ckpt checkpoints/pomoplus_op50_seed$s/last.ckpt \
      --dataset data/op/op_dist50_test_seed1234.pkl \
      --out outputs/pomoplus/pomoplus_op50_seed${s}_eval.json
  python src/pomoplus/pplus_eval_pctsp.py --arm pplus \
      --ckpt checkpoints/pomoplus_pctsp50_seed$s/last.ckpt \
      --dataset data/pctsp/pctsp50_test_seed1234.pkl \
      --out outputs/pomoplus/pomoplus_pctsp50_seed${s}_eval.json
done
python src/pomoplus/pplus_hit1_op50.py   # CPU, minutes; reads artifacts/nearopt/
```

The shipped POMO+ ladder and hit@1 aggregates live in
`artifacts/main_evals/pomoplus/`; the SAS figures quoted beside POMO+ are the
five-seed evaluations of seeds 42/43/44.

## 4. reproduce_calibration (the budget-sweep curve)

Stage 0 (optional; CPU, ~1-2 h at 14 workers): tier selection curve.
`python src/calibration/calibrate_budget.py` -> visit fraction vs budget in
`outputs/budget_calibration.json`; compare with
`artifacts/budget_calibration.json`. Input:
`data/op/op_dist50_test_seed1234.pkl` (instances 5000-5063 only).

Stage 1 (GPU; 12 runs x ~8-20 h): train the four non-anchor tiers:

```bash
for T in 1.5 2 4 4.5; do for s in 42 43 44; do
  python src/calibration/budget_sweep_train.py --seed $s --budget $T
done; done
```

The anchor tier T=3.0 is NOT retrained: it reuses
`checkpoints/sas_op50_seed{42,43,44}` from section 2.

Stage 2 (GPU; ~15-30 min per (tier, seed)): outcome measurement, e.g.:

```bash
mkdir -p outputs/op50_sweep_evals
python src/calibration/budget_sweep_eval.py \
    --ckpt checkpoints/sas_op50_T1.5_seed42/last.ckpt --budget 1.5 \
    --dataset data/op/op_dist50_test_seed1234.pkl \
    --out outputs/op50_sweep_evals/eval_T1.5_s42.json
# ... repeat for every (T, seed); for T=3.0 use
#     --ckpt checkpoints/sas_op50_seed42/last.ckpt --budget 3.0
#     --out outputs/op50_sweep_evals/eval_T3_s42.json
```

Stage 3 (CPU; ~2-6 h per tier at 14 workers): per-tier ILS references:
`python src/instruments/budget_sweep_nearopt.py 1.5` (likewise `2`, `4`, `4.5`)
-> `outputs/nearopt_T{T}_first1000.json`. The T=3.0 reference ships as
`artifacts/nearopt/nearopt_starts_op50_first1000.json`.

Stage 4 (CPU; ~1-3 h at 14 workers): start-stakes instrument:
`python src/instruments/budget_sweep_stakes.py` -> `outputs/stakes_first200.json`.

Stage 5 (CPU, seconds): stakes + exact verification:
`python src/calibration/stakes_v2_recompute.py`. This reads the shipped
artifacts; to verify regenerated stage-2/3/4 outputs instead, arrange them
in the `artifacts/` layout and point `ARTIFACT_ROOT` at that directory.

Stage 6 (CPU; seconds + ~2-3 min): aggregation, uncertainty and the
budget-only comparison:
`python src/calibration/budget_sweep_aggregate.py`, then
`python src/calibration/calibration_uncertainty.py` and
`python src/calibration/budget_predictors.py`.

Verify-only path (no GPU, no retraining — the out-of-the-box default): run
`budget_sweep_aggregate.py`, `calibration_uncertainty.py`,
`budget_predictors.py` and `ood_points.py` directly on the shipped artifacts,
as in section 1. Expected: the five-tier table in section 1; OLS slope 71.77
with 95% CI [43.44, 100.09] and R^2 0.9559 (bootstrap and
leave-one-tier-out ranges printed alongside); pooled realized/oracle ratio
0.7342 +/- 0.0164; leave-one-budget-out RMSE 1.80 (stakes), 1.38 (1/T),
0.85 (log T).

## 5. reproduce_ood_points (cross-problem + shifted calibration points)

CPU stakes instruments (inputs: `data/`, `artifacts/nearopt/`):

```bash
python src/calibration/ood_stakes.py --set op100    # ~4-12 h at 14 workers
python src/calibration/ood_stakes.py --set oplib    # ~6-24 h (n up to 300)
python src/calibration/pctsp_stakes.py              # ~2-8 h at 8 workers
```

-> `outputs/stakes_{op100,oplib,pctsp50}.json`; the shipped copies live in
`artifacts/stakes/`. Expected stakes means (the x-coordinates recorded in
`artifacts/ood_points.json`): OP100 0.1936, OPLib 0.2422, PCTSP50 0.0944.

GPU calibration-point eval (~15-30 min per seed; needs the section-3 SAS
checkpoints):

```bash
mkdir -p outputs/pctsp_point
for s in 42 43 44; do
  python src/calibration/pctsp_budget_sweep_eval.py \
      --ckpt checkpoints/sas_pctsp50_seed$s/last.ckpt \
      --dataset data/pctsp/pctsp50_test_seed1234.pkl \
      --out outputs/pctsp_point/pctsp_cal_s$s.json
done
```

Assembly (CPU, seconds; runs out of the box on the shipped artifacts):
`python src/calibration/ood_points.py` -> `outputs/ood_points.json`.
The shifted points are the zero-shot one-shot advantage over the separately
trained uniform configuration, paired by seed over seeds 42-46, read from the
five-seed evaluations in `artifacts/shift_evals/` (section 7). Expected:
PCTSP50 realized 7.17 +/- 0.24 (oracle 9.61, ratio 0.746); OP100 realized
3.72 +/- 1.15, oracle persistence 7.10 pp (one-shot 14.05 -> multistart
6.95); OPLib realized -0.89 +/- 3.42, oracle persistence 14.17 pp (one-shot
48.32 -> ms50 34.16, n <= 300 subset).

## 6. reproduce_figures (CPU, seconds each)

All four paper figures are reproducible from the shipped release; the first
three use only shipped artifacts, while the geometry overlay additionally
uses the canonical public OP50 test pickle:

```bash
python src/figures/fig_variance_contrast.py    # reads artifacts/main_evals/five_seed/
python src/figures/make_calibration_fig.py     # reads artifacts/budget_sweep_curve.json
                                               # + artifacts/ood_points.json
python src/figures/make_advantage_curve_fig.py # reads artifacts/main_evals/five_seed/
python src/figures/make_head_overlay_fig.py    # reads artifacts/nearopt/ +
                                               # artifacts/head_diag/ +
                                               # artifacts/main_evals/op50/ +
                                               # data/op/op_dist50_test_seed1234.pkl
```

Outputs: `outputs/fig_variance_contrast.png` (five SAS runs, plain greedy
13.69 +- 1.85 against argmax start 3.769 +- 0.183), `outputs/fig_calibration.png`
(five-tier line labeled `71.8 x stakes - 2.6`, `R^2 = 0.96`, the dotted
line y = 100 sigma, the three amber off-distribution points and the PCTSP50
oracle point), `outputs/fig_advantage_curve.png` (one-shot advantage
`+7.66` over the five seed pairs, zero at N = n = 50; also writes
`outputs/advantage_curve_5seed.json`, which matches the shipped
`artifacts/main_evals/op50/advantage_curve_5seed.json`: `7.6624 +- 0.1062`),
and `outputs/fig_head_vs_nearopt.png` (the only figure needing the test-set
pickle; it has no in-image title and writes the aggregate hit@1 52.8% +- 5.6%
to `outputs/head_vs_nearopt_stats.json`).

## 7. Other analyses

- Constructor baseline, OP50 (CPU, ~15 s at 16 processes; input `data/`):
  `python src/calibration/constructor_op50_full.py` ->
  `outputs/constructor_op50_full.json` + per-instance npz. Expected gaps
  against Compass on all 10,000 test instances (from
  `artifacts/constructor_op50_full.json`): best-of-1 30.16 / best-of-5
  14.96 / best-of-16 11.32, against SAS one-shot 3.77.
- Constructor baseline, OPLib (CPU, ~10-40 min; inputs `data/` +
  OPLib + `artifacts/nearopt/`): `python src/calibration/constructor_baseline.py`
  -> `outputs/constructor_baseline.json`. Expected OPLib (n<=300) gaps
  27.39 / 19.48 / 17.20 (from `artifacts/constructor_baseline.json`).
- Batched throughput (GPU + CPU, ~10 min):
  `python src/calibration/batched_throughput.py` (the Table 6 batched rows;
  the paper's run ships as `artifacts/timing/batched_throughput.json`).
- Single-instance latency (one CPU thread, ~5 min; inputs `data/`, OPLib and
  the `sas_op50_seed42` checkpoint):
  `python src/calibration/latency_single_instance.py --devices cpu`
  (batch 1; the first 100 OP50 test instances and the 102 OPLib instances
  with n <= 300; ILS rows on the first 32 OP50 instances; `--smoke` for a
  dry run). The run behind the single-instance rows of Table 6, on one core
  of an Intel Xeon Gold 6240R, ships as
  `artifacts/timing/latency_single_instance_cpu_xeon6240r.json`: SAS
  one-shot 22.6 / 37.7 ms, one construction 1.8 / 10.2 ms and best of
  sixteen 31.0 / 183 ms (OP50 / OPLib), ILS 0.32 s and 5.2 s. Wall times
  are machine-dependent. The same script on one AMD EPYC 7543 core and with
  `--devices cuda` on one A100 ships as
  `artifacts/timing/latency_single_instance_cpu_amd.json` and
  `artifacts/timing/latency_single_instance_gpu_a100.json`.
- Head hit@1 on 1000 instances (CPU, ~30-60 min; needs section-2/3 SAS
  checkpoints): `python src/instruments/head_hit1_first1000.py`; the
  aggregate ships as
  `artifacts/head_diag/head_hit1_first1000.json`.
- Near-optimal start sets that are the two ends of the best tour (CPU,
  seconds): `python src/instruments/nearopt_two_ends.py` reads
  `artifacts/nearopt/nearopt_starts_op50_first1000.json` and prints 830 of
  1,000 instances (0.83), mean set size 2.215 and maximum 5 (Section 3.5).
- Feasible starts (CPU, seconds; input `data/`):
  `python src/instruments/guard_feasibility.py` reproduces
  `artifacts/guard_feasibility_five_tiers.json` (Appendix B: 39.4 / 48.6
  feasible starts and fewer than 50 feasible on 90.7% / 45.3% of instances
  at T = 1.5 / 2.0).
- Achievability probe (CPU, ~1-2 h):
  `python src/instruments/achievability_probe.py` (reads the shipped
  `artifacts/op50_sweep_evals/` head snapshots; the per-instance arrays of
  the paper's run ship as `artifacts/achievability_probe_perinst.npz`).
- OP100 zero-shot table (GPU, minutes per run): the section-2 checkpoints on
  `data/op/op_dist100_test_seed1234.pkl` via
  `src/sas/op_eval.py --n 100 --schemes greedy,greedy_head,ms10,ms25,ms50,multistart,msaug`
  (POMO without `greedy_head`), five seeds per configuration; shipped per-run
  outputs for all fifteen runs are `artifacts/shift_evals/op100_*`, and
  `src/sas/five_seed_tables.py` prints the table.
- OPLib zero-shot table (GPU, ~1-3 h per run):
  `python src/shift/oplib_eval.py --arm sas
  --ckpt checkpoints/sas_op50_seed42/last.ckpt --out outputs/oplib_sas_seed42_eval.json`
  (likewise every seed of `--arm uniform` and `--arm pomo`; all fifteen
  shipped runs are `artifacts/shift_evals/oplib_*`). Head diagnostics: GPU
  snapshots via `src/shift/oplib_head_diag.py` (shipped in
  `artifacts/head_diag/`), then the CPU cross-analysis
  `python src/shift/oplib_head_nearopt_analysis.py` (seconds; prints the
  paper's OPLib hit@1 `11.8% +- 4.3%`, the per-gen split, and the
  cross-seed argmax agreement).
- OP100 / OPLib / PCTSP near-optimal references (CPU, several hours each;
  the originals ran with many workers):
  `src/instruments/op100_nearopt.py`,
  `src/instruments/oplib_nearopt.py`,
  `src/instruments/pctsp_nearopt.py` (shipped copies in
  `artifacts/nearopt/`).
- TSP/CVRP must-visit control (GPU; ~8-20 h per training run):

  ```bash
  python src/controls/mustvisit_train.py --problem tsp --arm pomo --seed 1234
  python src/controls/mustvisit_train.py --problem cvrp --arm pomo --seed 1234
  for s in 42 43; do
    python src/controls/mustvisit_train.py --problem tsp --arm sas --seed $s
    python src/controls/mustvisit_train.py --problem cvrp --arm sas --seed $s
  done
  python src/controls/mustvisit_eval.py --problem tsp --out outputs/control_tsp_eval.json
  python src/controls/mustvisit_eval.py --problem cvrp --out outputs/control_cvrp_eval.json
  ```

  The shipped evals (uniformity + parity, exactly the paper's control
  table) are `artifacts/main_evals/mustvisit/control_{tsp,cvrp}_eval.json`.

## 8. reproduce_scale_ladder (Section 4.4, Appendix H)

Budget per scale (CPU): `python src/scale/scale_ladder_calibrate_budget.py`
(T = 4.0 / 5.5 / 9.0 at n = 100 / 200 / 500). Training (GPU, one run per
scale and seed 42/43/44): `python src/scale/scale_ladder_train.py` (see its
arguments). Evaluation (GPU) on the held-out set of each scale, with the
oracle taken over every feasible start:

```bash
python src/scale/scale_ladder_eval.py --arm sas --full-oracle \
    --ckpt checkpoints/sas_op500_T9_seed42/last.ckpt --budget 9.0 \
    --dataset <held-out n=500 set> --out outputs/scale_ladder/sas_op500_seed42.json
```

The held-out sets are named in the `dataset` field of every shipped cell
JSON and pinned by its `dataset_md5`. The n = 50 cells
(`artifacts/scale_ladder/eval_out/sas_op50_seed*.json`) are the section-2 SAS
checkpoints of seeds 42/43/44 on all 10,000 OP50 test instances; at n = 50 the
fifty-draw pool is every feasible start. `python src/scale/scale_ladder_numbers.py`
(CPU, seconds) then prints, from the shipped per-instance files:
realized advantage +1.812 / +2.317 / +3.221 prize units (standard errors
0.057 / 0.100 / 0.423); normalized advantage 0.0876 / 0.0600 / 0.0375 /
0.0196 at n = 50 / 100 / 200 / 500; log-log exponent -0.695 with
multiplier 1.48 over the three trained scales; capture over every feasible
start 0.712 / 0.574 / 0.505 (pooled 0.597 over nine cells, 0.358 to 0.745),
fifty-draw pool 0.729 / 0.618 / 0.584; and from n = 50 to 500 a 4.46x fall
of the normalized advantage that factors into 3.05x (available value) and
1.46x (capture). `python src/scale/scale_ladder_aggregate.py --evdir
artifacts/scale_ladder/eval_out --out outputs/scale_ladder_summary.json`
writes the per-scale significance summary shipped as
`artifacts/scale_ladder/eval_out/scale_ladder_summary.json`.

Zero-shot comparison (GPU; Section 4.5): the same evaluation with each section-2
SAS checkpoint of seeds 42/43/44 on the n = 100 held-out set
(`--ckpt checkpoints/sas_op50_seed42/last.ckpt --budget 4.0 --dataset <held-out n=100 set>`,
shipped as `artifacts/scale_ladder/eval_out/zeroshot_sas_op50_on_op100_seed*.json`).
`scale_ladder_numbers.py` then also prints capture 0.369 for the n = 50 model
against 0.729 for the n = 100 model in the fifty-draw pool, paired +0.360 +- 0.012,
and pool start values of 3.43 against 2.46 prize units.

Start-stakes across scales (CPU; Appendix H): `python
src/calibration/stakes_across_scales_report.py` prints, from the shipped
`artifacts/stakes/stakes_across_scales.json`, the stakes at n = 50 / 100 / 200 /
500 under the three solver configurations (as defined 0.170 / 0.190 / 0.210 /
0.217; light solver on both sides 0.090 to 0.034; reference solver on both
sides 0.091 to 0.024) and the light solver's unconstrained result over the
reference (0.967 to 0.835). One scale is remeasured with
`python src/calibration/stakes_across_scales.py --variant <as_defined | light_both |
reference_both> --n <n> --budget <T> --dataset <evaluation set> --out <json>`
(minutes at n = 50, hours on many cores at n = 500). The driver imports
`src/instruments/budget_sweep_stakes.py`, which loads
`$DATA_ROOT/op/op_dist50_test_seed1234.pkl` at import, so that file must be in
place at every n.
