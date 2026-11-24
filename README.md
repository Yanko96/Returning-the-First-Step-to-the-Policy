# Returning the First Step to the Policy — code release

Code for the paper *"Returning the First Step to the Policy: Repairing the
First-Step Bias of Multistart Neural Solvers on Selective Routing"* by Yangzhe
Kong and Yanyuan Mo (equal contribution), Transactions on Machine Learning
Research, 2026 (https://openreview.net/forum?id=KfzwYcSe7I). It contains everything behind the reported
numbers: the SAS start head and its training, the exact uniform-start
control, the start-rule, sampling and beam-search baselines, the head fitted
to a frozen POMO checkpoint, the POMO+ implementation following the published
description, the local-search instruments (near-optimal start sets,
start-stakes, the constructor baseline), the budget-sweep calibration with
its budget-only comparison predictors, the scale ladder, and the
distribution-shift studies.

See `RUN.md` for step-by-step commands: a CPU-only smoke test first, then
the full reproduction pipelines with hardware requirements, rough wall
times, and the expected headline numbers.

## Environment

Validated environment (CUDA 12.1, NVIDIA): Python 3.10-3.12,
PyTorch 2.5.1 (CUDA 12.1 wheels), RL4CO 0.6.0. Appendix A of the paper lists
the hardware behind the timing figures; the recipe below is the
configuration this release was validated against. `requirements.txt` lists
the core validated versions; use the installation commands below for the
complete environment. RL4CO and Lightning must be installed with
`--no-deps` to avoid a conflicting torch pull; installing lightning 2.6.5
together with pytorch-lightning 2.5.1 under `--no-deps` is intentional
(rl4co 0.6.0 imports both namespaces), and a plain
`pip install -r requirements.txt` does not express that order — follow the
commands below:

```bash
pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps rl4co==0.6.0 lightning==2.6.5 \
    pytorch-lightning==2.5.1 tensordict==0.8.0 torchrl==0.8.0
pip install cloudpickle orjson packaging einops omegaconf hydra-core \
    hydra-colorlog torchmetrics lightning-utilities pyrootutils rich \
    scipy robust-downloader matplotlib pandas
```

Instrument and analysis scripts are pure numpy/scipy (CPU only).

## Path configuration

Every script reads its locations from five environment variables, each with
a release-relative default. Run everything from the release root and the
defaults just work:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ROOT` | `data` | test-set pickles + OPLib checkout |
| `CHECKPOINT_ROOT` | `checkpoints` | trained model checkpoints |
| `ARTIFACT_ROOT` | `artifacts` | shipped result artifacts (read-only inputs) |
| `OPLIB_ROOT` | `$DATA_ROOT/OPLib-master` | OPLib repository checkout |
| `OUTPUT_ROOT` | `outputs` | everything the scripts write |

Expected data layout (each evaluation records the md5 of the file it reads):

```
data/
  op/op_dist50_test_seed1234.pkl      md5 47a828139a9fcf1e642a060f40837d9e
  op/op_dist100_test_seed1234.pkl     md5 53eeff929a4339563a044b9fdee5fd69
  pctsp/pctsp50_test_seed1234.pkl     md5 df23990603d21aa25ed41f87640db4fa
  OPLib-master/                       https://github.com/bcamath-ds/OPLib
    instances/gen{1,2,3}/*.oplib
    solutions/ea4op/gen{1,2,3}/*.sol
```

### Data acquisition

The three test pickles are the canonical Kool et al. test sets, generated
deterministically by the public, MIT-licensed repository
https://github.com/wouterkool/attention-learn-to-route (`generate_data.py`);
we do not redistribute them to keep this archive small. From that
repository's root:

```bash
python generate_data.py --problem op --data_distribution dist \
    --graph_sizes 50 100 --name test --seed 1234
python generate_data.py --problem pctsp --graph_sizes 50 --name test --seed 1234
```

This writes `data/op/op_dist50_test_seed1234.pkl`,
`data/op/op_dist100_test_seed1234.pkl`, and
`data/pctsp/pctsp50_test_seed1234.pkl` (10,000 instances each); copy or
symlink them into `$DATA_ROOT` keeping the `op/` and `pctsp/`
subdirectories. The md5s above verify the version only; on a mismatch,
regenerate with the pinned seed and check your numpy version.

OPLib: clone https://github.com/bcamath-ds/OPLib (instances and EA4OP
solutions are static files; any current checkout works) and place it as
`$DATA_ROOT/OPLib-master`.

### Checkpoint layout

Checkpoint layout: `checkpoints/<name>/` with `<name>` one of
`sas_op50_seed{42..46}`, `uniform_op50_seed{42..46}`,
`pomo_op50_seed{1234,42,43,44,45}`, `sas_pctsp50_seed{42..46}`,
`uniform_pctsp50_seed{42..46}`, `pomo_pctsp50_seed{1234,42,43,44,45}`,
`pomoplus_op50_seed{42,43,44}`, `pomoplus_pctsp50_seed{42,43,44}`,
`sas_op50_T{1.5,2,4,4.5}_seed{42,43,44}` (budget-sweep tiers),
`sas_op{100,200,500}_T{4,5.5,9}_seed{42,43,44}` (scale ladder),
`sas_tsp50_seed{42,43}`, `sas_cvrp50_seed{42,43}`, `pomo_tsp50`,
`pomo_cvrp50`. The training commands in `RUN.md` write exactly this layout.
Each run keeps its three best epochs by validation reward plus `last.ckpt`;
evaluations load the highest-numbered epoch file a run kept (epoch 99, or 98
for one run), else `last.ckpt`, and every shipped evaluation JSON records the
file it loaded.

## Artifact strategy

Checkpoints are **not** shipped: all models retrain from scratch with fixed
seeds via the commands in `RUN.md`. Everything downstream of training,
however, ships: the `artifacts/` tree carries the per-run outputs behind
every table and figure in the paper (per-run evaluation JSONs plus the
per-instance arrays they summarize). The artifact-backed aggregation,
recalculation, and figure scripts run without a GPU; the geometry overlay
additionally needs the canonical public OP50 test pickle, as noted in
`RUN.md`. Scripts that regenerate raw experiments require the public data,
checkpoints, and (where noted) a GPU. Every reported table cell can be
recomputed and checked against the paper from the shipped per-run outputs.

## Layout

```
src/sas          SAS model (sas_pomo.py: POMO + 128-param start head +
                 restored step-zero credit term; arms "sas"/"uniform"),
                 training (OP50, PCTSP50, plain-POMO baselines), the
                 budget-ladder evaluation harnesses (op_eval.py,
                 pctsp_eval.py), the five-seed tables, the start-rule /
                 sampling / beam-search baselines, and the head fitted to a
                 frozen POMO checkpoint
src/pomoplus     POMO+ implementation following the published description
                 (auxiliary selection agent, own REINFORCE loop) + its
                 evaluations and hit@1
src/instruments  reference ILS solvers (op_ils_reference.py,
                 pctsp_ils_reference.py), near-optimal start-set
                 extraction (OP50 tiers, OP100, OPLib, PCTSP),
                 start-stakes instrument, achievability probe, OPLib parser
src/calibration  budget-sweep training/eval/aggregation, uncertainty and
                 robustness recomputation, budget-only predictors,
                 cross-problem and shifted-set stakes, calibration-point
                 eval, constructor baseline (OPLib and the full OP50 test
                 set), stakes recomputation, the start-stakes across scales,
                 batched throughput and single-instance latency
src/scale        scale ladder (budget calibration, training, evaluation,
                 the reported numbers) and OPLib capture by prize scheme
src/shift        OPLib zero-shot evaluation (integer-metric re-validation
                 against EA4OP), head diagnostics, and the head-vs-near-opt
                 cross-analysis (oplib_head_nearopt_analysis.py)
src/controls     TSP50/CVRP50 must-visit transplant control
src/figures      generators for all four paper figures (variance contrast,
                 calibration, advantage curve, head overlay)
artifacts/       the per-run outputs behind every table and figure:
                 aggregated JSONs at the root (budget_sweep_curve.json,
                 budget_calibration.json, calibration_uncertainty.json,
                 budget_predictors.json, ood_points.json,
                 constructor_baseline.json, constructor_op50_full.json +
                 its per-instance npz) plus
                 achievability_probe_perinst.npz,
                 main_evals/five_seed/ Tables 1 and 2: eval JSON +
                                      per-instance npz for every
                                      configuration and seed
                 main_evals/op50/     baselines/ (start rule, sampling,
                                      beam search) and frozen_head/ with
                                      their summaries; the five-seed
                                      advantage curve; the evaluations
                                      behind the sampling-1280 and
                                      augmentation-1280 rows (three seeds
                                      for uniform and SAS, one POMO run)
                 main_evals/pctsp50/  the evaluations behind the PCTSP50
                                      sampling/augmentation rows (three
                                      seeds for uniform and SAS, one POMO run)
                 main_evals/pomoplus/ POMO+ budget ladder + hit@1
                 main_evals/mustvisit/ TSP/CVRP control evals
                 op50_sweep_evals/    per-tier eval JSONs + per-instance npz
                 nearopt/             near-optimal start sets + ILS references
                 stakes/              start-stakes instrument outputs
                 pctsp_point/         PCTSP calibration-point eval
                 shift_evals/         OP100 and OPLib zero-shot evals, all
                                      three configurations at five seeds
                                      (+ per-instance npz for OP100)
                 head_diag/           OPLib head snapshots (3 seeds) + OP50
                                      hit@1 aggregates
                 scale_ladder/        per-cell ladder evals (n = 50, 100,
                                      200, 500) + OPLib regime zero-shot
                 prize_axis/          controlled prize-structure pool
                 timing/              batched throughput and single-instance
                                      latency, one JSON per machine
```

## Paper number -> script map

| Paper item | Producing scripts | Shipped per-run outputs |
|---|---|---|
| Tables 1 and 2 (five seeds per configuration), seed-paired differences in Section 3 | `src/sas/train_sas_op50.py`, `src/sas/pomo_train_op50.py`, `src/sas/train_sas_pctsp50.py`, `src/sas/pomo_train_pctsp50.py`, `src/sas/op_eval.py`, `src/sas/pctsp_eval.py`; tables: **`src/sas/five_seed_tables.py`** | `artifacts/main_evals/five_seed/` |
| Table 1 start-rule, sampling and beam-search rows; learned head vs rule, sampling and beam (paired) | `src/sas/start_rule_baselines.py` (GPU), **`src/sas/aggregate_baselines.py`** | `artifacts/main_evals/op50/baselines/`, `artifacts/main_evals/op50/baselines_summary.json` |
| Table 1 head fitted to a frozen POMO checkpoint | `src/sas/frozen_head_fit.py` (GPU), then `src/sas/op_eval.py --arm sas` on the fitted checkpoint; **`src/sas/aggregate_frozen_head.py`** | `artifacts/main_evals/op50/frozen_head/`, `artifacts/main_evals/op50/frozen_head_summary.json` |
| Sampling-1280 and augmentation-1280 rows of the full tables (three seeds for uniform and SAS and a single run for POMO, as stated in Appendices C and F) | `src/sas/op_eval.py`, `src/sas/pctsp_eval.py` | `artifacts/main_evals/op50/`, `artifacts/main_evals/pctsp50/` |
| Advantage-vs-budget curve, variance figure | `src/figures/make_advantage_curve_fig.py`, `src/figures/fig_variance_contrast.py`, per-seed report `src/sas/advantage_curve.py` | `artifacts/main_evals/five_seed/`, `artifacts/main_evals/op50/advantage_curve_5seed.json` |
| Per-seed paired Wilcoxon statistics (Appendix A) | `src/sas/paired_stats.py` | reads `artifacts/main_evals/five_seed/` |
| Near-optimal start sets, hit@1, overlay figure | `src/instruments/budget_sweep_nearopt.py` (OP50 tiers), `src/instruments/pctsp_nearopt.py`, `src/instruments/op100_nearopt.py`, `src/instruments/oplib_nearopt.py`, `src/instruments/head_hit1_first1000.py`, `src/figures/make_head_overlay_fig.py` | `artifacts/nearopt/`, `artifacts/head_diag/` |
| Near-optimal start sets that are exactly the two ends of the best tour, 830 of 1,000 instances (Section 3.5) | **`src/instruments/nearopt_two_ends.py`** (CPU, seconds; reads the shipped sets, reruns no solver) | `artifacts/nearopt/nearopt_starts_op50_first1000.json` |
| Feasible starts at the five length budgets (Appendix B: 39.4 / 48.6 of 50 feasible starts and fewer than 50 feasible on 90.7% / 45.3% of instances at T = 1.5 / 2.0, every start feasible at T >= 3, first 1,000 test instances) | **`src/instruments/guard_feasibility.py`** (CPU, seconds) | `artifacts/guard_feasibility_five_tiers.json` |
| Length-budget sweep (Figure 2 and Table 3, Section 4.2) | `src/calibration/budget_sweep_train.py`, `src/calibration/budget_sweep_eval.py`, `src/instruments/budget_sweep_nearopt.py`, `src/instruments/budget_sweep_stakes.py`, `src/calibration/budget_sweep_aggregate.py`; figure: `src/figures/make_calibration_fig.py` | `artifacts/op50_sweep_evals/`, `artifacts/budget_sweep_curve.json` |
| Start-stakes against budget-only predictors (leave-one-budget-out Table 4, Section 4.3) and the stakes with the light solver on both sides | **`src/calibration/budget_predictors.py`** | `artifacts/budget_predictors.json` |
| Calibration uncertainty + robustness | `src/calibration/calibration_uncertainty.py` | `artifacts/calibration_uncertainty.json` |
| Cross-problem / shifted calibration points | `src/calibration/pctsp_stakes.py`, `ood_stakes.py`, `pctsp_budget_sweep_eval.py`, `ood_points.py` (shifted points from the five-seed zero-shot evaluations) | `artifacts/ood_points.json`, `artifacts/stakes/`, `artifacts/pctsp_point/`, `artifacts/shift_evals/` |
| Stakes recomputation (verification) | `src/calibration/stakes_v2_recompute.py` | reads `artifacts/stakes/` + `artifacts/nearopt/` |
| Hit-rate lower-bound probe | `src/instruments/achievability_probe.py` | `artifacts/achievability_probe_perinst.npz` |
| Constructor baseline: OP50 on all 10,000 test instances against Compass (30.2 / 15.0 / 11.3 %) | **`src/calibration/constructor_op50_full.py`** | `artifacts/constructor_op50_full.json`, `artifacts/constructor_op50_full_perinst.npz` |
| Constructor baseline: OPLib n <= 300 (27.4 / 19.5 / 17.2 %) | `src/calibration/constructor_baseline.py` | `artifacts/constructor_baseline.json` |
| Single-instance latency (Table 6, batch 1, one core of an Intel Xeon Gold 6240R): SAS one-shot 22.6 / 37.7 ms, one construction 1.8 / 10.2 ms, best of sixteen 31.0 / 183 ms on OP50 / OPLib n <= 300; ILS 0.32 s and 5.2 s with 1 and 16 restarts on the first 32 OP50 instances; the size brackets of Appendix A and the "about 23 ms" of Section 2.1 | `src/calibration/latency_single_instance.py --devices cpu` (wall times are machine-dependent) | `artifacts/timing/latency_single_instance_cpu_xeon6240r.json` |
| Batched-throughput rows (Table 6: 0.051 / 0.053 / 0.41 / 4.2 ms per instance on one RTX 4090) | `src/calibration/batched_throughput.py` (wall times are machine-dependent) | `artifacts/timing/batched_throughput.json` |
| OP100 / OPLib zero-shot tables (five seeds per configuration) | `src/sas/op_eval.py --n 100` (OP100) and **`src/sas/five_seed_tables.py`**; `src/shift/oplib_eval.py` (OPLib); head diagnostics `src/shift/oplib_head_diag.py` + `src/shift/oplib_head_nearopt_analysis.py` | `artifacts/shift_evals/`, `artifacts/head_diag/` |
| TSP/CVRP must-visit control | `src/controls/mustvisit_train.py`, `src/controls/mustvisit_eval.py` | `artifacts/main_evals/mustvisit/` |
| POMO+ head-to-head | `src/pomoplus/pplus_train.py`, `pplus_eval_op.py`, `pplus_eval_pctsp.py`, `pplus_hit1_op50.py` | `artifacts/main_evals/pomoplus/` |
| Scale ladder: training and length-budget choice (Section 4.4, Appendix H) | `src/scale/scale_ladder_calibrate_budget.py` fixes T per scale (4.0 / 5.5 / 9.0) by matching the near-optimal visit fraction to 0.55; `src/scale/scale_ladder_train.py` trains one run per (scale, seed) | `artifacts/scale_ladder/eval_out/` |
| Scale ladder: realized benefit, normalized benefit, log-log exponent, capture (Section 4.4, Appendix H) | `src/scale/scale_ladder_eval.py --full-oracle` -> per-cell JSON + per-instance npz; **`src/scale/scale_ladder_numbers.py` prints the reported numbers** (+1.81 / +2.32 / +3.22 prize units; normalized 0.0876 / 0.0600 / 0.0375 / 0.0196 at n = 50 / 100 / 200 / 500; exponent -0.695 with multiplier 1.48 over the three trained scales; capture over every feasible start 0.712 / 0.574 / 0.505, pooled 0.597, cells 0.358-0.745; fifty-draw pool 0.729 / 0.618 / 0.584); `src/scale/scale_ladder_aggregate.py` -> `scale_ladder_summary.json` | `artifacts/scale_ladder/eval_out/*.json`, `*_perinst.npz`, `scale_ladder_summary.json` |
| Zero-shot comparison at n = 100 (Section 4.5): capture in the fifty-draw pool 0.369 for the n = 50 model against 0.729 for the n = 100 model, paired +0.360 +- 0.012; pool start value 3.43 against 2.46 prize units | `src/scale/scale_ladder_eval.py` with each section-2 SAS checkpoint (seeds 42/43/44) on the n = 100 held-out set (`--ckpt checkpoints/sas_op50_seed42/last.ckpt --budget 4.0 --dataset <held-out n=100 set>`); **`src/scale/scale_ladder_numbers.py` prints the comparison** | `artifacts/scale_ladder/eval_out/zeroshot_sas_op50_on_op100_seed*.json` with the shipped `sas_op100_seed*` cells |
| Start-stakes across scales (Appendix H): as defined 0.170 / 0.190 / 0.210 / 0.217 at n = 50 / 100 / 200 / 500, light solver on both sides 0.090 to 0.034, reference solver on both sides 0.091 to 0.024 | `src/calibration/stakes_across_scales.py` (CPU; with the seeds it documents it reproduces all three series exactly); **`src/calibration/stakes_across_scales_report.py` prints the reported numbers** | `artifacts/stakes/stakes_across_scales.json` |
| OPLib capture by prize scheme (Section 4.5) | `src/scale/oplib_regime_eval.py --root <OPLIB_ROOT>/instances --no-equiv --n300 --seeds 42,43,44` -> the per-instance JSON; **`src/scale/oplib_capture_by_scheme.py` prints the three reported values** (-0.059 +- 0.082, +0.033 +- 0.055, +0.340 +- 0.072 over the 102 instances with n <= 300) | `artifacts/scale_ladder/oplib_regime_zeroshot_n300.json` |
| Prize structure on a controlled pool, 17 / 1 / 18 of 150 instances (Appendix G) | `src/instruments/regime_pool.py` generates the pool (seed and namespace recorded in its `manifest.json`); `src/instruments/hit1_probe.py --stage probe --control --n-inst 150` measures how often a *random* feasible start reaches 99.5 % of the unconstrained optimum, using `src/instruments/op_ils_reference_forced.py` as the local-search kernel. CPU only: `--control` replaces the head's pick with a random feasible start, so **no checkpoint is required**; the `scheme` column of `head_picks_g.npz` is the only field it reads from there and is identical to the pool's own `meta_*.npz`. The reported counts are the `probe_ge_995` fields times 150 | `artifacts/prize_axis/regime_holdout/`, `artifacts/prize_axis/hit1_probe_control.json`, `artifacts/prize_axis/head_picks_g.npz` |

## Conventions

- Training seeds (passed with `--seed`, as in `RUN.md`) and instrument seed
  bases (documented in file docstrings) are fixed and recorded. The main
  evaluation harnesses (`src/sas/op_eval.py`, `src/sas/pctsp_eval.py`,
  `src/shift/oplib_eval.py`, `src/pomoplus/pplus_eval_*.py`) do not seed their
  stochastic schemes (the 16 augmentations, sampling and augmentation at 1,280
  rollouts, the uniform control's random starts) and ship their per-instance
  outputs instead. The matched-budget sampling baselines
  (`src/sas/start_rule_baselines.py`) and the random-start draws and pools of
  the budget-sweep, scale-ladder and OPLib-capture evaluators
  (`src/calibration/budget_sweep_eval.py`,
  `src/calibration/pctsp_budget_sweep_eval.py`, `src/scale/scale_ladder_eval.py`,
  `src/scale/oplib_regime_eval.py`) use fixed seeds set in those scripts.
- Five-seed configurations: POMO
  1234/42/43/44/45 (the released recipe seed in place of one of the five),
  uniform and SAS 42-46; seed-paired statistics pair configurations trained
  with the same seed. POMO+, the budget-sweep tiers, the scale ladder and the
  local-search instruments run at three seeds (42/43/44).
- The primary evaluation harnesses load checkpoints with `strict=True`,
  record the dataset md5, and write per-instance arrays (`*_perinst.npz`)
  next to their summary JSON.
- Stakes convention: OP `1 - mean_forced/anchor`, PCTSP mirror
  `mean_forced/anchor - 1`, anchor = per-instance 16-restart reference
  best, clipped to [0, 1].
- The OP50 calibration fit is computed once from `budget_sweep_curve.json`;
  cross-problem and shifted points are overlays and never enter the fit.
- Shipped eval JSONs record their checkpoint under the release naming
  convention above and the dataset by release-relative path + md5.
- Arm naming: `--arm pomo` is the paper's "enumerated (POMO)" configuration --
  unmodified POMO with the library's enumerated forced starts;
  `--arm uniform` and `--arm sas` match the paper's configuration names directly.

## License

MIT (see LICENSE).
