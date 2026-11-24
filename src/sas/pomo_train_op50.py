"""Plain-POMO baseline — OP50, rl4co official recipe.

Recipe = rl4co v0.6.0 configs/experiment/routing/pomo.yaml VERBATIM
(batch 64, 160k inst/epoch, lr 1e-4 wd 1e-6, MultiStepLR [80,95] gamma 0.1,
100 epochs, seed 1234), env swapped tsp->op (num_loc=50, generator defaults:
dist prizes, T=3.0). Deviations from the published yaml: none.

Baseline check: greedy one-shot gap on the canonical Kool
op_dist50_test_seed1234 set is expected to land in [10%, 20%] (RL4CO E.1: 14.26%);
op_eval.py records it as `baseline_check`.
"""
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs import OPEnv
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.trainer import RL4COTrainer

import argparse
import os

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=1234,
                help="1234 is the released recipe seed; the five-seed tables add 42-45")
args = ap.parse_args()
RUN = f"pomo_op50_seed{args.seed}"

pl.seed_everything(args.seed, workers=True)

env = OPEnv(generator_params=dict(num_loc=50))

model = POMO(
    env,
    batch_size=64,
    train_data_size=160_000,
    val_data_size=10_000,
    test_data_size=10_000,
    optimizer_kwargs=dict(lr=1e-4, weight_decay=1e-6),
    lr_scheduler="MultiStepLR",
    lr_scheduler_kwargs=dict(milestones=[80, 95], gamma=0.1),
)

ckpt_cb = ModelCheckpoint(
    dirpath=os.path.join(CHECKPOINT_ROOT, RUN),
    filename="epoch{epoch:03d}",
    auto_insert_metric_name=False,
    monitor="val/reward",
    mode="max",
    save_top_k=3,
    save_last=True,
)
logger = CSVLogger(save_dir=os.path.join(OUTPUT_ROOT, "logs"), name=RUN)

trainer = RL4COTrainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    logger=logger,
    callbacks=[ckpt_cb],
    default_root_dir=OUTPUT_ROOT,
)
trainer.fit(model)
print("BASELINE_FIT_COMPLETE best:", ckpt_cb.best_model_path, ckpt_cb.best_model_score)
