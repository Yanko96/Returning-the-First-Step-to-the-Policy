"""SAS training — SAS-POMO (or uniform-ablation) on PCTSP50.

Recipe IDENTICAL to the plain-POMO baseline (pomo.yaml verbatim); the ONLY
difference vs
the baseline = the SAS mechanism (SASPOMO: detach + fp32 island +
tau schedule, carried over from OP unchanged). Seeds:
sas x {42,43,44}, uniform x {42,43,44}.

--num-workers: use the same value for every arm; recorded in the run name.
"""
import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs import PCTSPEnv
from rl4co.utils.trainer import RL4COTrainer

from sas_pomo import SASPOMO

import os

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--arm", choices=["sas", "uniform"], required=True)
ap.add_argument("--num-workers", type=int, default=0,
                help="dataloader workers; same value across ALL arms")
args = ap.parse_args()

pl.seed_everything(args.seed, workers=True)

env = PCTSPEnv(generator_params=dict(num_loc=50))
model = SASPOMO(
    env,
    sas_arm=args.arm,
    batch_size=64,
    train_data_size=160_000,
    val_data_size=10_000,
    test_data_size=10_000,
    optimizer_kwargs=dict(lr=1e-4, weight_decay=1e-6),
    lr_scheduler="MultiStepLR",
    lr_scheduler_kwargs=dict(milestones=[80, 95], gamma=0.1),
    dataloader_num_workers=args.num_workers,
)

run = f"{args.arm}_pctsp50_seed{args.seed}"
print(f"RUN {run} num_workers={args.num_workers}")
ckpt_cb = ModelCheckpoint(
    dirpath=os.path.join(CHECKPOINT_ROOT, run),
    filename="epoch{epoch:03d}",
    auto_insert_metric_name=False,
    monitor="val/reward",
    mode="max",
    save_top_k=3,
    save_last=True,
)
logger = CSVLogger(save_dir=os.path.join(OUTPUT_ROOT, "logs"), name=run)

trainer = RL4COTrainer(
    max_epochs=100,
    accelerator="gpu",
    devices=1,
    logger=logger,
    callbacks=[ckpt_cb],
    default_root_dir=OUTPUT_ROOT,
)
trainer.fit(model)
print(f"B_FIT_COMPLETE {run} best:", ckpt_cb.best_model_path, ckpt_cb.best_model_score)
