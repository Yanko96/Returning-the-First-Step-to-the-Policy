"""POMO+-ported training — OP50 / PCTSP50, recipe IDENTICAL to the SAS/
uniform/POMO arms (rl4co v0.6.0 pomo.yaml verbatim: batch 64, 160k
inst/epoch, lr 1e-4 wd 1e-6, MultiStepLR [80,95] gamma 0.1, 100 epochs);
the ONLY difference vs those arms = the start mechanism (PPlusPOMO,
POMO+ port following the published description). Seeds: 42/43/44.
"""
import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs import OPEnv, PCTSPEnv
from rl4co.utils.trainer import RL4COTrainer

from pomoplus_pomo import PPlusPOMO

import os

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--problem", choices=["op", "pctsp"], required=True)
ap.add_argument("--num-workers", type=int, default=0,
                help="dataloader workers; same value across ALL arms")
args = ap.parse_args()

pl.seed_everything(args.seed, workers=True)

env = (OPEnv if args.problem == "op" else PCTSPEnv)(
    generator_params=dict(num_loc=50))
model = PPlusPOMO(
    env,
    batch_size=64,
    train_data_size=160_000,
    val_data_size=10_000,
    test_data_size=10_000,
    optimizer_kwargs=dict(lr=1e-4, weight_decay=1e-6),
    lr_scheduler="MultiStepLR",
    lr_scheduler_kwargs=dict(milestones=[80, 95], gamma=0.1),
    dataloader_num_workers=args.num_workers,
)

run = f"pomoplus_{args.problem}50_seed{args.seed}"
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
print(f"PPLUS_FIT_COMPLETE {run} best:", ckpt_cb.best_model_path, ckpt_cb.best_model_score)
