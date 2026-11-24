"""Budget-sweep calibration — SAS on OP50 at a budget tier. Recipe IDENTICAL
to the main paper's SAS arm (src/sas/train_sas_op50.py; rl4co pomo.yaml
verbatim);
the ONLY new degree of freedom is the generator budget --budget T.
Anchor tier T=3.0 is NOT retrained (main-paper checkpoints reused).
"""
import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs import OPEnv
from rl4co.utils.trainer import RL4COTrainer

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sas"))
from sas_pomo import SASPOMO  # noqa: E402

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--budget", type=float, required=True)
ap.add_argument("--num-workers", type=int, default=0)
ap.add_argument("--resume-ckpt", default=None,
                help="Lightning ckpt to resume from (weights + optimizer + "
                     "scheduler + epoch; on-the-fly data stream does not "
                     "bit-replay, same distribution)")
args = ap.parse_args()

pl.seed_everything(args.seed, workers=True)

env = OPEnv(generator_params=dict(num_loc=50, max_length=args.budget))
model = SASPOMO(
    env,
    sas_arm="sas",
    batch_size=64,
    train_data_size=160_000,
    val_data_size=10_000,
    test_data_size=10_000,
    optimizer_kwargs=dict(lr=1e-4, weight_decay=1e-6),
    lr_scheduler="MultiStepLR",
    lr_scheduler_kwargs=dict(milestones=[80, 95], gamma=0.1),
    dataloader_num_workers=args.num_workers,
)

run = f"sas_op50_T{args.budget:g}_seed{args.seed}"
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
trainer.fit(model, ckpt_path=args.resume_ckpt)
print(f"BUDGET_SWEEP_FIT_COMPLETE {run} best:", ckpt_cb.best_model_path, ckpt_cb.best_model_score)
