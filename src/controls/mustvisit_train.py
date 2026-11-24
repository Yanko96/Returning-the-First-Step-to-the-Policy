"""Must-visit control training — plain POMO or SAS on TSP50/CVRP50,
official recipe
verbatim (nothing tuned for must-visit)."""
import argparse

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs import TSPEnv, CVRPEnv
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.trainer import RL4COTrainer

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "sas"))
from sas_pomo import SASPOMO  # noqa: E402

CHECKPOINT_ROOT = os.environ.get("CHECKPOINT_ROOT", "checkpoints")
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")

ap = argparse.ArgumentParser()
ap.add_argument("--problem", choices=["tsp", "cvrp"], required=True)
ap.add_argument("--arm", choices=["pomo", "sas"], required=True)
ap.add_argument("--seed", type=int, required=True)
args = ap.parse_args()

pl.seed_everything(args.seed, workers=True)
env = (TSPEnv if args.problem == "tsp" else CVRPEnv)(
    generator_params=dict(num_loc=50))
mask_first = args.problem != "tsp"  # TSP: node 0 is an ordinary city

common = dict(
    batch_size=64, train_data_size=160_000, val_data_size=10_000,
    test_data_size=10_000,
    optimizer_kwargs=dict(lr=1e-4, weight_decay=1e-6),
    lr_scheduler="MultiStepLR",
    lr_scheduler_kwargs=dict(milestones=[80, 95], gamma=0.1),
)
if args.arm == "pomo":
    model = POMO(env, **common)
else:
    model = SASPOMO(env, sas_arm="sas", start_mask_first=mask_first, **common)

run = (f"sas_{args.problem}50_seed{args.seed}" if args.arm == "sas"
       else f"pomo_{args.problem}50")
print(f"RUN {run} mask_first={mask_first}")
ckpt_cb = ModelCheckpoint(
    dirpath=os.path.join(CHECKPOINT_ROOT, run), filename="epoch{epoch:03d}",
    auto_insert_metric_name=False, monitor="val/reward", mode="max",
    save_top_k=1, save_last=True)
trainer = RL4COTrainer(
    max_epochs=100, accelerator="gpu", devices=1,
    logger=CSVLogger(save_dir=os.path.join(OUTPUT_ROOT, "logs"), name=run),
    callbacks=[ckpt_cb], default_root_dir=OUTPUT_ROOT)
trainer.fit(model)
print(f"CONTROL_FIT_COMPLETE {run}")
