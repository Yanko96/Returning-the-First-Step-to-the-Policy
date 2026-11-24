"""SAS training at scale n. Recipe IDENTICAL to the main paper's SAS
arm (sas_pomo.SASPOMO, rl4co pomo.yaml verbatim); the ONLY new degrees
of freedom are --num-loc (scale) and, at the top of the ladder, effective-batch
preservation via gradient accumulation (--micro-batch x --accum = 64), a
recipe-preserving device adaptation (the reported runs use 32 x 2 at n = 200 and
16 x 4 at n = 500, and the split at n = 100 varied with the GPU).

--smoke: 1 epoch, a few batches, small data; prints peak VRAM to check the run fits.

NOTE: execution is under `if __name__ == "__main__":` — REQUIRED for
--num-workers > 0 (DataLoader workers re-import this module; without the guard
they recursively re-run trainer.fit and deadlock).
"""

import os as _os, sys as _sys  # resolve sibling package modules
_SRC = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _d in ('sas', 'instruments'):
    _p = _os.path.join(_SRC, _d)
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from rl4co.envs import OPEnv
from rl4co.utils.trainer import RL4COTrainer

from sas_pomo import SASPOMO

import os as _os; WORKDIR = _os.environ.get("SCALE_LADDER_WORKDIR", ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--num-loc", type=int, default=50)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--micro-batch", type=int, default=64, help="per-step batch")
    ap.add_argument("--accum", type=int, default=1, help="accumulate_grad_batches; "
                    "micro-batch*accum should equal the recipe's 64")
    ap.add_argument("--max-epochs", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--resume-ckpt", default=None)
    ap.add_argument("--pool", default=None, help="The regime study: fixed-pool dir (regime_pool); "
                    "replaces on-the-fly OPEnv generator. num-loc/budget come from the pool.")
    ap.add_argument("--arm", default="sas", choices=["sas", "pomo"], help="SAS head or plain POMO")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # rl4co builds its DataLoader without a multiprocessing_context; under the
    # default 'spawn' it hangs here (workers pickle/re-import rl4co's dataset and
    # deadlock). Force 'fork' for worker DataLoaders: workers inherit parent memory,
    # no pickle, no re-import. (num_workers=0 unaffected.)
    if args.num_workers > 0:
        import multiprocessing as _mp
        import torch.utils.data as _tud
        _orig_dl_init = _tud.DataLoader.__init__
        def _dl_init(self, *a, **k):
            if k.get("num_workers", 0) and not k.get("multiprocessing_context"):
                k["multiprocessing_context"] = _mp.get_context("fork")
            return _orig_dl_init(self, *a, **k)
        _tud.DataLoader.__init__ = _dl_init

    pl.seed_everything(args.seed, workers=True)

    eff_batch = args.micro_batch * args.accum
    if args.pool:
        from pool_generator import PoolOPGenerator
        env = OPEnv(generator=PoolOPGenerator(args.pool, seed=args.seed))
        args.num_loc = env.generator.num_loc          # authoritative from the pool
    else:
        env = OPEnv(generator_params=dict(num_loc=args.num_loc, max_length=args.budget))
    common = dict(
        batch_size=args.micro_batch,
        train_data_size=1_024 if args.smoke else 160_000,
        val_data_size=512 if args.smoke else 10_000,
        test_data_size=512 if args.smoke else 10_000,
        optimizer_kwargs=dict(lr=1e-4, weight_decay=1e-6),
        lr_scheduler="MultiStepLR",
        lr_scheduler_kwargs=dict(milestones=[80, 95], gamma=0.1),
        dataloader_num_workers=args.num_workers,
    )
    if args.arm == "pomo":
        # independent POMO baseline = plain rl4co POMO (pomo_train_op50.py recipe,
        # no SAS head). SASPOMO only supports sas/uniform; POMO must be the vanilla class.
        from rl4co.models.zoo.pomo import POMO
        model = POMO(env, **common)
    else:
        model = SASPOMO(env, sas_arm=args.arm, **common)   # sas | uniform

    run = (f"{args.arm}_regime_seed{args.seed}" if args.pool
           else f"{args.arm}_op{args.num_loc}_T{args.budget:g}_seed{args.seed}")
    print(f"RUN {run} eff_batch={eff_batch} (micro={args.micro_batch} x accum={args.accum}) "
          f"num_loc={args.num_loc} workers={args.num_workers} smoke={args.smoke}", flush=True)

    trainer_kw = dict(accelerator="gpu", devices=1,
                      accumulate_grad_batches=args.accum, default_root_dir=WORKDIR)
    if args.smoke:
        trainer = RL4COTrainer(max_epochs=1, limit_train_batches=4, limit_val_batches=2,
                               enable_checkpointing=False, logger=False, **trainer_kw)
        trainer.fit(model)
        if torch.cuda.is_available():
            print(f"SMOKE_PEAK_GB {torch.cuda.max_memory_allocated() / 1024**3:.2f}", flush=True)
        print(f"SMOKE_DONE {run}", flush=True)
    else:
        ckpt_cb = ModelCheckpoint(
            dirpath=f"{WORKDIR}/checkpoints/{run}", filename="epoch{epoch:03d}",
            auto_insert_metric_name=False, monitor="val/reward", mode="max",
            save_top_k=3, save_last=True,
        )
        logger = CSVLogger(save_dir=f"{WORKDIR}/logs", name=run)
        trainer = RL4COTrainer(max_epochs=args.max_epochs, logger=logger,
                               callbacks=[ckpt_cb], **trainer_kw)
        trainer.fit(model, ckpt_path=args.resume_ckpt)
        if torch.cuda.is_available():
            print(f"PEAK_GB {torch.cuda.max_memory_allocated() / 1024**3:.2f}", flush=True)
        print(f"SCALE_LADDER_FIT_COMPLETE {run} best:", ckpt_cb.best_model_path,
              ckpt_cb.best_model_score, flush=True)
        open(f"{WORKDIR}/checkpoints/{run}/DONE", "w").close()


if __name__ == "__main__":
    main()
