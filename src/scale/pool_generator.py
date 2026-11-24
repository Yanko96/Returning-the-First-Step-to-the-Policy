"""Fixed-pool generator adapter for the rl4co OP environment.

Replaces OPEnv's on-the-fly OPGenerator with one that INDEXES the pre-generated fixed pool
(regime_pool sharded float32 .npy in shared memory). Training then draws from the fixed
1.6M pool (reuse <=10x over 100ep x 160k), not fresh uniform OP. Emits exactly the OP env's
TensorDict format: locs (B,n,2), depot (B,2), prize (B,n), max_length (B,). A fixed seeded
permutation gives a reproducible sampling order; the pool is memmapped (no RAM blowup, shared
across concurrent runs on the same machine).
"""
import glob

import numpy as np
import torch
from tensordict.tensordict import TensorDict


class PoolOPGenerator:
    def __init__(self, pooldir, seed=0):
        self.pooldir, self.seed = pooldir, seed
        self.co = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(f"{pooldir}/coords_*.npy"))]
        self.pr = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(f"{pooldir}/prize_*.npy"))]
        self.ml = [np.load(f, mmap_mode="r") for f in sorted(glob.glob(f"{pooldir}/maxlen_*.npy"))]
        assert self.co, f"no coords_*.npy in {pooldir}"
        sizes = [c.shape[0] for c in self.co]
        self.offsets = np.concatenate([[0], np.cumsum(sizes)])
        self.N = int(self.offsets[-1])
        self.n = self.co[0].shape[1] - 1              # num_loc (coords = depot + n locs)
        self.num_loc = self.n                          # POMO reads this for num_starts
        self.min_loc, self.max_loc = 0.0, 1.0          # normalized coords (OPEnv._make_spec)
        self.perm = np.random.default_rng(seed).permutation(self.N)
        self.ptr = 0
        print(f"PoolOPGenerator: {self.N} instances, n={self.n}, {len(self.co)} shards, "
              f"reuse-order seed={seed}", flush=True)

    def __getstate__(self):
        # CRITICAL: do NOT pickle the memmaps — that materializes the whole 1.6M pool into
        # every Lightning checkpoint (~5GB each -> fills disk, cross-device rename crash).
        # Save only the light config; reload the pool on unpickle.
        return {"pooldir": self.pooldir, "seed": self.seed}

    def __setstate__(self, state):
        self.__init__(state["pooldir"], state["seed"])

    def __call__(self, batch_size):
        if not isinstance(batch_size, int):
            batch_size = int(batch_size[0])
        gi = self.perm[(self.ptr + np.arange(batch_size)) % self.N]
        self.ptr = (self.ptr + batch_size) % self.N
        m = self.co[0].shape[1]
        depot = np.empty((batch_size, 2), np.float32)
        locs = np.empty((batch_size, m - 1, 2), np.float32)
        prize = np.empty((batch_size, m - 1), np.float32)
        maxlen = np.empty((batch_size,), np.float32)
        sh = np.searchsorted(self.offsets, gi, side="right") - 1
        for j in range(batch_size):
            s = int(sh[j]); li = int(gi[j] - self.offsets[s])
            c = self.co[s][li]
            depot[j] = c[0]; locs[j] = c[1:]; prize[j] = self.pr[s][li]; maxlen[j] = self.ml[s][li]
        return TensorDict(
            {"locs": torch.from_numpy(locs), "depot": torch.from_numpy(depot),
             "prize": torch.from_numpy(prize), "max_length": torch.from_numpy(maxlen)},
            batch_size=[batch_size],
        )
