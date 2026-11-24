"""SAS-POMO (Selection-Aware Starts) — POMO with a learned start head.

Verified against rl4co v0.6.0 sources:
- ConstructivePolicy.forward replicated verbatim (models/common/constructive/base.py,
  L157-268) with ONE marked insertion (SAS-INSERT: start-fn injection);
- start flat layout "(n b)" start-major == ops.select_start_nodes convention;
- POMO.shared_step train path: reward/ll unbatchified to [B, n_start] -> our
  start_logp is stored [B, n_start] column-aligned (col s == flat s*B+b);
- loss extension: L = L_POMO - mean(adv * logpi_head(start)), lambda == 1
  (the credit POMO hardcodes to zero at step 0, restored; no new hyperparam);
- tau anneal: epoch 0 -> 5.0, epoch 1 -> 3.0, epoch >= 2 -> 1.0.

Arms: sas_arm="sas" (learned head) | "uniform" (ablation: uniform over feasible,
zero credit term). Diagnosis arm = plain POMO (pomo_train_*.py).

Note: with N == n (=50) the
multistart start SET equals POMO's arange set (top-50 of 50); ms+aug differences
between arms are then TRAINING effects (step-0 credit + sampled-subset shaping),
while eval-time selection shows in greedy/sampling/N<n schemes.
"""
import torch
import torch.nn as nn
from einops import rearrange

from rl4co.envs import get_env
from rl4co.models.zoo.am import AttentionModelPolicy
from rl4co.models.zoo.pomo import POMO
from rl4co.utils.decoding import (
    DecodingStrategy,
    get_decoding_strategy,
    get_log_likelihood,
)
from rl4co.utils.ops import calculate_entropy
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class StartHead(nn.Module):
    """Single linear scorer on encoder node embeddings (~embed_dim params)."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(embed_dim, 1, bias=False)

    def forward(self, h):  # h: [B, n+1, d] -> [B, n+1]
        return self.proj(h).squeeze(-1)


class SASPolicy(AttentionModelPolicy):
    """AM policy + selection-aware start head."""

    def __init__(self, *args, embed_dim: int = 128, sas_arm: str = "sas",
                 start_mask_first: bool = True, **kwargs):
        super().__init__(*args, embed_dim=embed_dim, **kwargs)
        assert sas_arm in ("sas", "uniform")
        self.sas_arm = sas_arm
        # Mask convention: OP/PCTSP/CVRP mask index 0 (depot never a start); TSP has
        # no depot, node 0 is an ordinary city -> pass False there.
        self.start_mask_first = start_mask_first
        self.start_head = StartHead(embed_dim)
        self.start_tau = 1.0
        self._start_logp = None
        self._head_entropy = None
        self._pad_events = 0

    def pop_start_stats(self):
        out = (self._start_logp, self._head_entropy, self._pad_events)
        self._start_logp, self._head_entropy, self._pad_events = None, None, 0
        return out

    def _make_start_fn(self, hidden, sampling: bool):
        """select_start_nodes_fn closure; called by pre_decoder_hook on the
        PRE-batchify td. Stores per-start credit log-probs for the loss."""

        def fn(td, env, num_starts):
            # The whole head/Gumbel/log-softmax path runs in an fp32
            # island. Under the trainer default "16-mixed", clamp_min(1e-12)
            # underflows to 0 in fp16 -> -log(0)=inf -> NaN scores -> infeasible
            # starts selected -> -inf start_logp -> NaN weights.
            B = td.shape[0]
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                # detach — the credit term updates ONLY head weights;
                # no gradient path into the encoder.
                h32 = hidden.float().detach()
                mask = td["action_mask"].clone()  # [B, n+1] feasible first actions
                if self.start_mask_first:
                    mask[..., 0] = False          # depot never a start (POMO conv.)
                if self.sas_arm == "uniform":
                    logits = torch.zeros(B, mask.size(-1), device=td.device)
                else:
                    logits = self.start_head(h32)
                    if sampling:
                        logits = logits / max(self.start_tau, 1e-6)
                logits = logits.masked_fill(~mask, float("-inf"))
                k = min(num_starts, mask.size(-1))
                if self.sas_arm == "uniform":
                    scores = torch.where(
                        mask, torch.rand_like(logits),
                        torch.full_like(logits, float("-inf")),
                    )
                elif sampling:
                    # Gumbel-top-k: sample k distinct starts from softmax(logits)
                    u = torch.rand_like(logits).clamp(1e-10, 1.0 - 1e-7)
                    g = -torch.log(-torch.log(u))
                    scores = logits + g
                else:
                    scores = logits  # deterministic top-k (eval)
                selected = scores.topk(k, dim=-1).indices  # [B, k]
                n_feas = mask.sum(-1)  # [B]
                if (n_feas < k).any():
                    # pad: with-replacement resample from head softmax over feasible
                    probs = torch.softmax(logits, dim=-1)
                    probs = torch.nan_to_num(probs, nan=0.0) * mask
                    probs = probs.clamp_min(1e-12)
                    pad = torch.multinomial(probs, k, replacement=True)
                    need = torch.arange(k, device=td.device)[None, :] >= n_feas[:, None]
                    selected = torch.where(need, pad, selected)
                    self._pad_events += int(need.any(dim=1).sum())
                # fail fast at the source
                assert bool(mask.gather(1, selected).all()), "infeasible start selected"
                # credit term uses the marginal log-softmax (fixed formula)
                logp_all = torch.log_softmax(logits, dim=-1)  # [B, n+1] fp32
                self._start_logp = logp_all.gather(1, selected)  # [B, k]
                assert bool(torch.isfinite(self._start_logp).all()), "non-finite start_logp"
                self._last_selected = selected.detach()  # for wiring assertions
                with torch.no_grad():
                    p = logp_all.exp()
                    ent = (-p * torch.where(mask, logp_all, torch.zeros_like(logp_all))).sum(-1)
                    self._head_entropy = float(ent.mean())
            return rearrange(selected, "b n -> (n b)")  # start-major flat layout

        return fn

    def forward(
        self,
        td,
        env=None,
        phase: str = "train",
        calc_reward: bool = True,
        return_actions: bool = True,
        return_entropy: bool = False,
        return_hidden: bool = False,
        return_init_embeds: bool = False,
        return_sum_log_likelihood: bool = True,
        actions=None,
        max_steps=1_000_000,
        **decoding_kwargs,
    ) -> dict:
        # one_shot_start — true 1-rollout head-argmax (or uniform) first
        # step for PLAIN greedy decode, replicating the library's own multistart
        # injection mechanics (td.set/env.step/logprobs.append(0)/actions.append).
        one_shot_start = decoding_kwargs.pop("one_shot_start", False)
        # === replicated verbatim from rl4co v0.6.0 ConstructivePolicy.forward ===
        hidden, init_embeds = self.encoder(td)

        if isinstance(env, str) or env is None:
            env_name = self.env_name if env is None else env
            log.info(f"Instantiated environment not provided; instantiating {env_name}")
            env = get_env(env_name)

        decode_type = decoding_kwargs.pop("decode_type", None)
        if actions is not None:
            decode_type = "evaluate"
        elif decode_type is None:
            decode_type = getattr(self, f"{phase}_decode_type")

        # --- SAS-INSERT (the only change vs library forward) ---
        if (
            "multistart" in decode_type
            and decoding_kwargs.get("select_start_nodes_fn") is None
        ):
            sampling = ("sampling" in decode_type) and self.training
            decoding_kwargs["select_start_nodes_fn"] = self._make_start_fn(
                hidden, sampling=sampling
            )
        # --- end SAS-INSERT ---

        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # --- one-shot start injection (only for plain single-rollout decode) ---
        if one_shot_start:
            assert "multistart" not in decode_type and num_starts <= 1, \
                "one_shot_start is for plain single-rollout decode only"
            mask0 = td["action_mask"].clone()
            if self.start_mask_first:
                mask0[..., 0] = False  # depot never a start
            if self.sas_arm == "uniform":
                scores0 = torch.where(
                    mask0, torch.rand_like(mask0, dtype=hidden.dtype),
                    torch.full(mask0.shape, float("-inf"), device=td.device, dtype=hidden.dtype),
                )
            else:
                scores0 = self.start_head(hidden).masked_fill(~mask0, float("-inf"))
            a0 = scores0.argmax(-1)  # [B]
            td.set("action", a0)
            td = env.step(td)["next"]
            # replicate the library's multistart bookkeeping (logprob 0, recorded)
            if decode_strategy.store_all_logp:
                decode_strategy.logprobs.append(torch.zeros_like(td["action_mask"]))
            else:
                decode_strategy.logprobs.append(
                    torch.zeros_like(a0, dtype=hidden.dtype, device=td.device)
                )
            decode_strategy.actions.append(a0)
        # --- end one-shot start injection ---

        td, env, hidden = self.decoder.pre_decoder_hook(td, env, hidden, num_starts)

        step = 0
        while not td["done"].all():
            logits, mask = self.decoder(td, hidden, num_starts)
            td = decode_strategy.step(
                logits,
                mask,
                td,
                action=actions[..., step] if actions is not None else None,
            )
            td = env.step(td)["next"]
            step += 1
            if step > max_steps:
                log.error(f"Exceeded maximum number of steps ({max_steps})")
                break

        logprobs, actions, td, env = decode_strategy.post_decoder_hook(td, env)

        if calc_reward:
            td.set("reward", env.get_reward(td, actions))

        outdict = {
            "reward": td["reward"],
            "log_likelihood": get_log_likelihood(
                logprobs, actions, td.get("mask", None), return_sum_log_likelihood
            ),
        }
        if return_actions:
            outdict["actions"] = actions
        if return_entropy:
            outdict["entropy"] = calculate_entropy(logprobs)
        if return_hidden:
            outdict["hidden"] = hidden
        if return_init_embeds:
            outdict["init_embeds"] = init_embeds
        return outdict


class SASPOMO(POMO):
    """POMO + SAS start head. Loss = POMO loss + step-0 credit (lambda=1)."""

    def __init__(self, env, policy=None, policy_kwargs={}, sas_arm: str = "sas",
                 start_mask_first: bool = True, **kwargs):
        if policy is None:
            pk = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            pk.update(policy_kwargs)
            policy = SASPolicy(env_name=env.name, sas_arm=sas_arm,
                               start_mask_first=start_mask_first, **pk)
        super().__init__(env, policy, **kwargs)
        self.sas_arm = sas_arm

    def on_train_epoch_start(self):
        self.policy.start_tau = {0: 5.0, 1: 3.0}.get(self.current_epoch, 1.0)

    def calculate_loss(self, td, batch, policy_out, reward=None, log_likelihood=None):
        policy_out = super().calculate_loss(td, batch, policy_out, reward, log_likelihood)
        start_logp, head_ent, pads = self.policy.pop_start_stats()
        if self.sas_arm == "sas" and start_logp is not None:
            adv = (reward - policy_out["bl_val"]).detach()  # [B, n_start]
            start_loss = -(adv * start_logp).mean()
            policy_out["loss"] = policy_out["loss"] + start_loss
            policy_out["start_loss"] = start_loss
            # monitors logged explicitly to disk (rl4co log_metrics has a
            # fixed key set and would silently drop these)
            self.log("train/start_loss", start_loss, on_step=False, on_epoch=True)
        if head_ent is not None:
            policy_out["head_entropy"] = torch.as_tensor(head_ent)
            policy_out["start_pad_events"] = torch.as_tensor(float(pads))
            self.log("train/head_entropy", float(head_ent), on_step=False, on_epoch=True)
            self.log("train/start_pad_events", float(pads), on_step=False, on_epoch=True)
        return policy_out
