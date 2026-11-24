"""POMO+-ported — POMO+ auxiliary start agent implemented following the
published description (arXiv 2508.08493) inside the same harness as SAS,
for the head-to-head on selective routing. Code for POMO+ is not public;
every paper-anchored element and every interpretation choice is itemized
in the paper's appendix and mirrored in comments here.

Paper-anchored (follows the published description):
- separate auxiliary agent with its own parameters: one Multi-Head
  Attention layer + one feedforward network, residual + normalization
  after each block, then a scoring head of linear layers -> one score
  per node;
- Gumbel-Softmax distribution during training, Softmax at inference;
- encoder embeddings enter the agent through a stop-gradient (encoder
  trained by the POMO main loss only, as in SAS);
- the agent's own REINFORCE objective with POMO's shared mean baseline,
  gradient accumulation over n=100 steps per agent update;
- K = 0.2N starts sampled per instance (OP50/PCTSP50: K=10), each
  repeated N/K times to fill N trajectories.

Interpretation choices (the published description is silent):
- dims: 8 heads, d=128 (backbone dim), FFN hidden 512, LayerNorm;
  scoring head Linear(128,128)+ReLU+Linear(128,1).
- Gumbel-Softmax temperature 1.0.
- agent optimizer Adam lr 1e-4 wd 1e-6 + MultiStepLR [80,95] 0.1
  ("share all hyperparameters reported in the POMO paper").
- the K starts are sampled WITHOUT replacement.
- feasibility masking identical to SAS (depot masked, only feasible
  first actions scored) -- fairness addition for OP/PCTSP.
- the agent path runs in an fp32 island inside the mixed-precision
  forward (fp16 Gumbel noise underflows; same fix as SAS).
- at evaluation both methods' start distributions are consumed by
  the same budget-ladder rules (argmax one-shot, top-N multistart).

Engineering note on the n=100 accumulation: the agent loss graph is
disjoint from the main POMO loss graph (embeddings detached, agent
parameters excluded from the main optimizer), so it is backpropagated
directly with a raw .backward() outside the Lightning/AMP GradScaler
path (the agent path is fully fp32), gradients accumulate in the agent
parameters across steps, and a dedicated Adam steps every n=100 batches.
"""
import torch
import torch.nn as nn
from einops import rearrange
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

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


class StartAgent(nn.Module):
    """POMO+ auxiliary agent: MHA block + FFN block (residual+LN
    after each) + linear scoring head -> one score per node."""

    def __init__(self, embed_dim: int = 128, n_heads: int = 8, ff_hidden: int = 512):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden), nn.ReLU(), nn.Linear(ff_hidden, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.score = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, h):  # h: [B, n+1, d] -> [B, n+1]
        a, _ = self.mha(h, h, h, need_weights=False)
        h = self.norm1(h + a)
        h = self.norm2(h + self.ff(h))
        return self.score(h).squeeze(-1)


class PPlusPolicy(AttentionModelPolicy):
    """AM policy + POMO+ auxiliary start agent. The forward replicates
    rl4co v0.6.0 ConstructivePolicy.forward with the same single insertion
    point as SASPolicy (select_start_nodes_fn injection)."""

    def __init__(self, *args, embed_dim: int = 128, k_frac: float = 0.2,
                 start_mask_first: bool = True, **kwargs):
        super().__init__(*args, embed_dim=embed_dim, **kwargs)
        self.start_mask_first = start_mask_first
        self.k_frac = k_frac          # K = 0.2N (paper)
        self.gs_tau = 1.0             # Gumbel-Softmax temperature (fixed)
        self.start_agent = StartAgent(embed_dim)
        self._start_logp = None
        self._agent_entropy = None
        self._pad_events = 0

    def pop_start_stats(self):
        out = (self._start_logp, self._agent_entropy, self._pad_events)
        self._start_logp, self._agent_entropy, self._pad_events = None, None, 0
        return out

    @staticmethod
    def _feasible_pad_draw(logp_all, mask, k):
        """With-replacement draws that can NEVER select an infeasible node:
        infeasible probabilities are exactly zero (no clamp floor), and rows
        whose feasible mass underflowed entirely fall back to uniform over
        feasible."""
        probs = torch.nan_to_num(logp_all.exp(), nan=0.0) * mask
        dead = probs.sum(-1, keepdim=True) <= 0
        probs = torch.where(dead, mask.float(), probs)
        return torch.multinomial(probs, k, replacement=True)

    def _make_start_fn(self, hidden, sampling: bool):
        """Training (sampling=True): Gumbel-Softmax distribution, K distinct
        starts sampled without replacement, each repeated N/K times (paper).
        Eval (sampling=False): Softmax scores, deterministic top-N distinct
        starts (budget-ladder rules, identical consumption to SAS)."""

        def fn(td, env, num_starts):
            B = td.shape[0]
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                h32 = hidden.float().detach()  # stop-gradient (paper; same as SAS)
                mask = td["action_mask"].clone()  # [B, n+1] feasible first actions
                if self.start_mask_first:
                    mask[..., 0] = False          # depot never a start (fairness masking)
                scores = self.start_agent(h32)    # [B, n+1] fp32 island
                if sampling:
                    # Gumbel-Softmax distribution over nodes (paper; tau=1)
                    u = torch.rand_like(scores).clamp(1e-10, 1.0 - 1e-7)
                    g = -torch.log(-torch.log(u))
                    dist_logits = (scores + g) / max(self.gs_tau, 1e-6)
                else:
                    dist_logits = scores          # Softmax at inference (paper)
                dist_logits = dist_logits.masked_fill(~mask, float("-inf"))
                logp_all = torch.log_softmax(dist_logits, dim=-1)  # [B, n+1]
                n_feas = mask.sum(-1)  # [B]

                if sampling:
                    k = min(int(round(self.k_frac * num_starts)), mask.size(-1))
                    k = max(k, 1)
                    # K distinct starts without replacement (paper), sampled from
                    # the Gumbel-Softmax distribution via the Gumbel-top-k
                    # trick on its logits. (Gumbel-top-k is used rather than
                    # multinomial on clamp_min'ed exp() probabilities: once
                    # the agent concentrates, fp32 exp underflow zeroes the
                    # feasible tail and a clamp floor would make infeasible
                    # nodes samplable; Gumbel-top-k keeps infeasible at
                    # -inf with no underflow path.)
                    u2 = torch.rand_like(dist_logits).clamp(1e-10, 1.0 - 1e-7)
                    g2 = -torch.log(-torch.log(u2))
                    sel_k = (dist_logits + g2).topk(k, dim=-1).indices  # [B, k]
                    if bool((n_feas < k).any()):
                        # rows with fewer feasible than k: replace the trailing
                        # (-inf) picks by feasible draws (counted, as in SAS)
                        pad = self._feasible_pad_draw(logp_all, mask, k)
                        need = torch.arange(k, device=td.device)[None, :] >= n_feas[:, None]
                        sel_k = torch.where(need, pad, sel_k)
                        self._pad_events += int(need.any(dim=1).sum())
                    reps = max(num_starts // k, 1)
                    selected = sel_k.repeat_interleave(reps, dim=1)  # [B, k*reps]
                    if selected.size(1) < num_starts:  # remainder: cycle from top
                        selected = torch.cat(
                            [selected, sel_k[:, : num_starts - selected.size(1)]], dim=1
                        )
                    selected = selected[:, :num_starts]
                else:
                    # eval: deterministic top-N distinct, pad rule as SAS
                    k = min(num_starts, mask.size(-1))
                    selected = dist_logits.topk(k, dim=-1).indices  # [B, k]
                    if bool((n_feas < k).any()):
                        pad = self._feasible_pad_draw(logp_all, mask, k)
                        need = torch.arange(k, device=td.device)[None, :] >= n_feas[:, None]
                        selected = torch.where(need, pad, selected)
                        self._pad_events += int(need.any(dim=1).sum())

                assert bool(mask.gather(1, selected).all()), "infeasible start selected"
                # REINFORCE credit: log p_omega(a1 | s) under the (Gumbel-)Softmax
                # distribution, column-aligned with the start-major flat layout
                self._start_logp = logp_all.gather(1, selected)  # [B, num_starts]
                assert bool(torch.isfinite(self._start_logp).all()), "non-finite start_logp"
                self._last_selected = selected.detach()
                with torch.no_grad():
                    p = logp_all.exp()
                    ent = (-p * torch.where(mask, logp_all, torch.zeros_like(logp_all))).sum(-1)
                    self._agent_entropy = float(ent.mean())
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

        # --- PPLUS-INSERT (the only change vs library forward) ---
        if (
            "multistart" in decode_type
            and decoding_kwargs.get("select_start_nodes_fn") is None
        ):
            sampling = ("sampling" in decode_type) and self.training
            decoding_kwargs["select_start_nodes_fn"] = self._make_start_fn(
                hidden, sampling=sampling
            )
        # --- end PPLUS-INSERT ---

        decode_strategy: DecodingStrategy = get_decoding_strategy(
            decode_type,
            temperature=decoding_kwargs.pop("temperature", self.temperature),
            tanh_clipping=decoding_kwargs.pop("tanh_clipping", self.tanh_clipping),
            mask_logits=decoding_kwargs.pop("mask_logits", self.mask_logits),
            store_all_logp=decoding_kwargs.pop("store_all_logp", return_entropy),
            **decoding_kwargs,
        )

        td, env, num_starts = decode_strategy.pre_decoder_hook(td, env)

        # --- one-shot agent-argmax first step (same mechanics as SAS) ---
        if one_shot_start:
            assert "multistart" not in decode_type and num_starts <= 1, \
                "one_shot_start is for plain single-rollout decode only"
            mask0 = td["action_mask"].clone()
            if self.start_mask_first:
                mask0[..., 0] = False
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                scores0 = self.start_agent(hidden.float().detach())
            scores0 = scores0.masked_fill(~mask0, float("-inf"))
            a0 = scores0.argmax(-1)  # [B]
            td.set("action", a0)
            td = env.step(td)["next"]
            if decode_strategy.store_all_logp:
                decode_strategy.logprobs.append(torch.zeros_like(td["action_mask"]))
            else:
                decode_strategy.logprobs.append(
                    torch.zeros_like(a0, dtype=hidden.dtype, device=td.device)
                )
            decode_strategy.actions.append(a0)
        # --- end one-shot ---

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


class PPlusPOMO(POMO):
    """POMO backbone (unchanged main loss/optimizer) + POMO+ auxiliary
    agent trained by its own REINFORCE with n=100 gradient accumulation."""

    AGENT_ACCUM = 100  # paper: n=100 for all models

    def __init__(self, env, policy=None, policy_kwargs={},
                 start_mask_first: bool = True, **kwargs):
        if policy is None:
            pk = {
                "num_encoder_layers": 6,
                "normalization": "instance",
                "use_graph_context": False,
            }
            pk.update(policy_kwargs)
            policy = PPlusPolicy(env_name=env.name,
                                 start_mask_first=start_mask_first, **pk)
        super().__init__(env, policy, **kwargs)
        self._agent_opt = None
        self._agent_sched = None
        self._agent_accum_count = 0

    def configure_optimizers(self):
        # main optimizer/scheduler over the backbone ONLY; agent parameters
        # are excluded and owned by the dedicated agent optimizer below
        backbone = [p for n, p in self.named_parameters() if "start_agent" not in n]
        return super().configure_optimizers(backbone)

    def on_fit_start(self):
        super().on_fit_start()
        # same hyperparameters as the main recipe
        self._agent_opt = Adam(self.policy.start_agent.parameters(),
                               lr=1e-4, weight_decay=1e-6)
        self._agent_sched = MultiStepLR(self._agent_opt, milestones=[80, 95], gamma=0.1)
        self._agent_accum_count = 0
        n_agent = sum(p.numel() for p in self.policy.start_agent.parameters())
        log.info(f"POMO+ agent parameters: {n_agent}")

    def on_train_epoch_end(self):
        super().on_train_epoch_end()
        if self._agent_sched is not None:
            self._agent_sched.step()

    def calculate_loss(self, td, batch, policy_out, reward=None, log_likelihood=None):
        policy_out = super().calculate_loss(td, batch, policy_out, reward, log_likelihood)
        start_logp, agent_ent, pads = self.policy.pop_start_stats()
        if self.training and start_logp is not None:
            adv = (reward - policy_out["bl_val"]).detach()  # [B, n_start]
            agent_loss = -(adv * start_logp).mean()
            # disjoint fp32 graph: raw backward outside the AMP scaler path;
            # gradients accumulate in agent params (not in the main optimizer)
            agent_loss.backward()
            self._agent_accum_count += 1
            if self._agent_accum_count % self.AGENT_ACCUM == 0:
                self._agent_opt.step()
                self._agent_opt.zero_grad(set_to_none=True)
            policy_out["agent_loss"] = agent_loss.detach()
            self.log("train/agent_loss", agent_loss.detach(),
                     on_step=False, on_epoch=True)
        if agent_ent is not None:
            self.log("train/agent_entropy", float(agent_ent),
                     on_step=False, on_epoch=True)
            self.log("train/start_pad_events", float(pads),
                     on_step=False, on_epoch=True)
        return policy_out
