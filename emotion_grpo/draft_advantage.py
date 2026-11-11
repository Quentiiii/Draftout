"""Two-stage, unnormalized advantages for 2 x 2 emotion draft rollouts.

The Ray trainer normally supplies the complete rollout batch on its driver, so
``global_batch=True`` is the default. A sharded caller can opt into collective
statistics with ``global_batch=False``; every shard must contain complete image
groups. PPO still receives one advantage per response token, including repeated
draft prefixes. No GRPO standardization or advantage whitening is applied here.
"""

import math
import operator
from collections.abc import Mapping
from typing import Any

import torch
import torch.distributed as dist


@torch.no_grad()
def compute_draft_advantages(rewards: torch.Tensor, shrink: Any) -> dict[str, torch.Tensor]:
    """Apply the specified segment formulas in BF16, without normalization."""
    if rewards.ndim != 3 or tuple(rewards.shape[1:]) != (2, 2) or rewards.shape[0] == 0:
        raise ValueError("rewards must have shape [B, 2, 2] with B > 0")
    rewards = rewards.detach().to(dtype=torch.bfloat16)
    shrink = torch.as_tensor(shrink, dtype=torch.bfloat16, device=rewards.device).detach()
    if shrink.numel() != 1 or not torch.isfinite(shrink).all() or not (0 <= shrink.item() <= 1):
        raise ValueError("shrink must be one finite scalar between 0 and 1")
    shrink = shrink.reshape(())
    draft_mean = rewards.mean(dim=2)
    image_mean = draft_mean.mean(dim=1)
    value = image_mean[:, None] + shrink * (draft_mean - image_mean[:, None])
    draft_adv = value - image_mean[:, None]
    suffix_adv = rewards - value[:, :, None]
    # Arithmetic is deliberately BF16: allow one BF16 epsilon for cancellation.
    if not torch.allclose(
        draft_adv[:, :, None] + suffix_adv,
        rewards - image_mean[:, None, None],
        rtol=0,
        atol=torch.finfo(torch.bfloat16).eps,
    ):
        raise RuntimeError("draft advantage decomposition failed")
    return {
        "draft_mean": draft_mean,
        "image_mean": image_mean,
        "value": value,
        "draft_adv": draft_adv,
        "suffix_adv": suffix_adv,
    }


class DraftAdvantageEstimator:
    """Checkpointable global shrinkage state and token-level draft estimator."""

    DEFAULTS = {
        "num_drafts": 2,
        "num_continuations_per_draft": 2,
        "ema_beta": 0.9,
        "shrink_min": 0.1,
        "eps": 1.0e-6,
    }

    def __init__(self, config: Mapping | None = None):
        config = {} if config is None else config
        self.config = {key: config.get(key, default) for key, default in self.DEFAULTS.items()}
        for key in ("num_drafts", "num_continuations_per_draft"):
            if self.config[key] != 2:
                raise ValueError(f"This estimator requires {key}=2")
            self.config[key] = 2
        for key in ("ema_beta", "shrink_min", "eps"):
            self.config[key] = float(self.config[key])
            if not math.isfinite(self.config[key]):
                raise ValueError(f"{key} must be finite")
        if not 0 <= self.config["ema_beta"] < 1:
            raise ValueError("ema_beta must be in [0, 1)")
        if not 0 <= self.config["shrink_min"] <= 1:
            raise ValueError("shrink_min must be in [0, 1]")
        if self.config["eps"] <= 0:
            raise ValueError("eps must be positive")
        # Explicit zero initialization; no bias correction is needed for the ratio.
        self.ema_ms_b = torch.zeros((), dtype=torch.bfloat16)
        self.ema_ms_w = torch.zeros((), dtype=torch.bfloat16)
        self.num_updates = 0

    def state_dict(self) -> dict[str, Any]:
        """Return JSON-serializable state, including the estimator contract."""
        return {
            "version": 1,
            "config": dict(self.config),
            "ema_ms_b": float(self.ema_ms_b.item()),
            "ema_ms_w": float(self.ema_ms_w.item()),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: Mapping) -> None:
        """Restore state without silently accepting a different EMA recipe."""
        if state.get("version") != 1:
            raise ValueError("Unsupported draft advantage state version")
        if state.get("config") != self.config:
            raise ValueError("Draft advantage checkpoint config does not match current config")
        new_values = []
        for key in ("ema_ms_b", "ema_ms_w"):
            value = float(state[key])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {key} in draft advantage checkpoint")
            new_values.append(torch.tensor(value, dtype=torch.bfloat16))
        count = operator.index(state["num_updates"])
        if count < 0:
            raise ValueError("num_updates cannot be negative")
        self.ema_ms_b, self.ema_ms_w = new_values
        self.num_updates = count

    @staticmethod
    def _image_group_rows(batch, responses: torch.Tensor, response_mask: torch.Tensor) -> list[list[list[int]]]:
        """Recover complete image groups by IDs, independently of the batch row order."""
        metadata = batch.non_tensor_batch
        keys = ("image_id", "draft_id", "branch_id", "draft_token_count")
        row_count = responses.shape[0]
        for key in keys:
            if key not in metadata or len(metadata[key]) != row_count:
                raise ValueError(f"Draft rollout requires one {key} per response")
        images = {}
        counts = []
        active_lengths = response_mask.sum(dim=-1).tolist()
        for row in range(row_count):
            image_id, draft_id, branch_id = (metadata[key][row] for key in keys[:3])
            try:
                branches = images.setdefault(image_id, {}).setdefault(draft_id, {})
                if branch_id in branches:
                    raise ValueError(f"Duplicate draft continuation for image {image_id!r}, draft {draft_id!r}")
                branches[branch_id] = row
            except TypeError as error:
                raise ValueError("Image, draft and branch IDs must be hashable scalars") from error
            try:
                count = operator.index(metadata["draft_token_count"][row])
            except TypeError as error:
                raise ValueError("draft_token_count must be an integer") from error
            if not 0 < count < active_lengths[row]:
                raise ValueError("draft_token_count must leave a nonempty draft and suffix in every response")
            counts.append(count)
        image_groups = []
        for image_id, drafts in images.items():
            if len(drafts) != 2 or any(len(branches) != 2 for branches in drafts.values()):
                raise ValueError(f"Image {image_id!r} must have exactly 2 drafts and 2 branches per draft")
            image_group = []
            for branches in drafts.values():
                rows = list(branches.values())
                first, second = rows
                prefix_size = counts[first]
                if counts[second] != prefix_size or not torch.equal(
                    responses[first, :prefix_size], responses[second, :prefix_size]
                ):
                    raise ValueError(f"Branches for image {image_id!r} must share an identical draft prefix")
                image_group.append(rows)
            image_groups.append(image_group)
        if not image_groups:
            raise ValueError("Draft advantage requires at least one complete image group")
        return image_groups

    @torch.no_grad()
    def compute(self, batch, *, global_batch: bool = True, process_group=None):
        """Fill ``advantages``/``returns`` and return ``(batch, metrics)``.

        ``token_level_scores`` must be the pure binary classification reward;
        KL-shaped ``token_level_rewards`` are intentionally never read. The
        caller retains any separately configured PPO KL loss. All response
        tokens up to ``draft_token_count`` receive draft advantage, while every
        remaining active token (including EOS) receives suffix advantage.
        """
        tensors = batch.batch
        responses = tensors["responses"]
        response_mask = tensors["response_mask"]
        scores = tensors["token_level_scores"]
        if responses.ndim != 2 or response_mask.shape != responses.shape or scores.shape != responses.shape:
            raise ValueError("responses, response_mask and token_level_scores must have matching [N, T] shapes")
        if responses.device != response_mask.device or responses.device != scores.device:
            raise ValueError("Response tensors must be on the same device")
        if not torch.all((response_mask == 0) | (response_mask == 1)):
            raise ValueError("response_mask must be binary")
        active = response_mask.bool()
        lengths = active.sum(dim=-1)
        expected_mask = torch.arange(responses.shape[1], device=responses.device)[None, :] < lengths[:, None]
        if not torch.equal(active, expected_mask):
            raise ValueError("Responses must have contiguous active tokens followed by right padding")
        if not torch.isfinite(scores).all() or torch.any(scores.masked_select(~active) != 0):
            raise ValueError("Task rewards must be finite and zero on padding")
        # Check before casting: BF16 must not round a fractional reward to 0/1.
        reward_per_row = scores.detach().sum(dim=-1)
        if not torch.all((reward_per_row == 0) | (reward_per_row == 1)):
            raise ValueError("Draft variance statistics require binary classification rewards (0 or 1)")
        image_groups = self._image_group_rows(batch, responses, active)
        row_indices = torch.tensor(image_groups, dtype=torch.long, device=responses.device)
        rewards = reward_per_row[row_indices].to(dtype=torch.bfloat16)
        draft_mean = rewards.mean(dim=2)
        image_mean = draft_mean.mean(dim=1)
        # Center within each image before summing across images/ranks.
        totals = torch.stack(
            (
                (draft_mean - image_mean[:, None]).square().sum(),
                (rewards - draft_mean[:, :, None]).square().sum(),
                rewards.sum(),
            )
        )
        image_count = torch.tensor(len(image_groups), dtype=torch.int64, device=responses.device)
        if not global_batch:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("global_batch=False requires an initialized torch.distributed process group")
            dist.all_reduce(totals, op=dist.ReduceOp.SUM, group=process_group)
            # Integer counts remain exact even for large or unequal rank shards.
            dist.all_reduce(image_count, op=dist.ReduceOp.SUM, group=process_group)
        denominator = image_count.to(dtype=torch.bfloat16)
        ms_b = 2 * totals[0] / denominator
        ms_w = totals[1] / (2 * denominator)
        beta = self.config["ema_beta"]
        ema_ms_b = beta * self.ema_ms_b.to(responses.device) + (1 - beta) * ms_b
        ema_ms_w = beta * self.ema_ms_w.to(responses.device) + (1 - beta) * ms_w
        shrink = (1 - ema_ms_w / ema_ms_b.clamp_min(self.config["eps"])).clamp(
            min=self.config["shrink_min"], max=1.0
        )
        segment = compute_draft_advantages(rewards, shrink)
        advantages = torch.zeros(responses.shape, dtype=torch.bfloat16, device=responses.device)
        for image_index, image_group in enumerate(image_groups):
            for draft_index, branches in enumerate(image_group):
                for branch_index, row in enumerate(branches):
                    count = operator.index(batch.non_tensor_batch["draft_token_count"][row])
                    advantages[row, :count] = segment["draft_adv"][image_index, draft_index]
                    advantages[row, count:] = segment["suffix_adv"][image_index, draft_index, branch_index]
        advantages.masked_fill_(~active, 0)
        absolute_sums = torch.stack((segment["draft_adv"].abs().sum(), segment["suffix_adv"].abs().sum()))
        if not global_batch:
            dist.all_reduce(absolute_sums, op=dist.ReduceOp.SUM, group=process_group)
        # Commit the update only after all metadata and decomposition checks pass.
        self.ema_ms_b = ema_ms_b
        self.ema_ms_w = ema_ms_w
        self.num_updates += 1
        tensors["advantages"] = advantages
        tensors["returns"] = advantages.clone()
        metrics = {
            "draft/reward_mean": float((totals[2] / (4 * denominator)).item()),
            "draft/MS_B": float(ms_b.item()),
            "draft/MS_W": float(ms_w.item()),
            "draft/s": float(shrink.item()),
            "draft/mean_abs_draft_adv": float((absolute_sums[0] / (2 * denominator)).item()),
            "draft/mean_abs_suffix_adv": float((absolute_sums[1] / (4 * denominator)).item()),
            "draft/ema_ms_b": float(ema_ms_b.item()),
            "draft/ema_ms_w": float(ema_ms_w.item()),
            "draft/num_images": int(image_count.item()),
        }
        return batch, metrics
