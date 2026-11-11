"""Driver-side draft advantages with fresh-start GRPO warmup and optional rewarmup."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from emotion_grpo.draft_advantage import DraftAdvantageEstimator


class DraftTrainingController:
    state_filename = "draft_advantage_state.json"

    def __init__(self, config):
        self.warmup_steps = int(config.get("warmup_steps", 20))
        if self.warmup_steps != 20:
            raise ValueError("This recipe requires 20 ordinary GRPO updates when warmup is enabled.")
        self.warmup_on_resume = config.get("warmup_on_resume", False)
        if not isinstance(self.warmup_on_resume, bool):
            raise ValueError("warmup_on_resume must be a bool")
        self.estimator = DraftAdvantageEstimator(config)
        self.session_start_step = None

    def begin_session(self, completed_global_step: int) -> None:
        # Track this launch separately; checkpoint EMA state remains untouched.
        self.session_start_step = int(completed_global_step)

    def is_warmup(self, global_step: int) -> bool:
        if self.session_start_step is None:
            raise RuntimeError("begin_session must be called after checkpoint loading")
        session_step = int(global_step) - self.session_start_step
        if session_step <= 0:
            raise ValueError("Training global_step must follow the resumed checkpoint step")
        if self.session_start_step > 0 and not self.warmup_on_resume:
            return False
        return session_step <= self.warmup_steps

    def prepare_batch(self, batch, global_step: int) -> None:
        warmup = self.is_warmup(global_step)
        batch.meta_info["draft_rollout"] = not warmup
        # RewardManager passes extra_info, not batch.meta_info, to the reward fn.
        # Repeated dataset rows can share dictionaries: never mutate them in place.
        original = batch.non_tensor_batch.get("extra_info", [{} for _ in range(len(batch))])
        extras = np.empty(len(batch), dtype=object)
        extras[:] = [dict(info or {}, draft_grpo_warmup=warmup) for info in original]
        batch.non_tensor_batch["extra_info"] = extras

    @torch.no_grad()
    def compute_advantage(self, batch, global_step: int):
        warmup = self.is_warmup(global_step)
        if warmup:
            from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

            advantages, returns = compute_grpo_outcome_advantage(
                token_level_rewards=batch.batch["token_level_rewards"],
                response_mask=batch.batch["response_mask"],
                index=batch.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=True,
            )
            batch.batch["advantages"] = advantages.to(torch.bfloat16)
            batch.batch["returns"] = returns.to(torch.bfloat16)
            # Warmup neither estimates draft statistics nor advances their EMA.
            metrics = {}
        else:
            # Ray gathers *all* rollout workers before this driver-side call.
            # _balance_batch may reorder the leaves; estimator reconstructs IDs.
            batch, metrics = self.estimator.compute(batch, global_batch=True)
        metrics.update({
            "draft/warmup": float(warmup),
            "draft/session_step": int(global_step) - self.session_start_step,
        })
        return batch, metrics

    def save(self, checkpoint_dir, global_step: int) -> None:
        path = Path(checkpoint_dir) / self.state_filename
        payload = {
            "version": 1,
            "global_step": int(global_step),
            "warmup_steps_per_launch": self.warmup_steps,
            "warmup_on_resume": self.warmup_on_resume,
            "estimator": self.estimator.state_dict(),
        }
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def load(self, checkpoint_dir, global_step: int) -> None:
        path = Path(checkpoint_dir) / self.state_filename
        if not path.is_file():
            raise FileNotFoundError(f"Draft checkpoint is missing EMA state: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("version") != 1 or payload.get("global_step") != int(global_step)
                or payload.get("warmup_steps_per_launch") != self.warmup_steps):
            raise ValueError(f"Draft EMA checkpoint metadata mismatch: {path}")
        # warmup_on_resume is a launch policy, not an EMA compatibility field.
        # Older checkpoints omit it; current configuration governs either case.
        self.estimator.load_state_dict(payload["estimator"])
