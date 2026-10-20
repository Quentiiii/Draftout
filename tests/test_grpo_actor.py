"""CPU regression for the real vanilla actor forward, loss, and optimizer path."""

import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig, SelfDistillationConfig


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Embedding(8, 8)
        torch.nn.init.zeros_(self.projection.weight)

    def forward(self, input_ids, **kwargs):
        logits = self.projection(input_ids)
        # Also implement the fused forward interface using CPU operations.
        log_distribution = logits.log_softmax(dim=-1)
        next_tokens = input_ids.roll(shifts=-1, dims=1)
        return SimpleNamespace(
            logits=logits,
            log_probs=log_distribution.gather(-1, next_tokens.unsqueeze(-1)).squeeze(-1),
            entropy=-(log_distribution.exp() * log_distribution).sum(-1),
        )


class VanillaActorTests(unittest.TestCase):
    def test_vanilla_updates_without_teacher_or_distillation_logits(self):
        variants = (
            ("no_distillation_config", None, False),
            ("default_distillation_config", SelfDistillationConfig(), False),
            ("topk_distillation_config", SelfDistillationConfig(distillation_topk=2), False),
            (
                "fused_with_unused_trust_region_config",
                SelfDistillationConfig(teacher_regularization="trust-region"),
                True,
            ),
        )
        for name, distillation_config, fused in variants:
            with (
                self.subTest(name=name),
                patch("torch.distributed.get_rank", return_value=0),
                patch("verl.workers.actor.dp_actor.get_device_name", return_value="cpu"),
                patch("verl.workers.actor.dp_actor.get_device_id", return_value=torch.device("cpu")),
                patch("verl.utils.profiler.performance.get_torch_device", return_value=torch.cpu),
                patch("verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False),
                patch("verl.utils.torch_functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False),
            ):
                config = FSDPActorConfig(
                    rollout_n=2,
                    ppo_mini_batch_size=2,
                    ppo_micro_batch_size_per_gpu=1,
                    ppo_epochs=1,
                    self_distillation=distillation_config,
                    use_torch_compile=False,
                    use_fused_kernels=fused,
                    calculate_entropy=False,
                    entropy_coeff=0.0,
                    use_kl_loss=False,
                )
                model = TinyPolicy()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                actor = DataParallelPPOActor(config, model, optimizer)
                before = model.projection.weight.detach().clone()
                input_ids = torch.tensor([[1, 2, 3, 4], [1, 2, 5, 6]])
                batch = DataProto.from_dict(
                    tensors={
                        "input_ids": input_ids,
                        "attention_mask": torch.ones_like(input_ids),
                        "position_ids": torch.arange(4).expand(2, -1),
                        "responses": input_ids[:, -2:],
                        "response_mask": torch.ones(2, 2),
                        "old_log_probs": torch.full((2, 2), -math.log(8)),
                        # Two sampled responses from one GRPO group: correct / wrong.
                        "advantages": torch.tensor([[1.0, 1.0], [-1.0, -1.0]]),
                    },
                    meta_info={"temperature": 1.0, "pad_token_id": 0},
                )
                with patch.object(actor, "_forward_micro_batch", wraps=actor._forward_micro_batch) as forward:
                    metrics = actor.update_policy(batch)

                self.assertEqual(forward.call_count, 2)
                for call in forward.call_args_list:
                    self.assertFalse(call.kwargs["return_all_logps"])
                    self.assertIsNone(call.kwargs["distill_topk"])
                    self.assertNotIn("module", call.kwargs)
                self.assertFalse(torch.equal(before, model.projection.weight.detach()))
                self.assertGreater(metrics["actor/grad_norm"], 0.0)
                self.assertTrue(math.isfinite(metrics["actor/pg_loss"]))
                self.assertEqual(metrics["actor/kl_loss"], 0.0)
                self.assertNotIn("actor/vopd_loss", metrics)
                self.assertIsNone(actor.teacher_module)


if __name__ == "__main__":
    unittest.main()
