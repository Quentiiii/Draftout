"""Regressions for bounded-memory entropy in padded old-policy evaluation."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from verl import DataProto
from verl.utils import torch_functional as verl_F
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.config.actor import FSDPActorConfig


class TinyEntropyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Embedding(13, 13)
        with torch.no_grad():
            self.projection.weight.copy_(torch.arange(169).reshape(13, 13).float().sin())

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.projection(input_ids))


class ChunkedEntropyTests(unittest.TestCase):
    def test_matches_dense_formula_for_noncontiguous_response_logits(self):
        generator = torch.Generator().manual_seed(42)
        for dtype in (torch.float32, torch.bfloat16):
            for strided_vocabulary in (False, True):
                with self.subTest(dtype=dtype, strided_vocabulary=strided_vocabulary):
                    base = torch.randn(3, 19, 34, generator=generator).to(dtype)
                    logits = base[:, 2:17, ::2] if strided_vocabulary else base[:, 2:17, :]
                    self.assertFalse(logits.is_contiguous())
                    before = base.clone()
                    expected = verl_F.entropy_from_logits(logits)
                    actual = verl_F.entropy_from_logits_chunked_no_grad(logits, chunk_size=4)
                    self.assertEqual(actual.shape, logits.shape[:-1])
                    self.assertEqual(actual.dtype, expected.dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(base, before, rtol=0, atol=0)

    def test_default_chunks_bound_softmax_and_share_original_storage(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                base = torch.randn(3, 604, 37).to(dtype).requires_grad_()
                logits = base[:, 3:-4, :]
                self.assertFalse(logits.is_contiguous())
                original_softmax = torch.nn.functional.softmax
                storage_pointer = base.untyped_storage().data_ptr()
                chunk_token_counts = []

                def bounded_softmax(chunk, *args, **kwargs):
                    token_count = chunk.numel() // chunk.shape[-1]
                    chunk_token_counts.append(token_count)
                    self.assertLessEqual(token_count, 256)
                    self.assertEqual(chunk.untyped_storage().data_ptr(), storage_pointer)
                    self.assertEqual(chunk.dtype, dtype)
                    self.assertFalse(torch.is_grad_enabled())
                    return original_softmax(chunk, *args, **kwargs)

                with patch("torch.nn.functional.softmax", side_effect=bounded_softmax):
                    actual = verl_F.entropy_from_logits_chunked_no_grad(logits)

                self.assertGreater(len(chunk_token_counts), 1)
                self.assertEqual(sum(chunk_token_counts), 3 * 597)
                self.assertFalse(actual.requires_grad)
                self.assertIsNone(base.grad)
                with torch.no_grad():
                    expected = verl_F.entropy_from_logits(logits)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_empty_token_dimensions_and_single_distribution(self):
        for dtype in (torch.float32, torch.bfloat16):
            for shape in ((0, 7), (2, 0, 7), (0, 3, 7), (0, 0, 7), (7,)):
                with self.subTest(dtype=dtype, shape=shape):
                    logits = torch.randn(shape, dtype=dtype)
                    expected = verl_F.entropy_from_logits(logits)
                    actual = verl_F.entropy_from_logits_chunked_no_grad(logits, chunk_size=2)
                    self.assertEqual(actual.shape, expected.shape)
                    self.assertEqual(actual.dtype, expected.dtype)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class PaddedActorEntropyTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        for target, value in (
            ("torch.distributed.get_rank", 0),
            ("verl.workers.actor.dp_actor.get_device_name", "cpu"),
            ("verl.workers.actor.dp_actor.get_device_id", torch.device("cpu")),
            ("verl.utils.profiler.performance.get_torch_device", torch.cpu),
        ):
            self.patches.enter_context(patch(target, return_value=value))
        self.patches.enter_context(patch("verl.utils.torch_functional.FLAH_ATTN_CROSS_ENTROPY_LOSS_AVAILABLE", False))
        self.patches.enter_context(patch("verl.utils.torch_functional.NPU_CROSS_ENTROPY_LOSS_AVAILABLE", False))
        self.config = FSDPActorConfig(
            rollout_n=2,
            ppo_mini_batch_size=2,
            ppo_micro_batch_size_per_gpu=2,
            use_torch_compile=False,
            use_fused_kernels=False,
        )
        self.model = TinyEntropyPolicy()
        self.actor = DataParallelPPOActor(self.config, self.model)
        self.input_ids = torch.tensor([[1, 2, 3, 4, 5], [1, 2, 6, 7, 8]])
        self.batch = {
            "input_ids": self.input_ids,
            "attention_mask": torch.ones_like(self.input_ids),
            "position_ids": torch.arange(5).expand(2, -1),
            "responses": self.input_ids[:, -2:],
        }

    def reference(self, temperature):
        logits = self.model(self.input_ids).logits[:, -3:-1, :] / temperature
        log_probs = logits.log_softmax(-1).gather(-1, self.batch["responses"].unsqueeze(-1)).squeeze(-1)
        return log_probs, verl_F.entropy_from_logits(logits)

    def test_real_compute_log_prob_uses_chunked_entropy(self):
        temperature = 0.7
        data = DataProto.from_dict(
            tensors=self.batch,
            meta_info={"temperature": temperature, "micro_batch_size": 2, "use_dynamic_bsz": False},
        )
        with patch.object(
            verl_F,
            "entropy_from_logits_chunked_no_grad",
            wraps=verl_F.entropy_from_logits_chunked_no_grad,
        ) as chunked:
            outputs = self.actor.compute_log_prob(data, calculate_entropy=True)
        chunked.assert_called()
        with torch.no_grad():
            expected_log_probs, expected_entropy = self.reference(temperature)
        torch.testing.assert_close(outputs["log_probs"], expected_log_probs)
        torch.testing.assert_close(outputs["entropys"], expected_entropy)
        self.assertFalse(outputs["log_probs"].requires_grad)
        self.assertFalse(outputs["entropys"].requires_grad)

    def test_grad_enabled_entropy_keeps_original_values_and_gradients(self):
        temperature = 0.7
        with torch.enable_grad(), patch.object(
            verl_F,
            "entropy_from_logits_chunked_no_grad",
            side_effect=AssertionError("The detached entropy helper must not run during gradient computation"),
        ) as chunked:
            outputs = self.actor._forward_micro_batch(self.batch, temperature=temperature, calculate_entropy=True)
            actual_loss = -(outputs["log_probs"].mean() + 0.2 * outputs["entropys"].mean())
            actual_gradient = torch.autograd.grad(actual_loss, self.model.projection.weight)[0]
            expected_log_probs, expected_entropy = self.reference(temperature)
            expected_loss = -(expected_log_probs.mean() + 0.2 * expected_entropy.mean())
            expected_gradient = torch.autograd.grad(expected_loss, self.model.projection.weight)[0]
        chunked.assert_not_called()
        torch.testing.assert_close(outputs["log_probs"], expected_log_probs)
        torch.testing.assert_close(outputs["entropys"], expected_entropy)
        self.assertGreater(actual_gradient.abs().sum().item(), 0)
        torch.testing.assert_close(actual_gradient, expected_gradient)


if __name__ == "__main__":
    unittest.main()
