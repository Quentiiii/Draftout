"""CPU regression for Qwen3.5's padded, 16-response multimodal microbatch."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

from verl.models.transformers.monkey_patch import apply_monkey_patch


class QwenPaddedBatchTests(unittest.TestCase):
    def test_batch16_matches_independent_images_and_backpropagates(self):
        # Use Transformers' real CPU fallbacks, avoiding installed CUDA-only FLA
        # kernels. Keep both kinds of attention present in the actual 4B model.
        with ExitStack() as stack:
            for name in (
                "FusedRMSNormGated",
                "causal_conv1d_fn",
                "causal_conv1d_update",
                "chunk_gated_delta_rule",
                "fused_recurrent_gated_delta_rule",
            ):
                stack.enter_context(patch.object(qwen, name, None))
            # apply_monkey_patch changes class methods globally; restore them so
            # this test does not change other tests' model implementation.
            for cls, method in (
                (qwen.Qwen3_5Model, "forward"),
                (qwen.Qwen3_5ForConditionalGeneration, "forward"),
                (qwen.Qwen3_5VisionModel, "fast_pos_embed_interpolate"),
            ):
                stack.enter_context(patch.object(cls, method, getattr(cls, method)))
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            stack.callback(torch.set_num_threads, previous_threads)
            stack.enter_context(torch.random.fork_rng(devices=[]))
            torch.manual_seed(42)

            config = Qwen3_5Config(
                image_token_id=3,
                video_token_id=4,
                vision_start_token_id=5,
                vision_end_token_id=6,
                text_config={
                    "vocab_size": 64,
                    "hidden_size": 32,
                    "intermediate_size": 48,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 2,
                    "head_dim": 8,
                    "linear_key_head_dim": 8,
                    "linear_value_head_dim": 8,
                    "linear_num_key_heads": 2,
                    "linear_num_value_heads": 4,
                    "layer_types": ["linear_attention", "full_attention"],
                    "pad_token_id": 0,
                    "use_cache": False,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 1.0,
                        "mrope_section": [2, 1, 1],
                    },
                },
                vision_config={
                    "depth": 1,
                    "hidden_size": 32,
                    "intermediate_size": 48,
                    "num_heads": 4,
                    "patch_size": 2,
                    "temporal_patch_size": 1,
                    "spatial_merge_size": 2,
                    "out_hidden_size": 32,
                    "num_position_embeddings": 16,
                },
            )
            config._attn_implementation = "eager"
            model = qwen.Qwen3_5ForConditionalGeneration(config)
            apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=False)
            model.train()

            batch_size, sequence_length, patches_per_image = 16, 18, 16
            input_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)
            attention_mask = torch.zeros_like(input_ids)
            # Different prompt lengths (left padding) and response lengths
            # (right padding) reproduce the actor's prompt+response layout.
            for row in range(batch_size):
                tokens = [1, 5, 3, 3, 3, 3, 6, 7 + row, 23 + row]
                tokens += [40 + row % 8] * (row % 4)
                tokens += [2]
                left_padding = row % 3
                input_ids[row, left_padding : left_padding + len(tokens)] = torch.tensor(tokens)
                attention_mask[row, left_padding : left_padding + len(tokens)] = 1
            grid = torch.tensor([[1, 4, 4]]).repeat(batch_size, 1)
            pixels = torch.randn(batch_size * patches_per_image, 3 * 1 * 2**2)
            vision_positions, _ = model.model.get_rope_index(
                input_ids=input_ids,
                mm_token_type_ids=(input_ids == config.image_token_id).long(),
                image_grid_thw=grid,
                attention_mask=attention_mask,
            )
            text_positions = (attention_mask.cumsum(-1) - 1).clamp_min(0)
            position_ids = torch.cat([text_positions.unsqueeze(0), vision_positions], dim=0)
            self.assertEqual(position_ids.shape, (4, batch_size, sequence_length))
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "pixel_values": pixels,
                "image_grid_thw": grid,
                "use_cache": False,
            }
            logits = model(**inputs).logits
            self.assertEqual(logits.shape, (batch_size, sequence_length, config.text_config.vocab_size))
            self.assertTrue(torch.isfinite(logits).all())

            # A padded batch must give each image the same valid-token logits as
            # its isolated unpadded sequence: no cross-example linear state.
            with torch.no_grad():
                for row in range(batch_size):
                    valid = attention_mask[row].bool()
                    reference = model(
                        input_ids=input_ids[row : row + 1, valid],
                        attention_mask=torch.ones(1, int(valid.sum()), dtype=torch.long),
                        position_ids=position_ids[:, row : row + 1, valid],
                        pixel_values=pixels[row * patches_per_image : (row + 1) * patches_per_image],
                        image_grid_thw=grid[row : row + 1],
                        use_cache=False,
                    ).logits
                    torch.testing.assert_close(logits[row, valid], reference[0], atol=2e-6, rtol=2e-5)

            valid_predictions = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
            loss = F.cross_entropy(logits[:, :-1][valid_predictions], input_ids[:, 1:][valid_predictions])
            loss.backward()
            for module in (
                model.model.visual,
                model.model.language_model.layers[0],
                model.model.language_model.layers[1],
                model.lm_head,
            ):
                gradients = [parameter.grad for parameter in module.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(gradient.abs().sum().item() for gradient in gradients), 0.0)

            # Also exercise BF16 parameters and autocast, matching the default
            # training dtype without loading checkpoint weights or using a GPU.
            model.zero_grad(set_to_none=True)
            model.to(dtype=torch.bfloat16)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                bf16_logits = model(**inputs).logits
                bf16_loss = F.cross_entropy(
                    bf16_logits[:, :-1][valid_predictions], input_ids[:, 1:][valid_predictions]
                )
            self.assertEqual(bf16_logits.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(bf16_loss))
            bf16_loss.backward()
            self.assertTrue(torch.isfinite(model.lm_head.weight.grad).all())
            self.assertGreater(model.lm_head.weight.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
