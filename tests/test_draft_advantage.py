"""CPU correctness checks for the 2 x 2 draft advantage estimator."""

import itertools
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from emotion_grpo.draft_advantage import DraftAdvantageEstimator, compute_draft_advantages


def make_batch(rewards, *, order=None):
    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    image_count = rewards.shape[0]
    rows = image_count * 4
    responses = torch.zeros((rows, 8), dtype=torch.long)
    masks = torch.zeros_like(responses)
    scores = torch.zeros((rows, 8), dtype=torch.float32)
    metadata = {key: [] for key in ("image_id", "draft_id", "branch_id", "draft_token_count")}
    for image, draft, branch in itertools.product(range(image_count), range(2), range(2)):
        row = image * 4 + draft * 2 + branch
        draft_count = 2 + draft
        active_count = 6 + branch
        responses[row, :draft_count] = torch.arange(draft_count) + 10 * image + 3 * draft + 1
        responses[row, draft_count:active_count] = 50 + branch
        responses[row, active_count - 1] = 99  # EOS is an active suffix token.
        masks[row, :active_count] = 1
        scores[row, active_count - 1] = rewards[image, draft, branch]
        for key, value in zip(metadata, (f"image-{image}", draft, branch, draft_count), strict=True):
            metadata[key].append(value)
    if order is None:
        order = list(range(rows))
    return SimpleNamespace(
        batch={
            "responses": responses[order],
            "response_mask": masks[order],
            "token_level_scores": scores[order],
            # Deliberately unrelated: estimator must never use KL-shaped rewards.
            "token_level_rewards": torch.full((rows, 8), -123.0),
            "input_ids": torch.full((rows, 11), 77, dtype=torch.long),
        },
        non_tensor_batch={key: np.array(values, dtype=object)[order] for key, values in metadata.items()},
    )


class SegmentFormulaTests(unittest.TestCase):
    def test_draft_success_example(self):
        result = compute_draft_advantages(torch.tensor([[[1, 1], [0, 0]]]), 1)
        torch.testing.assert_close(result["draft_adv"], torch.tensor([[0.5, -0.5]], dtype=torch.bfloat16))
        torch.testing.assert_close(result["suffix_adv"], torch.zeros((1, 2, 2), dtype=torch.bfloat16))

    def test_equal_draft_means_example_for_any_shrink(self):
        for shrink in (0.0, 0.1, 0.33, 0.9, 1.0):
            with self.subTest(shrink=shrink):
                result = compute_draft_advantages(torch.tensor([[[1, 0], [1, 0]]]), shrink)
                torch.testing.assert_close(result["draft_adv"], torch.zeros((1, 2), dtype=torch.bfloat16))
                expected = torch.tensor([[[0.5, -0.5], [0.5, -0.5]]], dtype=torch.bfloat16)
                torch.testing.assert_close(result["suffix_adv"], expected)

    def test_every_binary_reward_group_satisfies_branch_decomposition_without_gradients(self):
        rewards = torch.tensor(list(itertools.product((0.0, 1.0), repeat=4))).reshape(-1, 2, 2)
        rewards.requires_grad_()
        for shrink in (0, 0.1, 0.3, 0.7, 1):
            result = compute_draft_advantages(rewards, torch.tensor(shrink, dtype=torch.float32, requires_grad=True))
            for value in result.values():
                self.assertEqual(value.dtype, torch.bfloat16)
                self.assertFalse(value.requires_grad)
            expected = rewards.detach().to(torch.bfloat16) - result["image_mean"][:, None, None]
            torch.testing.assert_close(
                result["draft_adv"][:, :, None] + result["suffix_adv"],
                expected,
                rtol=0,
                atol=torch.finfo(torch.bfloat16).eps,
            )
        self.assertIsNone(rewards.grad)


class DraftEstimatorTests(unittest.TestCase):
    def test_stats_center_within_images_instead_of_across_image_difficulty(self):
        batch = make_batch([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
        result, metrics = DraftAdvantageEstimator().compute(batch)
        self.assertIs(result, batch)
        self.assertEqual(metrics["draft/MS_B"], 0)
        self.assertEqual(metrics["draft/MS_W"], 0)
        self.assertEqual(metrics["draft/reward_mean"], 0.5)
        self.assertEqual(metrics["draft/s"], 1)
        self.assertEqual(torch.count_nonzero(batch.batch["advantages"]).item(), 0)

    def test_zero_and_within_only_variance_are_finite(self):
        for rewards, expected_ms_w, expected_shrink in (
            ([[[0, 0], [0, 0]]], 0, 1),
            ([[[1, 0], [1, 0]]], 0.5, float(torch.tensor(0.1, dtype=torch.bfloat16))),
        ):
            with self.subTest(rewards=rewards):
                batch, metrics = DraftAdvantageEstimator().compute(make_batch(rewards))
                self.assertEqual(metrics["draft/MS_B"], 0)
                self.assertEqual(metrics["draft/MS_W"], expected_ms_w)
                self.assertEqual(metrics["draft/s"], expected_shrink)
                self.assertTrue(torch.isfinite(batch.batch["advantages"]).all())

    def test_exact_stats_and_raw_advantage_not_standardized(self):
        estimator = DraftAdvantageEstimator({"ema_beta": 0})
        batch, metrics = estimator.compute(make_batch([[[1, 1], [0, 0]], [[1, 0], [0, 0]]]))
        self.assertEqual(metrics["draft/MS_B"], 0.625)
        self.assertEqual(metrics["draft/MS_W"], 0.125)
        self.assertEqual(metrics["draft/reward_mean"], 0.375)
        self.assertAlmostEqual(metrics["draft/s"], 0.8, delta=0.004)
        expected = compute_draft_advantages(torch.tensor([[[1, 1], [0, 0]], [[1, 0], [0, 0]]]), metrics["draft/s"])
        self.assertEqual(float(batch.batch["advantages"][0, 0]), float(expected["draft_adv"][0, 0]))
        self.assertLess(float(batch.batch["advantages"][0, 0]), 0.5)
        self.assertEqual(estimator.num_updates, 1)
        self.assertEqual(estimator.ema_ms_b.dtype, torch.bfloat16)
        self.assertEqual(estimator.ema_ms_w.dtype, torch.bfloat16)

    def test_reordered_rows_repeat_prefix_advantage_and_mask_padding_including_eos(self):
        rewards = [[[1, 1], [0, 0]], [[1, 0], [1, 0]]]
        canonical, metrics = DraftAdvantageEstimator().compute(make_batch(rewards))
        order = [7, 2, 0, 5, 3, 4, 1, 6]
        shuffled = make_batch(rewards, order=order)
        prompt_before = shuffled.batch["input_ids"].clone()
        shuffled.batch["token_level_scores"].requires_grad_()
        shuffled, shuffled_metrics = DraftAdvantageEstimator().compute(shuffled)
        torch.testing.assert_close(shuffled.batch["advantages"], canonical.batch["advantages"][order], rtol=0, atol=0)
        self.assertEqual(metrics, shuffled_metrics)
        torch.testing.assert_close(shuffled.batch["input_ids"], prompt_before)
        self.assertEqual(shuffled.batch["advantages"].dtype, torch.bfloat16)
        self.assertFalse(shuffled.batch["advantages"].requires_grad)
        torch.testing.assert_close(shuffled.batch["returns"], shuffled.batch["advantages"], rtol=0, atol=0)
        for row in range(8):
            length = int(shuffled.batch["response_mask"][row].sum())
            adv = shuffled.batch["advantages"][row]
            self.assertEqual(float(adv[length - 1]), float(adv[length - 2]))
            self.assertEqual(float(adv[length:].abs().sum()), 0)
        # Identical duplicated draft tokens receive identical advantages.
        for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
            count = canonical.non_tensor_batch["draft_token_count"][first]
            torch.testing.assert_close(
                canonical.batch["advantages"][first, :count],
                canonical.batch["advantages"][second, :count],
                rtol=0,
                atol=0,
            )

    def test_metadata_shape_duplicates_and_prefix_errors_fail_without_updating_ema(self):
        def missing_field(batch):
            del batch.non_tensor_batch["draft_id"]

        def duplicate_branch(batch):
            batch.non_tensor_batch["branch_id"][1] = 0

        def third_draft(batch):
            batch.non_tensor_batch["draft_id"][3] = 2

        def changed_prefix(batch):
            batch.batch["responses"][1, 0] += 1

        def changed_count(batch):
            batch.non_tensor_batch["draft_token_count"][1] += 1

        def no_suffix(batch):
            batch.non_tensor_batch["draft_token_count"][0] = 6

        def fractional_count(batch):
            batch.non_tensor_batch["draft_token_count"][0] = 2.5

        mutations = (
            missing_field, duplicate_branch, third_draft, changed_prefix, changed_count, no_suffix, fractional_count,
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation.__name__):
                batch = make_batch([[[1, 1], [0, 0]]])
                mutation(batch)
                estimator = DraftAdvantageEstimator()
                with self.assertRaises(ValueError):
                    estimator.compute(batch)
                self.assertEqual(estimator.num_updates, 0)
                self.assertEqual(float(estimator.ema_ms_b), 0)

    def test_nonbinary_rewards_and_noncontiguous_or_scored_padding_rejected(self):
        for problem in ("fractional", "nan", "noncontiguous", "padding", "mask_fractional"):
            with self.subTest(problem=problem):
                batch = make_batch([[[1, 1], [0, 0]]])
                if problem == "fractional":
                    batch.batch["token_level_scores"][0, 5] = 0.999
                elif problem == "nan":
                    batch.batch["token_level_scores"][0, 5] = float("nan")
                elif problem == "padding":
                    batch.batch["token_level_scores"][0, 7] = 1
                elif problem == "mask_fractional":
                    batch.batch["response_mask"] = batch.batch["response_mask"].float()
                    batch.batch["response_mask"][0, 0] = 0.5
                else:
                    batch.batch["response_mask"][0, 0] = 0
                with self.assertRaises(ValueError):
                    DraftAdvantageEstimator().compute(batch)

    def test_json_checkpoint_round_trip_preserves_next_ema_and_advantages(self):
        estimator = DraftAdvantageEstimator()
        estimator.compute(make_batch([[[1, 1], [0, 0]]]))
        # The EMA starts at zero, rather than taking its first observation directly.
        self.assertEqual(float(estimator.ema_ms_b), float(torch.tensor(0.1, dtype=torch.bfloat16)))
        state = json.loads(json.dumps(estimator.state_dict()))
        restored = DraftAdvantageEstimator()
        restored.load_state_dict(state)
        rewards = [[[1, 0], [1, 0]], [[1, 1], [0, 0]]]
        original_batch, original_metrics = estimator.compute(make_batch(rewards))
        restored_batch, restored_metrics = restored.compute(make_batch(rewards))
        self.assertEqual(original_metrics, restored_metrics)
        self.assertEqual(estimator.state_dict(), restored.state_dict())
        torch.testing.assert_close(
            original_batch.batch["advantages"], restored_batch.batch["advantages"], rtol=0, atol=0
        )
        with self.assertRaisesRegex(ValueError, "config"):
            DraftAdvantageEstimator({"ema_beta": 0.5}).load_state_dict(state)

    def test_distributed_statistics_reduce_sums_and_image_counts_not_rank_means(self):
        local_rewards = [[[1, 1], [0, 0]]]
        remote_rewards = [[[1, 0], [1, 0]], [[0, 0], [0, 0]]]
        whole, expected_metrics = DraftAdvantageEstimator().compute(make_batch(local_rewards + remote_rewards))
        calls = []

        def reduce_remote(tensor, *, op, group):
            calls.append((tensor.dtype, tensor.shape, op, group))
            if len(calls) == 1:
                # Remote ranks contribute within-image sums and total rewards.
                tensor.add_(torch.tensor([0, 1, 2], dtype=torch.bfloat16))
            elif len(calls) == 2:
                tensor.add_(2)  # Two remote images versus one local image.
            elif len(calls) == 3:
                # Equal means imply zero remote draft advantage and suffix |A|=2.
                tensor.add_(torch.tensor([0, 2], dtype=torch.bfloat16))
            else:
                self.fail("Unexpected collective")

        with (
            patch("emotion_grpo.draft_advantage.dist.is_available", return_value=True),
            patch("emotion_grpo.draft_advantage.dist.is_initialized", return_value=True),
            patch("emotion_grpo.draft_advantage.dist.all_reduce", side_effect=reduce_remote),
        ):
            local, actual_metrics = DraftAdvantageEstimator().compute(make_batch(local_rewards), global_batch=False)
        self.assertEqual(actual_metrics, expected_metrics)
        torch.testing.assert_close(local.batch["advantages"], whole.batch["advantages"][:4], rtol=0, atol=0)
        self.assertEqual([call[0] for call in calls], [torch.bfloat16, torch.int64, torch.bfloat16])

    def test_full_global_driver_batch_never_calls_collectives(self):
        with patch("emotion_grpo.draft_advantage.dist.all_reduce") as collective:
            DraftAdvantageEstimator().compute(make_batch([[[1, 0], [0, 1]]]))
        collective.assert_not_called()

    def test_sharded_mode_requires_initialized_process_group(self):
        with patch("emotion_grpo.draft_advantage.dist.is_initialized", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "initialized"):
                DraftAdvantageEstimator().compute(make_batch([[[1, 1], [0, 0]]]), global_batch=False)

    def test_invalid_configuration_fails_early(self):
        for config in (
            {"num_drafts": 3}, {"num_continuations_per_draft": 1}, {"ema_beta": 1},
            {"ema_beta": -0.1}, {"shrink_min": -1}, {"eps": 0}, {"eps": float("nan")},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                DraftAdvantageEstimator(config)


if __name__ == "__main__":
    unittest.main()
