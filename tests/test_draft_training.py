"""Exercise launch warmup, real trainer plumbing/checkpoints, and PPO gradients."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from emotion_grpo.draft_training import DraftTrainingController
from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage, compute_policy_loss_vanilla
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def leaves():
    # Two drafts, different suffix lengths; 99 represents a sampled EOS.
    responses = torch.tensor([[10, 11, 20, 99, 0], [10, 11, 21, 22, 99],
                              [12, 13, 23, 99, 0], [12, 13, 24, 25, 99]])
    mask = (responses != 0).long()
    scores = torch.zeros((4, 5))
    scores[0, 3] = scores[1, 4] = 1
    return DataProto.from_dict(
        tensors={"responses": responses, "response_mask": mask,
                 "token_level_scores": scores, "token_level_rewards": scores.clone()},
        non_tensors={"image_id": np.array(["image"] * 4, dtype=object),
                     "uid": np.array(["image"] * 4, dtype=object),
                     "draft_id": np.array([0, 0, 1, 1], dtype=object),
                     "branch_id": np.array([0, 1, 0, 1], dtype=object),
                     "draft_token_count": np.array([2] * 4, dtype=object)},
    )


class TinyLoader:
    def __len__(self):
        return 3

    def state_dict(self):
        return {"_iterator_finished": True, "position": 2}

    def load_state_dict(self, state):
        self.loaded = state


class DraftTrainingTests(unittest.TestCase):
    def test_fresh_start_has_twenty_warmup_updates_but_resume_skips_them_by_default(self):
        controller = DraftTrainingController({})
        controller.begin_session(0)
        self.assertTrue(all(controller.is_warmup(step) for step in range(1, 21)))
        self.assertFalse(controller.is_warmup(21))
        for start in (1, 10, 20, 100, 375, 395):
            controller.begin_session(start)
            self.assertTrue(all(not controller.is_warmup(start + step) for step in range(1, 22)))

    def test_explicit_resume_warmup_retains_twenty_updates_on_every_launch(self):
        controller = DraftTrainingController({"warmup_on_resume": True})
        for start in (0, 100, 375):
            controller.begin_session(start)
            self.assertTrue(all(controller.is_warmup(start + step) for step in range(1, 21)))
            self.assertFalse(controller.is_warmup(start + 21))

    def test_resume_warmup_setting_requires_an_actual_bool(self):
        for value in (0, 1, None, "false", "true", [], np.bool_(False)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "warmup_on_resume.*bool"):
                DraftTrainingController({"warmup_on_resume": value})

    def test_stage_reaches_real_gen_batch_pop_and_repeat(self):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create({"actor_rollout_ref": {"actor": {}}})
        trainer.async_rollout_mode = True
        controller = DraftTrainingController({})
        controller.begin_session(0)
        shared = {"index": 42}
        for step in (1, 10, 11, 20, 21):
            batch = DataProto.from_dict(
                tensors={"input_ids": torch.ones((2, 3), dtype=torch.long)},
                non_tensors={"uid": np.array(["a", "b"], dtype=object),
                             "extra_info": np.array([shared, shared], dtype=object),
                             "raw_prompt": np.array(["p", "p"], dtype=object)},
            )
            controller.prepare_batch(batch, step)
            gen = trainer._get_gen_batch(batch).repeat(repeat_times=4, interleave=True)
            self.assertEqual(gen.meta_info["draft_rollout"], step > 20)
            self.assertEqual(len(gen), 8)
            self.assertTrue(all(info["draft_grpo_warmup"] == (step <= 20)
                                for info in gen.non_tensor_batch["extra_info"]))
        self.assertEqual(shared, {"index": 42})

    def test_warmup_uses_ordinary_grpo_and_does_not_update_ema(self):
        controller = DraftTrainingController({})
        controller.begin_session(0)
        batch = leaves()
        expected, _ = compute_grpo_outcome_advantage(
            batch.batch["token_level_rewards"], batch.batch["response_mask"],
            batch.non_tensor_batch["uid"], norm_adv_by_std_in_grpo=True,
        )
        # Warmup must work without any draft metadata or shared prefixes.
        for key in ("image_id", "draft_id", "branch_id", "draft_token_count"):
            batch.non_tensor_batch.pop(key)
        result, metrics = controller.compute_advantage(batch, 20)
        torch.testing.assert_close(result.batch["advantages"], expected.to(torch.bfloat16))
        self.assertEqual(controller.estimator.num_updates, 0)
        self.assertEqual(metrics["draft/warmup"], 1)

    def test_draft_bypasses_group_normalization_and_keeps_padding_zero(self):
        controller = DraftTrainingController({})
        controller.begin_session(0)
        with patch("verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage",
                   side_effect=AssertionError("draft must not normalize")):
            batch, metrics = controller.compute_advantage(leaves(), 21)
        expected = torch.tensor([[.5, .5, 0, 0, 0], [.5, .5, 0, 0, 0],
                                 [-.5, -.5, 0, 0, 0], [-.5, -.5, 0, 0, 0]], dtype=torch.bfloat16)
        torch.testing.assert_close(batch.batch["advantages"], expected)
        self.assertEqual(metrics["draft/s"], 1)
        self.assertEqual(controller.estimator.num_updates, 1)

    def test_existing_clipped_ppo_gets_segment_gradients_without_prefix_dedup(self):
        controller = DraftTrainingController({})
        controller.begin_session(0)
        batch, _ = controller.compute_advantage(leaves(), 21)
        # A full sequence proves that the sliced prompt gets no policy gradient.
        logits = torch.zeros((4, 8), requires_grad=True)
        log_probs = logits[:, 3:]
        config = OmegaConf.create({"clip_ratio": .2, "clip_ratio_low": .2,
                                   "clip_ratio_high": .2, "clip_ratio_c": 1e9,
                                   "global_batch_info": {}})
        loss, _ = compute_policy_loss_vanilla(
            old_log_prob=torch.zeros_like(log_probs), log_prob=log_probs,
            advantages=batch.batch["advantages"], response_mask=batch.batch["response_mask"],
            loss_agg_mode="seq-mean-token-mean", config=config,
        )
        loss.backward()
        self.assertEqual(torch.count_nonzero(logits.grad[:, :3]), 0)
        self.assertTrue(torch.all(logits.grad[:2, 3:5] < 0))
        self.assertTrue(torch.all(logits.grad[2:, 3:5] > 0))
        self.assertEqual(torch.count_nonzero(logits.grad[:, 5:]), 0)
        # Both copies of each sampled draft receive a gradient.
        self.assertEqual(torch.count_nonzero(logits.grad[:, 3:5]), 8)

    def test_real_trainer_saves_ema_before_tracker_and_restores_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            trainer = RayPPOTrainer.__new__(RayPPOTrainer)
            trainer.config = OmegaConf.create({
                "trainer": {"default_local_dir": temporary, "default_hdfs_dir": None,
                            "resume_mode": "auto", "del_local_ckpt_after_load": False,
                            "max_actor_ckpt_to_keep": 2, "max_critic_ckpt_to_keep": 2},
                "actor_rollout_ref": {"actor": {"checkpoint": {"async_save": False}}},
            })
            trainer.actor_rollout_wg = Mock()
            trainer.use_critic = False
            trainer.train_dataloader = TinyLoader()
            trainer.draft_training = DraftTrainingController({})
            trainer.draft_training.begin_session(0)
            trainer.draft_training.compute_advantage(leaves(), 21)
            trainer.draft_training.compute_advantage(leaves(), 22)
            expected = trainer.draft_training.estimator.state_dict()
            trainer.global_steps = 100
            trainer._save_checkpoint()
            checkpoint = Path(temporary) / "global_step_100"
            self.assertTrue((checkpoint / "draft_advantage_state.json").is_file())
            self.assertEqual((Path(temporary) / "latest_checkpointed_iteration.txt").read_text(), "100")
            trainer.draft_training = DraftTrainingController({})
            trainer._load_checkpoint()
            self.assertEqual(trainer.draft_training.estimator.state_dict(), expected)
            self.assertEqual(trainer.train_dataloader.loaded["position"], 2)
            trainer.draft_training.begin_session(trainer.global_steps)
            self.assertFalse(trainer.draft_training.is_warmup(101))
            batch = leaves()
            trainer.draft_training.prepare_batch(batch, 101)
            self.assertTrue(batch.meta_info["draft_rollout"])
            self.assertTrue(all(not info["draft_grpo_warmup"] for info in batch.non_tensor_batch["extra_info"]))
            with patch("verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage",
                       side_effect=AssertionError("resume must directly use Draft advantages")):
                _, metrics = trainer.draft_training.compute_advantage(batch, 101)
            self.assertEqual(metrics["draft/warmup"], 0)
            self.assertEqual(metrics["draft/session_step"], 1)
            self.assertEqual(trainer.draft_training.estimator.num_updates, expected["num_updates"] + 1)
            expected_ema = (0.9 * torch.tensor(expected["ema_ms_b"], dtype=torch.bfloat16)
                            + 0.1 * torch.tensor(1, dtype=torch.bfloat16))
            torch.testing.assert_close(trainer.draft_training.estimator.ema_ms_b, expected_ema, rtol=0, atol=0)

    def test_legacy_checkpoint_without_resume_policy_preserves_ema_and_skips_warmup(self):
        controller = DraftTrainingController({"warmup_on_resume": True})
        controller.begin_session(0)
        controller.compute_advantage(leaves(), 21)
        expected = controller.estimator.state_dict()
        with tempfile.TemporaryDirectory() as temporary:
            controller.save(temporary, 100)
            path = Path(temporary) / controller.state_filename
            payload = json.loads(path.read_text())
            self.assertTrue(payload.pop("warmup_on_resume"))
            self.assertEqual(payload["version"], 1)
            self.assertEqual(payload["warmup_steps_per_launch"], 20)
            path.write_text(json.dumps(payload))
            restored = DraftTrainingController({})
            restored.load(temporary, 100)
            self.assertEqual(restored.estimator.state_dict(), expected)
            restored.begin_session(100)
            self.assertFalse(restored.is_warmup(101))
            with patch("verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage",
                       side_effect=AssertionError("legacy resume must directly use Draft advantages")):
                _, metrics = restored.compute_advantage(leaves(), 101)
            self.assertEqual(metrics["draft/warmup"], 0)
            self.assertEqual(restored.estimator.num_updates, expected["num_updates"] + 1)

    def test_checkpoint_resume_policy_is_only_a_record_and_current_setting_controls_warmup(self):
        for saved_policy, current_policy in ((True, False), (False, True)):
            with self.subTest(saved=saved_policy, current=current_policy), tempfile.TemporaryDirectory() as temporary:
                controller = DraftTrainingController({"warmup_on_resume": saved_policy})
                controller.begin_session(0)
                controller.compute_advantage(leaves(), 21)
                controller.save(temporary, 100)
                expected = controller.estimator.state_dict()
                payload = json.loads((Path(temporary) / controller.state_filename).read_text())
                self.assertIs(payload["warmup_on_resume"], saved_policy)
                restored = DraftTrainingController({"warmup_on_resume": current_policy})
                restored.load(temporary, 100)
                restored.begin_session(100)
                self.assertEqual(restored.estimator.state_dict(), expected)
                self.assertIs(restored.is_warmup(101), current_policy)
                batch = leaves()
                restored.prepare_batch(batch, 101)
                self.assertIs(batch.meta_info["draft_rollout"], not current_policy)
                _, metrics = restored.compute_advantage(batch, 101)
                self.assertEqual(metrics["draft/warmup"], float(current_policy))
                self.assertEqual(restored.estimator.num_updates, expected["num_updates"] + int(not current_policy))

    def test_existing_ppo_trains_entire_fallback_response_including_eos(self):
        controller = DraftTrainingController({})
        controller.begin_session(0)
        batch = leaves()
        batch.non_tensor_batch["draft_token_count"][:2] = 0
        batch.batch["token_level_scores"].zero_()
        batch.batch["token_level_scores"][0, 3] = 1
        batch.batch["token_level_scores"][2, 3] = 1
        batch, metrics = controller.compute_advantage(batch, 21)
        self.assertEqual(metrics["draft/fallback_draft_count"], 1)
        # Both draft groups have mean 0.5, so every suffix has advantage +/-0.5.
        logits = torch.zeros((4, 8), requires_grad=True)
        config = OmegaConf.create({"clip_ratio": .2, "clip_ratio_low": .2,
                                   "clip_ratio_high": .2, "clip_ratio_c": 1e9,
                                   "global_batch_info": {}})
        loss, _ = compute_policy_loss_vanilla(
            old_log_prob=torch.zeros_like(logits[:, 3:]), log_prob=logits[:, 3:],
            advantages=batch.batch["advantages"], response_mask=batch.batch["response_mask"],
            loss_agg_mode="seq-mean-token-mean", config=config,
        )
        loss.backward()
        self.assertTrue(torch.all(logits.grad[0, 3:7] < 0))  # Entire fallback, including EOS.
        self.assertTrue(torch.all(logits.grad[1, 3:] > 0))
        self.assertEqual(torch.count_nonzero(logits.grad[:, :3]), 0)
        self.assertEqual(torch.count_nonzero(logits.grad[2:, 3:5]), 0)  # Valid draft prefix.
        self.assertEqual(float(logits.grad[0, 7]), 0)  # Right padding.

    def test_checkpoint_missing_or_wrong_step_cannot_silently_reset_ema(self):
        controller = DraftTrainingController({})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                controller.load(temporary, 25)
            controller.save(temporary, 25)
            with self.assertRaises(ValueError):
                controller.load(temporary, 50)

    def test_rollout_log_keeps_draft_ids_and_token_boundary(self):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.tokenizer = Mock()
        trainer.tokenizer.batch_decode.return_value = ["decoded"] * 4
        trainer.global_steps = 21
        batch = leaves()
        batch.batch["prompts"] = torch.ones((4, 3), dtype=torch.long)
        batch.non_tensor_batch["request_id"] = np.array(["request"] * 4, dtype=object)
        batch.non_tensor_batch["draft_token_count"][:2] = 0
        batch.non_tensor_batch["draft_fallback"] = np.array([True, True, False, False], dtype=object)
        batch.non_tensor_batch["draft_fallback_reason"] = np.array(["early EOS", "early EOS", "", ""], dtype=object)
        with tempfile.TemporaryDirectory() as temporary:
            trainer._log_rollout_data(batch, {"task_reward": [1, 1, 0, 0]}, {}, temporary)
            rows = [json.loads(line) for line in (Path(temporary) / "21.jsonl").read_text().splitlines()]
        self.assertEqual([(r["draft_id"], r["branch_id"]) for r in rows], [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertTrue(all(r["image_id"] == "image" for r in rows))
        self.assertEqual([r["draft_token_count"] for r in rows], [0, 0, 2, 2])
        self.assertEqual([r["draft_fallback"] for r in rows], [True, True, False, False])
        self.assertEqual([r["draft_fallback_reason"] for r in rows], ["early EOS", "early EOS", "", ""])
        self.assertTrue(all(r["request_id"] == "request" for r in rows))


class DraftLauncherTests(unittest.TestCase):
    def invoke(self, mode, use_default_data=False):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="draft launcher ") as temporary:
            root = Path(temporary)
            calls = root / "calls.jsonl"
            training_log = root / "full-training.log"
            python = root / "fake-python"
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['DRAFT_CALLS'], 'a') as output:\n"
                "    output.write(json.dumps({'args': sys.argv[1:], 'cuda': os.getenv('CUDA_VISIBLE_DEVICES'), 'full_error': os.getenv('HYDRA_FULL_ERROR')})+'\\n')\n"
                "print('fake training stdout')\n"
                "print('fake training exception on stderr', file=sys.stderr)\n"
            )
            python.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(python), DRAFT_CALLS=str(calls),
                       CUDA_VISIBLE_DEVICES="2,3", DATA_DIR=str(root / "data"),
                       DRAFT_TRAIN_LOG=str(training_log),
                       GRADIENT_ACCUMULATION_STEPS="2", ROLLOUT_N="4")
            env.pop("BATCH_SIZE", None)
            env.pop("HYDRA_FULL_ERROR", None)
            if use_default_data:
                env.pop("DATA_DIR")
            result = subprocess.run(["bash", str(project / "scripts/run_emoset_draft_grpo.sh"), mode],
                                    env=env, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            if mode == "--smoke":
                logged = training_log.read_text()
                self.assertIn("fake training stdout", logged)
                self.assertIn("fake training exception on stderr", logged)
            else:
                self.assertFalse(training_log.exists())
            recorded = [json.loads(row) for row in calls.read_text().splitlines()]
            prepare = recorded[0]["args"]
            self.assertEqual(prepare[:2], ["-m", "emotion_grpo.prepare_data"])
            self.assertEqual(prepare[prepare.index("--prompt-style") + 1], "draft")
            self.assertEqual(prepare[prepare.index("--train-size") + 1], "8000")
            return recorded

    def test_prepare_preserves_draft_prompt_in_its_separate_data_directory(self):
        calls = self.invoke("--prepare-only", use_default_data=True)
        self.assertEqual(len(calls), 1)
        args = calls[0]["args"]
        self.assertTrue(args[args.index("--output-dir") + 1].endswith("/outputs/emoset_grpo/data_draft_8k_seed42"))

    def test_check_selects_draft_config_and_hides_cuda_for_preflight(self):
        calls = self.invoke("--check")
        args = calls[-1]["args"]
        self.assertEqual(args[args.index("--config-name") + 1], "emoset_draft_grpo")
        self.assertIn("dry_run=true", args)
        self.assertEqual(calls[-1]["cuda"], "")

    def test_smoke_reaches_first_draft_step_in_its_own_experiment(self):
        calls = self.invoke("--smoke")
        self.assertEqual(calls[-1]["full_error"], "1")
        overrides = dict(arg.split("=", 1) for arg in calls[-1]["args"] if "=" in arg)
        self.assertEqual(overrides["trainer.total_training_steps"], "21")
        self.assertEqual(overrides["trainer.resume_mode"], "disable")
        self.assertEqual(overrides["trainer.experiment_name"], "emoset_draft_grpo_2b_8k_warmup20_fallback_smoke")
        self.assertEqual(overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"], "8")
        self.assertEqual(overrides["data.train_batch_size"], "8")
        self.assertEqual(overrides["actor_rollout_ref.rollout.n"], "4")
        keys = [arg.split("=", 1)[0] for arg in calls[-1]["args"] if "=" in arg]
        self.assertEqual(len(keys), len(set(keys)), "Smoke must not rely on conflicting overrides")


if __name__ == "__main__":
    unittest.main()
