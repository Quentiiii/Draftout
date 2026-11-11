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
    def test_ten_updates_per_launch_and_after_resume(self):
        controller = DraftTrainingController({})
        for start in (0, 375, 385):
            controller.begin_session(start)
            self.assertTrue(all(controller.is_warmup(start + step) for step in range(1, 11)))
            self.assertFalse(controller.is_warmup(start + 11))

    def test_stage_reaches_real_gen_batch_pop_and_repeat(self):
        trainer = RayPPOTrainer.__new__(RayPPOTrainer)
        trainer.config = OmegaConf.create({"actor_rollout_ref": {"actor": {}}})
        trainer.async_rollout_mode = True
        controller = DraftTrainingController({})
        controller.begin_session(0)
        shared = {"index": 42}
        for step in (1, 10, 11):
            batch = DataProto.from_dict(
                tensors={"input_ids": torch.ones((2, 3), dtype=torch.long)},
                non_tensors={"uid": np.array(["a", "b"], dtype=object),
                             "extra_info": np.array([shared, shared], dtype=object),
                             "raw_prompt": np.array(["p", "p"], dtype=object)},
            )
            controller.prepare_batch(batch, step)
            gen = trainer._get_gen_batch(batch).repeat(repeat_times=4, interleave=True)
            self.assertEqual(gen.meta_info["draft_rollout"], step > 10)
            self.assertEqual(len(gen), 8)
            self.assertTrue(all(info["draft_grpo_warmup"] == (step <= 10)
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
        result, metrics = controller.compute_advantage(batch, 10)
        torch.testing.assert_close(result.batch["advantages"], expected.to(torch.bfloat16))
        self.assertEqual(controller.estimator.num_updates, 0)
        self.assertEqual(metrics["draft/warmup"], 1)

    def test_draft_bypasses_group_normalization_and_keeps_padding_zero(self):
        controller = DraftTrainingController({})
        controller.begin_session(0)
        with patch("verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage",
                   side_effect=AssertionError("draft must not normalize")):
            batch, metrics = controller.compute_advantage(leaves(), 11)
        expected = torch.tensor([[.5, .5, 0, 0, 0], [.5, .5, 0, 0, 0],
                                 [-.5, -.5, 0, 0, 0], [-.5, -.5, 0, 0, 0]], dtype=torch.bfloat16)
        torch.testing.assert_close(batch.batch["advantages"], expected)
        self.assertEqual(metrics["draft/s"], 1)
        self.assertEqual(controller.estimator.num_updates, 1)

    def test_existing_clipped_ppo_gets_segment_gradients_without_prefix_dedup(self):
        controller = DraftTrainingController({})
        controller.begin_session(0)
        batch, _ = controller.compute_advantage(leaves(), 11)
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
            trainer.draft_training.compute_advantage(leaves(), 11)
            expected = trainer.draft_training.estimator.state_dict()
            trainer.global_steps = 11
            trainer._save_checkpoint()
            checkpoint = Path(temporary) / "global_step_11"
            self.assertTrue((checkpoint / "draft_advantage_state.json").is_file())
            self.assertEqual((Path(temporary) / "latest_checkpointed_iteration.txt").read_text(), "11")
            trainer.draft_training = DraftTrainingController({})
            trainer._load_checkpoint()
            self.assertEqual(trainer.draft_training.estimator.state_dict(), expected)
            self.assertEqual(trainer.train_dataloader.loaded["position"], 2)
            trainer.draft_training.begin_session(trainer.global_steps)
            self.assertTrue(trainer.draft_training.is_warmup(21))
            self.assertFalse(trainer.draft_training.is_warmup(22))

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
        trainer.global_steps = 11
        batch = leaves()
        batch.batch["prompts"] = torch.ones((4, 3), dtype=torch.long)
        batch.non_tensor_batch["request_id"] = np.array(["request"] * 4, dtype=object)
        with tempfile.TemporaryDirectory() as temporary:
            trainer._log_rollout_data(batch, {"task_reward": [1, 1, 0, 0]}, {}, temporary)
            rows = [json.loads(line) for line in (Path(temporary) / "11.jsonl").read_text().splitlines()]
        self.assertEqual([(r["draft_id"], r["branch_id"]) for r in rows], [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertTrue(all(r["image_id"] == "image" and r["draft_token_count"] == 2 for r in rows))
        self.assertTrue(all(r["request_id"] == "request" for r in rows))


class DraftLauncherTests(unittest.TestCase):
    def invoke(self, mode):
        project = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="draft launcher ") as temporary:
            root = Path(temporary)
            calls = root / "calls.jsonl"
            python = root / "fake-python"
            python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['DRAFT_CALLS'], 'a') as output:\n"
                "    output.write(json.dumps({'args': sys.argv[1:], 'cuda': os.getenv('CUDA_VISIBLE_DEVICES')})+'\\n')\n"
            )
            python.chmod(0o755)
            env = dict(os.environ, PYTHON_BIN=str(python), DRAFT_CALLS=str(calls),
                       CUDA_VISIBLE_DEVICES="2,3", DATA_DIR=str(root / "data"),
                       GRADIENT_ACCUMULATION_STEPS="2", ROLLOUT_N="4")
            env.pop("BATCH_SIZE", None)
            result = subprocess.run(["bash", str(project / "scripts/run_emoset_draft_grpo.sh"), mode],
                                    env=env, text=True, capture_output=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            return [json.loads(row) for row in calls.read_text().splitlines()]

    def test_check_selects_draft_config_and_hides_cuda_for_preflight(self):
        calls = self.invoke("--check")
        args = calls[-1]["args"]
        self.assertEqual(args[args.index("--config-name") + 1], "emoset_draft_grpo")
        self.assertIn("dry_run=true", args)
        self.assertEqual(calls[-1]["cuda"], "")

    def test_smoke_reaches_first_draft_step_in_its_own_experiment(self):
        calls = self.invoke("--smoke")
        overrides = dict(arg.split("=", 1) for arg in calls[-1]["args"] if "=" in arg)
        self.assertEqual(overrides["trainer.total_training_steps"], "11")
        self.assertEqual(overrides["trainer.resume_mode"], "disable")
        self.assertEqual(overrides["trainer.experiment_name"], "emoset_draft_grpo_2b_2k_smoke")
        self.assertEqual(overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"], "8")
        self.assertEqual(overrides["data.train_batch_size"], "8")
        self.assertEqual(overrides["actor_rollout_ref.rollout.n"], "4")


if __name__ == "__main__":
    unittest.main()
