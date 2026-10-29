"""Exercise batch controls through the real launcher without starting training."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_DIR / "scripts" / "run_emoset_grpo.sh"


class GrpoLauncherTests(unittest.TestCase):
    def run_launcher(self, *arguments, **environment):
        with tempfile.TemporaryDirectory(prefix="grpo launcher ") as temporary_dir:
            temporary = Path(temporary_dir)
            call_log = temporary / "python-calls.jsonl"
            fake_python = temporary / "fake-python"
            fake_python.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "with open(os.environ['LAUNCHER_CALL_LOG'], 'a') as stream:\n"
                "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env = os.environ.copy()
            for name in (
                "BATCH_SIZE",
                "GRADIENT_ACCUMULATION_STEPS",
                "ROLLOUT_N",
                "CUDA_VISIBLE_DEVICES",
            ):
                env.pop(name, None)
            env.update(
                PYTHON_BIN=str(fake_python),
                LAUNCHER_CALL_LOG=str(call_log),
                DATA_DIR=str(temporary / "data with spaces"),
                MODEL_PATH=str(temporary / "model with spaces"),
                EMOSET_ROOT=str(temporary / "dataset with spaces"),
            )
            env.update(environment)
            result = subprocess.run(
                ["bash", str(LAUNCHER), *arguments],
                cwd=PROJECT_DIR,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            calls = (
                [json.loads(line) for line in call_log.read_text(encoding="utf-8").splitlines()]
                if call_log.exists()
                else []
            )
            return result, calls

    def training_overrides(self, result, calls):
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual(calls[0][:2], ["-m", "emotion_grpo.prepare_data"])
        self.assertEqual(calls[1][:2], ["-m", "emotion_grpo.train"])
        self.assertIn("--train-size", calls[0])
        self.assertEqual(calls[0][calls[0].index("--train-size") + 1], "2000")
        options = calls[1][2:]
        keys = [option.split("=", 1)[0] for option in options]
        self.assertEqual(len(keys), len(set(keys)), "Conflicting duplicate Hydra overrides")
        return dict(option.split("=", 1) for option in options)

    def assert_batch_shape(self, overrides, batch, accumulation, gpu_count, rollout_n=4):
        responses = batch * accumulation * gpu_count
        prompts = responses // rollout_n
        self.assertEqual(overrides["data.train_batch_size"], str(prompts))
        self.assertEqual(overrides["actor_rollout_ref.actor.ppo_mini_batch_size"], str(prompts))
        self.assertEqual(overrides["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"], str(batch))
        self.assertEqual(overrides["actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu"], str(batch))
        self.assertEqual(overrides["actor_rollout_ref.rollout.n"], str(rollout_n))
        self.assertEqual(overrides["trainer.n_gpus_per_node"], str(gpu_count))
        self.assertEqual(overrides["trainer.nnodes"], "1")
        self.assertEqual(overrides["actor_rollout_ref.model.use_remove_padding"], str(batch == 1).lower())
        self.assertEqual(int(overrides["data.train_batch_size"]) * rollout_n, responses)
        self.assertEqual(2000 % prompts, 0)

    def test_defaults_preserve_batch16_and_one_accumulation_step(self):
        result, calls = self.run_launcher()
        overrides = self.training_overrides(result, calls)
        self.assert_batch_shape(overrides, batch=16, accumulation=1, gpu_count=2)
        self.assertTrue(overrides["actor_rollout_ref.model.path"].endswith("/model with spaces"))
        self.assertTrue(overrides["data_dir"].endswith("/data with spaces"))

    def test_environment_controls_effective_batch_and_padding(self):
        for batch, accumulation, devices, gpu_count, rollout_n in (
            (8, 4, "0,1", 2, 4),
            (16, 1, "0,1", 2, 4),
            (16, 2, "3", 1, 4),
            (1, 32, "0,1", 2, 4),
            (16, 2, "0,1", 2, 8),
        ):
            with self.subTest(batch=batch, accumulation=accumulation, devices=devices, rollout_n=rollout_n):
                result, calls = self.run_launcher(
                    BATCH_SIZE=str(batch),
                    GRADIENT_ACCUMULATION_STEPS=str(accumulation),
                    CUDA_VISIBLE_DEVICES=devices,
                    ROLLOUT_N=str(rollout_n),
                )
                overrides = self.training_overrides(result, calls)
                self.assert_batch_shape(overrides, batch, accumulation, gpu_count, rollout_n)

    def test_unrelated_hydra_overrides_pass_through(self):
        result, calls = self.run_launcher(
            "actor_rollout_ref.actor.optim.lr=5e-7",
            "trainer.total_epochs=3",
            "trainer.experiment_name=trial with spaces",
        )
        overrides = self.training_overrides(result, calls)
        self.assert_batch_shape(overrides, batch=16, accumulation=1, gpu_count=2)
        self.assertEqual(overrides["actor_rollout_ref.actor.optim.lr"], "5e-7")
        self.assertEqual(overrides["trainer.total_epochs"], "3")
        self.assertEqual(overrides["trainer.experiment_name"], "trial with spaces")

    def test_check_and_smoke_modes_keep_batch_controls(self):
        for mode in ("--check", "--smoke"):
            with self.subTest(mode=mode):
                result, calls = self.run_launcher(mode, BATCH_SIZE="8", GRADIENT_ACCUMULATION_STEPS="4")
                overrides = self.training_overrides(result, calls)
                self.assert_batch_shape(overrides, batch=8, accumulation=4, gpu_count=2)
                if mode == "--check":
                    self.assertEqual(overrides["dry_run"], "true")
                else:
                    self.assertEqual(overrides["trainer.resume_mode"], "disable")
                    self.assertEqual(overrides["trainer.total_training_steps"], "1")
                    self.assertEqual(overrides["trainer.val_before_train"], "false")

    def test_prepare_only_and_help_do_not_start_training(self):
        result, calls = self.run_launcher("--prepare-only")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 1, calls)
        self.assertEqual(calls[0][:2], ["-m", "emotion_grpo.prepare_data"])
        result, calls = self.run_launcher("--help")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(calls, [])
        for variable in ("BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "ROLLOUT_N"):
            self.assertIn(variable, result.stdout)

    def test_invalid_numbers_fail_before_data_preparation(self):
        for variable in ("BATCH_SIZE", "GRADIENT_ACCUMULATION_STEPS", "ROLLOUT_N"):
            for value in ("0", "-1", "1.5", "abc", "1+1", "9223372036854775824"):
                with self.subTest(variable=variable, value=value):
                    result, calls = self.run_launcher(**{variable: value})
                    self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual(calls, [], "Invalid input must fail before invoking Python")
                    self.assertIn(variable, result.stdout + result.stderr)

    def test_incomplete_response_groups_and_epoch_batches_fail_early(self):
        for environment in (
            {"BATCH_SIZE": "1", "GRADIENT_ACCUMULATION_STEPS": "1"},
            {"BATCH_SIZE": "3", "GRADIENT_ACCUMULATION_STEPS": "2"},
            {"BATCH_SIZE": "16", "GRADIENT_ACCUMULATION_STEPS": "4"},
            {"BATCH_SIZE": "4611686018427387920"},
        ):
            with self.subTest(environment=environment):
                result, calls = self.run_launcher(**environment)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(calls, [], "A partial batch must fail before invoking Python")
                self.assertTrue(result.stdout + result.stderr, "Explain why the batch shape is invalid")

    def test_direct_hydra_batch_overrides_cannot_bypass_accumulation(self):
        for override in (
            "data.train_batch_size=8",
            "+data.gen_batch_size=8",
            "actor_rollout_ref.actor.ppo_mini_batch_size=8",
            "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8",
            "actor_rollout_ref.rollout.n=8",
            "trainer.n_gpus_per_node=1",
            "trainer.nnodes=2",
            "actor_rollout_ref.model.use_remove_padding=true",
            "+data.train_batch_size=8",
            "++data.train_batch_size=8",
            "~data.train_batch_size",
        ):
            with self.subTest(override=override):
                result, calls = self.run_launcher(override)
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(calls, [], "Conflicting overrides must fail before invoking Python")
                self.assertIn("BATCH_SIZE", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
