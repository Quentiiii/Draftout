"""CPU checks for draft rewards, recipe validation, and resume contracts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from emotion_eval.data import LABELS
from emotion_grpo.reward import compute_score, compute_draft_score
from emotion_grpo.train import check_checkpoint_prompt, check_config, prompt_contract


DRAFT_SETTINGS = {
    "num_drafts": 2,
    "num_continuations_per_draft": 2,
    "ema_beta": 0.9,
    "shrink_min": 0.1,
    "eps": 1.0e-6,
    "warmup_steps": 10,
}


def strict_answer(label):
    return f"<draft>1. Facial expression 2. Environment</draft>\nThe evidence supports this emotion.\n\\boxed{{{label}}}"


def draft_config(directory):
    return OmegaConf.create({
        "algorithm": {"adv_estimator": "draft_grpo"},
        "draft_grpo": DRAFT_SETTINGS,
        "trainer": {
            "default_local_dir": str(directory), "resume_mode": "auto", "resume_from_path": None,
        },
    })


class DraftRewardTests(unittest.TestCase):
    def test_boxed_task_reward_is_independent_of_format(self):
        for label in LABELS["emoset"]:
            for text in (
                strict_answer(label), f"\\boxed{{{label}}}",
                f"<draft>Visible evidence</draft>\\boxed{{{label}}}",
                f"No draft here. \\boxed{{ {label.upper()} }}",
            ):
                with self.subTest(label=label, text=text):
                    result = compute_draft_score("emoset", text, label.lower())
                    self.assertEqual(result["score"], 1.0)
                    self.assertEqual(result["task_reward"], 1.0)
                    self.assertEqual(result["format_valid"], float(text == strict_answer(label)))

    def test_warmup_reuses_original_strict_reward_but_keeps_task_score(self):
        for text in (strict_answer("Fear"), r"\boxed{Fear}", strict_answer("Anger")):
            with self.subTest(text=text):
                regular = compute_draft_score("emoset", text, "Fear")
                warmup = compute_draft_score("emoset", text, "Fear", {"draft_grpo_warmup": True})
                self.assertEqual(warmup["score"], compute_score("emoset", text, "Fear"))
                self.assertEqual(warmup["task_reward"], regular["task_reward"])
                self.assertEqual(warmup["format_valid"], regular["format_valid"])
        self.assertEqual(compute_draft_score("emoset", strict_answer("Anger"), "Fear")["format_valid"], 1.0)

    def test_wrong_missing_multiple_or_malformed_boxes_receive_zero(self):
        for text in (
            None, 7, "", "Fear", strict_answer("Anger"), r"\boxed{}", r"\boxed{Joy}",
            r"\boxed{Fear or Anger}", r"\boxed{Fear", r"\boxed{{Fear}}",
            r"\boxed{Fear}\boxed{Fear}", r"\boxed{Fear}\boxed{Anger}",
            r"\boxed{Fear}\boxed{", r"\boxed{Fear}\boxed",
        ):
            with self.subTest(text=text):
                result = compute_draft_score("emoset", text, "Fear")
                self.assertEqual(result["score"], 0.0)
                self.assertEqual(result["task_reward"], 0.0)

    def test_invalid_ground_truth_is_an_error_in_both_phases(self):
        for truth in (None, "", "Joy", "Anger or Fear"):
            for warmup in (False, True):
                with self.subTest(truth=truth, warmup=warmup), self.assertRaisesRegex(ValueError, "ground-truth"):
                    compute_draft_score("emoset", r"\boxed{Fear}", truth, {"draft_grpo_warmup": warmup})


class DraftResumeContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "draft_run"
        self.config = draft_config(self.directory)

    def write_checkpoint(self, directory=None, step=25, contract=None, state=True):
        directory = directory or self.directory
        checkpoint = directory / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        (directory / "prompt_contract.json").write_text(json.dumps(contract or prompt_contract(self.config)))
        (directory / "latest_checkpointed_iteration.txt").write_text(str(step))
        if state:
            (checkpoint / "draft_advantage_state.json").write_text("{}")
        return checkpoint

    def test_old_contract_stays_unchanged_and_draft_fresh_check_writes_nothing(self):
        old = prompt_contract()
        self.assertEqual(old["schema_version"], 1)
        self.assertEqual(set(old), {"schema_version", "prompt", "prompt_sha256", "reward"})
        current = check_checkpoint_prompt(self.config)
        self.assertEqual(current["schema_version"], 2)
        self.assertEqual(current["method"], "draft_grpo_v1")
        self.assertEqual(current["warmup"]["steps_per_launch"], 10)
        self.assertEqual(current["draft_grpo"], {k: v for k, v in DRAFT_SETTINGS.items() if k != "warmup_steps"})
        self.assertFalse(self.directory.exists())

    def test_draft_resume_rejects_old_method_even_with_identical_prompt(self):
        self.write_checkpoint(contract=prompt_contract())
        with self.assertRaisesRegex(ValueError, "does not match"):
            check_checkpoint_prompt(self.config)

    def test_draft_resume_requires_matching_method_and_ema_settings(self):
        self.write_checkpoint()
        self.assertEqual(check_checkpoint_prompt(self.config), prompt_contract(self.config))
        for key, value in (("ema_beta", 0.8), ("shrink_min", 0.2), ("eps", 1.0e-5), ("warmup_steps", 9)):
            changed = copy.deepcopy(self.config)
            changed.draft_grpo[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "does not match"):
                check_checkpoint_prompt(changed)

    def test_selected_checkpoint_needs_ema_state_and_valid_tracker(self):
        checkpoint = self.write_checkpoint(state=False)
        with self.assertRaisesRegex(ValueError, "without EMA state"):
            check_checkpoint_prompt(self.config)
        (checkpoint / "draft_advantage_state.json").write_text("{}")
        tracker = self.directory / "latest_checkpointed_iteration.txt"
        tracker.write_text("50")
        with self.assertRaisesRegex(ValueError, "without EMA state"):
            check_checkpoint_prompt(self.config)
        tracker.write_text("bad")
        with self.assertRaisesRegex(ValueError, "Invalid latest checkpoint tracker"):
            check_checkpoint_prompt(self.config)
        tracker.unlink()
        with self.assertRaisesRegex(ValueError, "without a latest checkpoint tracker"):
            check_checkpoint_prompt(self.config)

    def test_explicit_resume_checks_source_state_without_writing_destination(self):
        checkpoint = self.write_checkpoint(self.directory.parent / "source", state=False)
        self.config.trainer.resume_mode = "resume_path"
        self.config.trainer.resume_from_path = str(checkpoint)
        with self.assertRaisesRegex(ValueError, "without EMA state"):
            check_checkpoint_prompt(self.config)
        (checkpoint / "draft_advantage_state.json").write_text("{}")
        self.assertEqual(check_checkpoint_prompt(self.config), prompt_contract(self.config))
        self.assertFalse(self.directory.exists())


class DraftRecipeConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_dir = Path(__file__).resolve().parents[1] / "verl" / "trainer" / "config"
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cls.base = compose(config_name="emoset_grpo")
        cls.draft = copy.deepcopy(cls.base)
        OmegaConf.update(cls.draft, "draft_grpo", DRAFT_SETTINGS, force_add=True)
        cls.draft.algorithm.adv_estimator = "draft_grpo"
        cls.draft.algorithm.norm_adv_by_std_in_grpo = False
        OmegaConf.update(
            cls.draft, "actor_rollout_ref.rollout.agent.agent_loop_manager_class",
            "emotion_grpo.draft_rollout.DraftAgentLoopManager", force_add=True,
        )
        cls.draft.custom_reward_function.name = "compute_draft_score"

    def test_ordinary_and_draft_configs_keep_existing_ppo_and_batch_settings(self):
        check_config(self.base)
        check_config(self.draft)
        self.assertEqual(self.draft.actor_rollout_ref.actor, self.base.actor_rollout_ref.actor)
        self.assertEqual(self.draft.data, self.base.data)

    def test_incompatible_draft_configs_fail_before_workers_are_created(self):
        changes = (
            ("draft_grpo.num_drafts", 3),
            ("draft_grpo.num_continuations_per_draft", 3),
            ("draft_grpo.warmup_steps", 0),
            ("draft_grpo.warmup_steps", 9),
            ("draft_grpo.ema_beta", 1.0),
            ("draft_grpo.ema_beta", float("nan")),
            ("draft_grpo.shrink_min", -0.1),
            ("draft_grpo.eps", 0.0),
            ("actor_rollout_ref.rollout.n", 8),
            ("algorithm.norm_adv_by_std_in_grpo", True),
            ("actor_rollout_ref.rollout.agent.agent_loop_manager_class", None),
            ("actor_rollout_ref.rollout.mode", "sync"),
            ("custom_reward_function.name", "compute_score"),
            ("algorithm.use_kl_in_reward", True),
            ("actor_rollout_ref.actor.policy_loss.loss_mode", "gpg"),
        )
        for key, value in changes:
            changed = copy.deepcopy(self.draft)
            OmegaConf.update(changed, key, value)
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                check_config(changed)


if __name__ == "__main__":
    unittest.main()
