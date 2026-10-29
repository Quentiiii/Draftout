"""Resume the real trainer's saved data state without loading model weights."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from verl.trainer.ppo.ray_trainer import RayPPOTrainer


def identity_collate(batch):
    # Keep multiprocessing CPU tests independent of Tensor IPC sockets.
    return batch


def make_loader(workers):
    dataset = list(range(40))
    sampler = RandomSampler(dataset, generator=torch.Generator().manual_seed(42))
    return StatefulDataLoader(
        dataset, batch_size=8, sampler=sampler, num_workers=workers,
        drop_last=True, collate_fn=identity_collate,
    )


class GrpoResumeTests(unittest.TestCase):
    def resume(self, state, step, workers):
        with tempfile.TemporaryDirectory(prefix="grpo-resume-") as directory:
            root = Path(directory)
            checkpoint = root / f"global_step_{step}"
            checkpoint.mkdir()
            torch.save(state, checkpoint / "data.pt")
            trainer = RayPPOTrainer.__new__(RayPPOTrainer)
            trainer.config = SimpleNamespace(trainer=SimpleNamespace(
                resume_mode="auto", default_hdfs_dir=None, default_local_dir=str(root),
                del_local_ckpt_after_load=False,
            ))
            trainer.actor_rollout_wg = Mock()
            trainer.use_critic = False
            trainer.train_dataloader = make_loader(workers)
            with patch("verl.trainer.ppo.ray_trainer.find_latest_ckpt_path", return_value=str(checkpoint)):
                trainer._load_checkpoint()
            self.assertEqual(trainer.global_steps, step)
            trainer.actor_rollout_wg.load_checkpoint.assert_called_once_with(
                str(checkpoint / "actor"), del_local_after_load=False,
            )
            return trainer.train_dataloader

    def test_epoch_boundary_runs_all_nine_remaining_epochs_in_original_order(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                original = make_loader(workers)
                iterator = iter(original)
                for _ in range(len(original)):
                    next(iterator)
                state = original.state_dict()
                self.assertFalse(state["_iterator_finished"])
                self.assertEqual(list(iterator), [])
                expected = [list(original) for _ in range(9)]
                resumed = self.resume(state, step=5, workers=workers)
                actual = [list(resumed) for _ in range(1, 10)]
                self.assertEqual([len(epoch) for epoch in actual], [5] * 9)
                self.assertEqual(actual, expected)

    def test_mid_epoch_preserves_remaining_batches_and_next_epoch(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                original = make_loader(workers)
                iterator = iter(original)
                next(iterator)
                next(iterator)
                state = original.state_dict()
                expected_remaining = list(iterator)
                expected_next = list(original)
                resumed = self.resume(state, step=2, workers=workers)
                self.assertEqual(list(resumed), expected_remaining)
                self.assertEqual(list(resumed), expected_next)

    def test_already_finished_iterator_does_not_consume_next_epoch(self):
        for workers in (0, 2):
            with self.subTest(workers=workers):
                original = make_loader(workers)
                list(original)
                state = original.state_dict()
                self.assertTrue(state["_iterator_finished"])
                expected_next = list(original)
                resumed = self.resume(state, step=5, workers=workers)
                self.assertEqual(list(resumed), expected_next)

    def test_inconsistent_epoch_boundary_fails_instead_of_dropping_data(self):
        original = make_loader(0)
        iterator = iter(original)
        next(iterator)
        with self.assertRaisesRegex(ValueError, "original training dataset and batch size"):
            self.resume(original.state_dict(), step=5, workers=0)


if __name__ == "__main__":
    unittest.main()
