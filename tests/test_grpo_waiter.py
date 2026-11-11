"""CPU-only regression checks for fixed-GPU GRPO waiting and launch control."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/wait_emoset_grpo.py"
SPEC = importlib.util.spec_from_file_location("grpo_waiter_under_test", SCRIPT)
WAITER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WAITER)
DRAFT_LAUNCHER = PROJECT / "scripts/run_emoset_draft_grpo.sh"


def snapshot(idle=(), *, uuid_suffix=""):
    return [
        {
            "index": index,
            "uuid": f"GPU-{index}{uuid_suffix}",
            "used_mib": 20 if index in idle else 15000,
            "utilization": 0 if index in idle else 90,
            "pids": [] if index in idle else [1000 + index],
        }
        for index in range(4)
    ]


class GrpoWaiterSelectionTests(unittest.TestCase):
    def test_requested_gpus_never_substitute_other_idle_cards(self):
        for available in ((0, 1), (0, 1, 2), (2,), (3,)):
            with self.subTest(available=available):
                self.assertEqual(WAITER.select_gpus(snapshot(available), (2, 3)), [])

    def test_requested_pair_preserves_requested_order_and_legacy_default(self):
        gpus = list(reversed(snapshot((0, 1, 2, 3))))
        selected = WAITER.select_gpus(gpus, (2, 3))
        self.assertEqual([gpu["index"] for gpu in selected], [2, 3])
        self.assertEqual([gpu["uuid"] for gpu in selected], ["GPU-2", "GPU-3"])
        self.assertEqual([gpu["index"] for gpu in WAITER.select_gpus(snapshot((0, 1, 2, 3)))], [0, 1])

    def test_process_memory_and_utilization_each_prevent_idle_selection(self):
        for field, value in (("pids", [123]), ("used_mib", 1025), ("utilization", 6)):
            with self.subTest(field=field):
                gpus = snapshot((2, 3))
                gpus[3][field] = value
                self.assertEqual(WAITER.select_gpus(gpus, (2, 3)), [])


class GrpoWaiterCommandTests(unittest.TestCase):
    def test_real_launchers_and_python_training_are_recognized(self):
        commands = [
            ["bash", str(WAITER.LAUNCHER)],
            ["/bin/bash", str(DRAFT_LAUNCHER)],
            ["bash", "-x", "--", str(DRAFT_LAUNCHER)],
            [str(DRAFT_LAUNCHER)],
            [str(WAITER.LAUNCHER)],
            ["/env/bin/python3.12", "-m", "emotion_grpo.train", "trainer.resume_mode=auto"],
        ]
        for command in commands:
            with self.subTest(command=command):
                self.assertTrue(WAITER.is_training_command(command))

    def test_diagnostic_shell_text_does_not_count_as_training(self):
        commands = [
            [],
            ["bash", "-lc", f"ps aux | rg {DRAFT_LAUNCHER}"],
            ["bash", "-c", f"cat {WAITER.LAUNCHER}"],
            ["/bin/bash", "-lc", f"python -m emotion_grpo.train; cat {DRAFT_LAUNCHER}"],
            ["python", "-c", "print('python -m emotion_grpo.train')"],
            ["python", "-m", "emotion_grpo.prepare_data"],
            ["rg", str(DRAFT_LAUNCHER)],
        ]
        for command in commands:
            with self.subTest(command=command):
                self.assertFalse(WAITER.is_training_command(command))


class GrpoWaiterLaunchTests(unittest.TestCase):
    def run_waiter(self, snapshots, *, active=None, exit_code=0):
        """All process, signal, hardware and sleep operations are fakes."""
        child = Mock(pid=987654)
        child.wait.return_value = exit_code
        states = []
        sleeps = []
        launches = Mock(return_value=child)

        def record_print(message, **kwargs):
            states.append(json.loads(message))

        def sleep(seconds):
            sleeps.append(seconds)
            # Waiting/confirmation is always before the sole training launch.
            launches.assert_not_called()
            if len(sleeps) > 10:
                raise AssertionError("Wait loop did not consume the expected snapshot sequence")

        with tempfile.TemporaryDirectory(prefix="grpo-waiter-test-") as temporary:
            status_path = Path(temporary) / "status.json"
            log_path = Path(temporary) / "training.log"
            with patch.object(WAITER, "gpu_snapshot", side_effect=snapshots) as gpu_query, \
                    patch.object(WAITER, "existing_training", side_effect=active or [[]] * 12) as processes, \
                    patch.object(WAITER.subprocess, "Popen", launches), \
                    patch.object(WAITER.time, "sleep", side_effect=sleep), \
                    patch.object(WAITER.signal, "signal"), \
                    patch("builtins.print", side_effect=record_print), \
                    patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}):
                result = WAITER.wait_and_run(
                    status_path, log_path, launcher=DRAFT_LAUNCHER,
                    requested_gpus=(2, 3), state_metadata={"experiment_name": "draft-waiter-test"},
                )
            state_on_disk = json.loads(status_path.read_text())
        return {
            "result": result, "states": states, "state_on_disk": state_on_disk,
            "sleeps": sleeps, "launches": launches, "child": child,
            "gpu_calls": gpu_query.call_count, "process_calls": processes.call_count,
        }

    def assert_draft_launch(self, result):
        result["launches"].assert_called_once()
        command = result["launches"].call_args.args[0]
        options = result["launches"].call_args.kwargs
        self.assertEqual(command, ["bash", str(DRAFT_LAUNCHER)])
        self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "2,3")
        self.assertEqual(options["cwd"], PROJECT)
        self.assertTrue(options["start_new_session"])
        self.assertEqual(options["stdin"], WAITER.subprocess.DEVNULL)
        self.assertEqual(options["stderr"], WAITER.subprocess.STDOUT)
        result["child"].wait.assert_called_once_with()
        self.assertEqual(result["states"][-1], result["state_on_disk"])
        self.assertEqual(result["states"][-1]["requested_gpus"], [2, 3])
        self.assertEqual(result["states"][-1]["experiment_name"], "draft-waiter-test")

    def test_waits_for_both_requested_cards_then_launches_draft_once(self):
        result = self.run_waiter([
            snapshot((0, 1)), snapshot((0, 1, 2)),
            snapshot((2, 3)), snapshot((2, 3)),
        ])
        self.assertEqual(result["result"], 0)
        self.assertEqual(result["gpu_calls"], 4)
        self.assertEqual(result["sleeps"], [WAITER.POLL_SECONDS, WAITER.POLL_SECONDS, WAITER.CONFIRM_SECONDS])
        self.assertEqual([state["status"] for state in result["states"]],
                         ["waiting", "waiting", "waiting", "starting", "running", "finished"])
        self.assert_draft_launch(result)

    def test_second_confirmation_busy_or_uuid_changed_returns_to_waiting(self):
        for confirmation in (snapshot((2,)), snapshot((2, 3), uuid_suffix="-changed")):
            with self.subTest(confirmation=confirmation):
                result = self.run_waiter([
                    snapshot((2, 3)), confirmation,
                    snapshot((2, 3)), snapshot((2, 3)),
                ])
                self.assertEqual(result["gpu_calls"], 4)
                self.assertEqual(result["sleeps"], [WAITER.CONFIRM_SECONDS, WAITER.POLL_SECONDS, WAITER.CONFIRM_SECONDS])
                self.assertEqual([state["status"] for state in result["states"]],
                                 ["waiting", "waiting", "starting", "running", "finished"])
                self.assert_draft_launch(result)

    def test_failed_training_exits_without_automatic_retry(self):
        result = self.run_waiter([snapshot((2, 3)), snapshot((2, 3))], exit_code=7)
        self.assertEqual(result["result"], 7)
        self.assertEqual(result["gpu_calls"], 2)
        self.assertEqual(result["sleeps"], [WAITER.CONFIRM_SECONDS])
        self.assertEqual([state["status"] for state in result["states"]], ["waiting", "starting", "running", "failed"])
        self.assertEqual(result["state_on_disk"]["exit_code"], 7)
        self.assert_draft_launch(result)

    def test_existing_training_before_poll_or_after_confirmation_prevents_popen(self):
        cases = [([[12345]], [], ["not_started_existing_training"]),
                 ([[], [12345]], [snapshot((2, 3)), snapshot((2, 3))], ["waiting", "not_started_existing_training"])]
        for active, snapshots, expected_states in cases:
            with self.subTest(active=active):
                result = self.run_waiter(snapshots, active=active)
                self.assertEqual(result["result"], 0)
                result["launches"].assert_not_called()
                result["child"].wait.assert_not_called()
                self.assertEqual([state["status"] for state in result["states"]], expected_states)
                self.assertEqual(result["state_on_disk"]["existing_pids"], [12345])


if __name__ == "__main__":
    unittest.main()
