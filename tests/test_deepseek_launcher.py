"""Check shell entry points using an isolated project and a fake Python executable."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from emotion_eval import deepseek


class DeepSeekLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "isolated project"
        self.scripts = self.project / "scripts"
        self.scripts.mkdir(parents=True)
        for name in ("run_deepseek_emotion.sh", "run_deepseek_emoset.sh"):
            shutil.copyfile(deepseek.PROJECT_ROOT / "scripts" / name, self.scripts / name)
        self.capture = self.root / "arguments.json"
        executable = self.root / "fake python"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys\n"
            "pathlib.Path(os.environ['DEEPSEEK_TEST_CAPTURE']).write_text(\n"
            "    json.dumps({'argv': sys.argv[1:], 'cwd': os.getcwd()}), encoding='utf-8')\n",
            encoding="utf-8",
        )
        executable.chmod(0o755)
        self.env = {
            **os.environ, "PYTHON_BIN": str(executable), "DEEPSEEK_API_KEY": "offline-dummy-key",
            "DEEPSEEK_TEST_CAPTURE": str(self.capture),
        }
        # Only this dummy file exists in the isolated project. A nonempty environment
        # key must win, and the test never reads the real project's credential file.
        (self.project / ".deepseek.env").write_text("exit 99\n", encoding="utf-8")

    def launch(self, *arguments, name="run_deepseek_emotion.sh"):
        completed = subprocess.run(
            ["bash", str(self.scripts / name), *arguments], cwd=self.root, env=self.env,
            capture_output=True, text=True, timeout=10, check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        captured = json.loads(self.capture.read_text(encoding="utf-8"))
        self.assertEqual(captured["cwd"], str(self.project))
        self.assertEqual(captured["argv"][:2], ["-m", "emotion_eval.deepseek"])
        return captured["argv"][2:]

    def test_defaults_select_three_full_datasets_and_use_environment_python(self):
        args = deepseek.argument_parser().parse_args(self.launch())
        self.assertEqual(args.datasets, ["emoset", "emotion6", "emo8"])
        self.assertEqual(args.output_dir, Path("outputs/deepseek_v41_flash_emotion"))
        self.assertEqual(args.model, "deepseek-flash")
        self.assertEqual(args.workers, 4)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.max_image_side, 1024)
        self.assertIsNone(args.test_list)
        self.assertIsNone(args.limit)

    def test_cli_values_override_launcher_defaults(self):
        args = deepseek.argument_parser().parse_args(self.launch(
            "--datasets", "emotion6", "--output-dir", "custom-results", "--workers", "2",
            "--model", "custom-model", "--max-tokens", "256", "--limit", "3", "--dry-run",
        ))
        self.assertEqual(args.datasets, ["emotion6"])
        self.assertEqual(args.output_dir, Path("custom-results"))
        self.assertEqual(args.workers, 2)
        self.assertEqual(args.model, "custom-model")
        self.assertEqual(args.max_tokens, 256)
        self.assertEqual(args.limit, 3)
        self.assertTrue(args.dry_run)

    def test_legacy_subset_omits_default_datasets_and_output_but_explicit_selection_keeps_new_layout(self):
        for option in (("--test-list", "subset.json"), ("--test-list=subset.json",)):
            with self.subTest(option=option):
                args = deepseek.argument_parser().parse_args(self.launch(*option, "--resume"))
                self.assertIsNone(args.datasets)
                self.assertIsNone(args.output_dir)
                self.assertEqual(args.test_list, Path("subset.json"))
                self.assertTrue(args.resume)
        args = deepseek.argument_parser().parse_args(self.launch(
            "--test-list", "subset.json", "--datasets", "emoset",
        ))
        self.assertEqual(args.datasets, ["emoset"])
        self.assertEqual(args.output_dir, Path("outputs/deepseek_v41_flash_emotion"))

    def test_old_entry_point_forwards_default_and_custom_commands(self):
        for arguments in ((), ("--datasets", "all", "--workers", "2"),
                          ("--test-list", "subset.json", "--resume")):
            with self.subTest(arguments=arguments):
                expected = self.launch(*arguments)
                self.assertEqual(self.launch(*arguments, name="run_deepseek_emoset.sh"), expected)


if __name__ == "__main__":
    unittest.main()
