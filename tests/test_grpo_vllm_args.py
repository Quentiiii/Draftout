"""Keep explicit cache settings through the real vLLM argument parser."""

import unittest
from unittest.mock import patch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config


class VllmArgumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Constructing parser defaults inspects DeviceConfig. No engine or
        # weights are created; allow parser tests on a host without CUDA.
        with patch.object(current_platform, "device_type", "cpu"):
            cls.parser = AsyncEngineArgs.add_cli_args(FlexibleArgumentParser())

    def test_cache_flags_round_trip_explicit_false_and_true(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                cli = build_cli_args_from_config(
                    {
                        "enable_prefix_caching": enabled,
                        "enable_chunked_prefill": enabled,
                    }
                )
                parsed = self.parser.parse_args(cli)
                args = AsyncEngineArgs.from_cli_args(parsed)
                self.assertIs(args.enable_prefix_caching, enabled)
                self.assertIs(args.enable_chunked_prefill, enabled)

    def test_false_store_true_flag_remains_omitted(self):
        cli = build_cli_args_from_config({"disable_log_stats": False})
        self.assertEqual(cli, [])
        args = AsyncEngineArgs.from_cli_args(self.parser.parse_args(cli))
        self.assertFalse(args.disable_log_stats)
        # This option is a store_true flag: vLLM has no --no-disable-log-stats.
        self.assertNotIn("--no-disable-log-stats", self.parser._option_string_actions)


if __name__ == "__main__":
    unittest.main()
