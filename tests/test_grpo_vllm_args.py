"""Keep explicit cache settings through the real vLLM argument parser."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

from verl.workers.rollout.vllm_rollout.utils import build_cli_args_from_config
from verl.workers.rollout.vllm_rollout import vllm_async_server


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

    def test_server_preserves_explicit_context_limit(self):
        # Exercise the real server constructor without loading model metadata,
        # starting Ray, opening sockets, or allocating any GPU memory.
        for configured_limit, expected_limit in ((2048, 2048), (None, 262144)):
            with (
                self.subTest(configured_limit=configured_limit),
                patch.object(vllm_async_server, "omega_conf_to_dataclass", side_effect=lambda value, **kwargs: value),
                patch.object(vllm_async_server, "get_max_position_embeddings", return_value=262144),
                patch.object(vllm_async_server.ray.util, "get_node_ip_address", return_value="127.0.0.1"),
            ):
                server = vllm_async_server.vLLMHttpServer(
                    config=SimpleNamespace(max_model_len=configured_limit),
                    model_config=SimpleNamespace(hf_config=object()),
                    rollout_mode=vllm_async_server.RolloutMode.HYBRID,
                    workers=[],
                    replica_rank=0,
                    node_rank=1,
                    gpus_per_node=1,
                    nnodes=2,
                )
                self.assertEqual(server.config.max_model_len, expected_limit)
                cli = build_cli_args_from_config({"max_model_len": server.config.max_model_len})
                args = AsyncEngineArgs.from_cli_args(self.parser.parse_args(cli))
                self.assertEqual(args.max_model_len, expected_limit)


if __name__ == "__main__":
    unittest.main()
