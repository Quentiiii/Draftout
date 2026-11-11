"""CPU fake-server checks for the two-by-two rollout contract."""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf

from emotion_grpo.draft_rollout import (
    DraftAgentLoopManager,
    DraftAgentLoopWorker,
    DraftImageAgent,
    DraftRolloutError,
    split_draft_input,
    validate_draft_prefix,
)
from verl.experimental.agent_loop.agent_loop import AgentLoopManager
from verl.protocol import DataProto
from verl.workers.rollout.replica import TokenOutput


class TinyTokenizer:
    eos_token_id = 99
    pad_token_id = 0
    padding_side = "left"
    pieces = {
        0: "<pad>", 10: "<draft>", 11: "1. Color", 12: "</draft>",
        13: "</draft>\n", 14: "</draft>Analysis", 15: "2. Posture",
        16: "\n", 30: "Analysis \\boxed{Happy}", 31: "Another analysis \\boxed{Sad}",
        32: "Analysis \\boxed{Sad}", 33: "Another analysis \\boxed{Happy}",
        90: "user image question", 91: "assistant header", 99: "<eos>",
    }

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[int(value)] for value in ids)

    def encode(self, *_args, **_kwargs):
        raise AssertionError("Sampled prefix must never be re-tokenized")

    def pad(self, data, *, max_length, return_attention_mask=False, **kwargs):
        ids = list(data["input_ids"])
        padding = [self.pad_token_id] * (max_length - len(ids))
        mask = [1] * len(ids)
        zeros = [0] * len(padding)
        if self.padding_side == "left":
            ids, mask = padding + ids, zeros + mask
        else:
            ids, mask = ids + padding, mask + zeros
        result = {"input_ids": torch.tensor([ids])}
        if return_attention_mask:
            result["attention_mask"] = torch.tensor([mask])
        return result


def configuration(response_length=12):
    return OmegaConf.create({
        "actor_rollout_ref": {"rollout": {
            "n": 4, "prompt_length": 5, "response_length": response_length,
            "max_model_len": 30, "temperature": 1.0, "top_p": 1.0, "top_k": -1,
            "calculate_log_probs": True, "val_kwargs": {},
        }},
        "reward_model": {"enable": False},
    })


class FakeServer:
    def __init__(self, *, logprobs=True, draft_ids=None, suffix_ids=None):
        self.calls = []
        self.logprobs = logprobs
        self.draft_ids = draft_ids or [[10, 11, 12], [10, 15, 13]]
        self.suffix_ids = suffix_ids
        self.draft_count = 0
        self.suffix_count = 0

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["sampling_params"].get("stop"):
            ids = self.draft_ids[self.draft_count % 2]
            self.draft_count += 1
        else:
            if self.draft_count < 2:
                raise AssertionError("Suffix dispatched before both drafts")
            ids = self.suffix_ids or [30 + (self.suffix_count % 4), 99]
            self.suffix_count += 1
        await asyncio.sleep(0)
        return TokenOutput(
            token_ids=list(ids),
            log_probs=[-float(token) / 100 for token in ids] if self.logprobs else None,
            stop_reason="completed",
        )


def make_agent(server=None, response_length=12):
    agent = object.__new__(DraftImageAgent)
    agent.config = configuration(response_length)
    agent.server_manager = server or FakeServer()
    agent.tokenizer = TinyTokenizer()
    agent.validate = False
    agent.tool_schemas = []
    agent.process_vision_info = AsyncMock(return_value={"images": ["test-image"]})
    agent.apply_chat_template = AsyncMock(return_value=[90, 91])
    return agent


class DraftGenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_exact_two_by_two_shared_ids_without_another_chat_turn(self):
        agent = make_agent()
        raw_prompt = [{"role": "user", "content": "image and question"}]
        trajectories = await agent.run_draft_rollout({"temperature": 1.0, "top_p": 1.0}, raw_prompt=raw_prompt, uid="image-a")
        self.assertEqual(len(trajectories), 4)
        self.assertEqual(len(agent.server_manager.calls), 6)
        self.assertEqual(agent.apply_chat_template.await_count, 1)
        self.assertEqual(agent.apply_chat_template.call_args.args[0], raw_prompt)
        self.assertEqual(agent.process_vision_info.await_count, 1)
        for row, trajectory in enumerate(trajectories):
            j, k = divmod(row, 2)
            expected_prefix = agent.server_manager.draft_ids[j]
            self.assertEqual(trajectory.prompt_ids, [90, 91])
            self.assertEqual(trajectory.response_ids[:3], expected_prefix)
            self.assertEqual(trajectory.response_ids[-1], 99)
            self.assertEqual(trajectory.response_mask, [1] * 5)
            self.assertEqual(trajectory.num_turns, 2)
            self.assertEqual(trajectory.extra_fields, {
                "image_id": "image-a", "draft_id": j, "branch_id": k, "draft_token_count": 3,
            })
            self.assertEqual(trajectory.response_logprobs, [-float(token) / 100 for token in trajectory.response_ids])
            continuation = agent.server_manager.calls[2 + row]
            self.assertEqual(continuation["prompt_ids"], [90, 91] + expected_prefix)
            self.assertEqual(continuation["image_data"], ["test-image"])
            self.assertEqual(continuation["request_id"], agent.server_manager.calls[j]["request_id"])
            self.assertEqual(continuation["sampling_params"]["max_tokens"], 9)
            self.assertNotIn("stop", continuation["sampling_params"])
        first, second = agent.server_manager.calls[:2]
        self.assertNotEqual(first["request_id"], second["request_id"])
        for draft_call in (first, second):
            self.assertEqual(draft_call["prompt_ids"], [90, 91])
            self.assertEqual(draft_call["sampling_params"]["stop"], ["</draft>"])
            self.assertTrue(draft_call["sampling_params"]["include_stop_str_in_output"])
            self.assertFalse(draft_call["sampling_params"]["ignore_eos"])
            self.assertEqual(draft_call["sampling_params"]["max_tokens"], 11)

    async def test_budget_and_suffix_truncation_retain_sampled_tokens(self):
        # One suffix token remains: no fake EOS is appended to a length cutoff.
        agent = make_agent(FakeServer(logprobs=False, suffix_ids=[30]), response_length=4)
        trajectories = await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="i")
        self.assertTrue(all(len(trajectory.response_ids) == 4 for trajectory in trajectories))
        self.assertTrue(all(trajectory.response_logprobs is None for trajectory in trajectories))
        self.assertTrue(all(call["sampling_params"]["max_tokens"] == 1 for call in agent.server_manager.calls[2:]))

    async def test_invalid_draft_fails_before_any_continuation(self):
        server = FakeServer(draft_ids=[[10, 11, 99], [10, 15, 12]])
        agent = make_agent(server)
        with self.assertRaisesRegex(DraftRolloutError, "EOS"):
            await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="i")
        self.assertEqual(len(server.calls), 2)

    async def test_invalid_generation_limits_are_explicit_errors(self):
        for case in ("temperature", "context", "empty_suffix", "long_suffix"):
            with self.subTest(case=case):
                agent = make_agent()
                params = {"temperature": 1.0}
                if case == "temperature":
                    params["temperature"] = 0
                elif case == "context":
                    agent.config.actor_rollout_ref.rollout.max_model_len = 5
                elif case == "empty_suffix":
                    original = agent.server_manager.generate

                    async def empty_suffix(**kwargs):
                        result = await original(**kwargs)
                        if "stop" not in kwargs["sampling_params"]:
                            result.token_ids, result.log_probs = [], []
                        return result

                    agent.server_manager.generate = empty_suffix
                else:
                    agent.server_manager.suffix_ids = [30] * 20
                with self.assertRaises(DraftRolloutError):
                    await agent.run_draft_rollout(params, raw_prompt=[], uid="i")

    async def test_existing_postprocess_masks_prompt_padding_and_retains_eos(self):
        agent = make_agent()
        trajectories = await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="i")
        worker = object.__new__(DraftAgentLoopWorker)
        worker.config = configuration()
        worker.tokenizer = TinyTokenizer()
        worker.processor = None
        worker.reward_router_address = None
        worker.use_reward_loop = False
        processed = [await worker._agent_loop_postprocess(trajectory, raw_prompt=[]) for trajectory in trajectories]
        batch = worker._postprocess(processed)
        self.assertEqual(tuple(batch.batch["input_ids"].shape), (4, 17))
        self.assertEqual(batch.batch["prompts"].tolist(), [[0, 0, 0, 90, 91]] * 4)
        self.assertEqual(batch.batch["response_mask"].tolist(), [[1] * 5 + [0] * 7] * 4)
        self.assertEqual(batch.batch["attention_mask"].tolist(), [[0] * 3 + [1] * 7 + [0] * 7] * 4)
        self.assertEqual(batch.batch["responses"][:, 4].tolist(), [99] * 4)
        self.assertEqual(batch.non_tensor_batch["draft_token_count"].tolist(), [3] * 4)
        self.assertEqual(batch.non_tensor_batch["draft_id"].tolist(), [0, 0, 1, 1])
        self.assertEqual(batch.non_tensor_batch["branch_id"].tolist(), [0, 1, 0, 1])

    async def test_worker_preserves_image_order_and_original_reward_kwargs(self):
        worker = object.__new__(DraftAgentLoopWorker)
        worker.config = configuration()
        worker.tokenizer = TinyTokenizer()
        worker.processor = None
        worker.dataset_cls = object
        worker.server_manager = object()
        worker.reward_router_address = None
        worker.use_reward_loop = False
        worker.config.data = {}
        batch = DataProto.from_dict(non_tensors={
            "uid": np.array(["a", "b"], dtype=object),
            "raw_prompt": np.array([[], []], dtype=object),
            "extra_info": np.array([{"phase": "draft-a"}, {"phase": "draft-b"}], dtype=object),
            "reward_model": np.array([{"ground_truth": "Happy"}, {"ground_truth": "Sad"}], dtype=object),
        }, meta_info={"draft_rollout": True})
        observed_kwargs = []

        async def observe_score(*args, **kwargs):
            observed_kwargs.append(kwargs["kwargs"])

        worker._compute_score = observe_score
        with patch("emotion_grpo.draft_rollout.DraftImageAgent", side_effect=lambda **kwargs: make_agent()):
            output = await worker.generate_draft_sequences(batch)
        self.assertEqual(output.non_tensor_batch["image_id"].tolist(), ["a"] * 4 + ["b"] * 4)
        self.assertEqual(output.non_tensor_batch["draft_id"].tolist(), [0, 0, 1, 1] * 2)
        self.assertEqual(output.non_tensor_batch["branch_id"].tolist(), [0, 1, 0, 1] * 2)
        for uid in ("a", "b"):
            original = [entry for entry in observed_kwargs if entry["uid"] == uid]
            self.assertEqual(len(original), 4)
            self.assertTrue(all(entry["extra_info"] == {"phase": f"draft-{uid}"} for entry in original))


class DraftBoundaryTests(unittest.TestCase):
    def test_closing_tag_merged_with_whitespace_is_indivisible(self):
        self.assertEqual(validate_draft_prefix(TinyTokenizer(), [10, 11, 13], 12), 3)

    def test_malformed_prefixes_fail_without_silent_repair(self):
        cases = [
            [], [10, 11], [10, 12], [10, 11, 99], [10, 11, 14],
            [10, 11, 12, 16], [10, 11, 12, 12], [11, 12], [10, 10, 11, 12],
        ]
        for ids in cases:
            with self.subTest(ids=ids), self.assertRaises(DraftRolloutError):
                validate_draft_prefix(TinyTokenizer(), ids, 12)
        with self.assertRaisesRegex(DraftRolloutError, "budget"):
            validate_draft_prefix(TinyTokenizer(), [10, 11, 12], 3)


def repeated_images(num_images=3):
    return DataProto.from_dict(non_tensors={
        "uid": np.repeat(np.array([f"image-{i}" for i in range(num_images)], dtype=object), 4),
        "index": np.repeat(np.arange(num_images), 4),
    }, meta_info={"draft_rollout": True})


class DraftDispatchTests(unittest.TestCase):
    def test_partition_keeps_whole_images_with_uneven_or_excess_workers(self):
        for workers, expected in ((2, [2, 1]), (4, [1, 1, 1]), (1, [3])):
            with self.subTest(workers=workers):
                chunks = split_draft_input(repeated_images(), workers)
                self.assertEqual([len(chunk) for chunk in chunks], expected)
                self.assertEqual([uid for chunk in chunks for uid in chunk.non_tensor_batch["uid"]],
                                 ["image-0", "image-1", "image-2"])
                self.assertTrue(all(chunk.meta_info["draft_rollout"] for chunk in chunks))

    def test_bad_interleaving_and_missing_rows_fail(self):
        for kind in ("short", "different_uid", "duplicate", "no_uid"):
            batch = repeated_images()
            if kind == "short":
                batch = batch[:11]
            elif kind == "different_uid":
                batch.non_tensor_batch["uid"][1] = "bad"
            elif kind == "duplicate":
                batch.non_tensor_batch["uid"][4:8] = "image-0"
            else:
                batch.non_tensor_batch.pop("uid")
            with self.subTest(kind=kind), self.assertRaises(DraftRolloutError):
                split_draft_input(batch, 2)

    def test_warmup_and_validation_use_unmodified_single_turn_manager(self):
        manager = object.__new__(DraftAgentLoopManager)
        for metadata in ({"draft_rollout": False}, {"draft_rollout": True, "validate": True}, {}):
            batch = repeated_images()
            batch.meta_info = metadata
            sentinel = object()
            with patch.object(AgentLoopManager, "generate_sequences", return_value=sentinel) as legacy:
                self.assertIs(manager.generate_sequences(batch), sentinel)
                legacy.assert_called_once_with(batch)

    def test_manager_gathers_all_images_once_and_cleans_up_on_failure(self):
        manager = object.__new__(DraftAgentLoopManager)
        manager.config = configuration()
        manager.reward_model_manager = None
        manager.wake_up = Mock()
        manager.sleep = Mock()
        manager._performance_metrics = Mock(return_value={"test": 1.0})
        dispatched = []

        def generate(chunk):
            dispatched.append(chunk.non_tensor_batch["uid"].tolist())
            size = 4 * len(chunk)
            return DataProto.from_dict(
                tensors={"prompts": torch.zeros(size, 5, dtype=torch.long)},
                non_tensors={"image_id": np.repeat(chunk.non_tensor_batch["uid"], 4)},
                meta_info={"metrics": [{"generate_sequences": 1.0}] * size},
            )

        manager.agent_loop_workers = [
            SimpleNamespace(generate_draft_sequences=SimpleNamespace(remote=generate)) for _ in range(2)
        ]
        with patch("emotion_grpo.draft_rollout.ray.get", side_effect=lambda values: values):
            output = manager.generate_sequences(repeated_images())
        self.assertEqual(dispatched, [["image-0", "image-1"], ["image-2"]])
        self.assertEqual(output.non_tensor_batch["image_id"].tolist(),
                         ["image-0"] * 4 + ["image-1"] * 4 + ["image-2"] * 4)
        manager.wake_up.assert_called_once()
        manager.sleep.assert_called_once()
        manager.sleep.reset_mock()
        with patch("emotion_grpo.draft_rollout.ray.get", side_effect=DraftRolloutError("draft invalid")):
            with self.assertRaisesRegex(DraftRolloutError, "invalid"):
                manager.generate_sequences(repeated_images())
        manager.sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()
