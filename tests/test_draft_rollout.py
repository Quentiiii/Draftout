"""CPU fake-server checks for the two-by-two rollout contract."""

import asyncio
import concurrent.futures
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import torch
from omegaconf import OmegaConf
from ray.exceptions import ActorDiedError, RayTaskError, WorkerCrashedError

from emotion_grpo.draft_rollout import (
    DraftAgentLoopManager,
    DraftAgentLoopWorker,
    DraftImageAgent,
    DraftRolloutError,
    _gather_settled,
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
    async def test_first_observed_error_is_preserved_after_later_sibling_failure(self):
        release = asyncio.Event()
        failed = asyncio.Event()
        first = DraftRolloutError("earliest failure")
        second = RuntimeError("later failure from first input")

        async def delayed_failure():
            await release.wait()
            raise second

        async def immediate_failure():
            failed.set()
            raise first

        pending = asyncio.create_task(_gather_settled(delayed_failure(), immediate_failure()))
        await asyncio.wait_for(failed.wait(), timeout=2)
        self.assertFalse(pending.done())
        release.set()
        with self.assertRaises(DraftRolloutError) as caught:
            await pending
        self.assertIs(caught.exception, first)

    async def test_failed_generation_drains_inflight_drafts_and_continuations(self):
        for stage in ("draft", "continuation"):
            with self.subTest(stage=stage):
                release = asyncio.Event()
                blocked = asyncio.Event()
                failure = DraftRolloutError(f"original {stage} failure")

                class RacingServer(FakeServer):
                    def __init__(self):
                        super().__init__()
                        self.stage_calls = 0
                        self.finished_sibling = False

                    async def generate(self, **kwargs):
                        is_draft = bool(kwargs["sampling_params"].get("stop"))
                        if is_draft == (stage == "draft"):
                            index = self.stage_calls
                            self.stage_calls += 1
                            if index == 0:
                                raise failure
                            if index == 1:
                                blocked.set()
                                await release.wait()
                                self.finished_sibling = True
                        return await super().generate(**kwargs)

                server = RacingServer()
                agent = make_agent(server)
                with self.assertLogs("emotion_grpo.draft_rollout", level="ERROR"):
                    pending = asyncio.create_task(agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="race"))
                    try:
                        await asyncio.wait_for(blocked.wait(), timeout=2)
                        await asyncio.sleep(0)
                        self.assertFalse(pending.done(), "Failure escaped while another server request was active")
                        self.assertFalse(server.finished_sibling)
                    finally:
                        release.set()
                    with self.assertRaises(DraftRolloutError) as caught:
                        await pending
                self.assertIs(caught.exception, failure)
                self.assertTrue(server.finished_sibling)
                self.assertEqual(server.stage_calls, 2 if stage == "draft" else 4)

    async def test_caller_cancellation_does_not_cancel_or_abandon_server_requests(self):
        release = asyncio.Event()
        both_started = asyncio.Event()
        started, finished = [], []

        class BlockingServer(FakeServer):
            async def generate(self, **kwargs):
                started.append(kwargs["request_id"])
                if len(started) == 2:
                    both_started.set()
                await release.wait()
                result = await super().generate(**kwargs)
                finished.append(kwargs["request_id"])
                return result

        agent = make_agent(BlockingServer())
        pending = asyncio.create_task(agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="cancel"))
        try:
            await asyncio.wait_for(both_started.wait(), timeout=2)
            pending.cancel()
            await asyncio.sleep(0)
            self.assertFalse(pending.done())
            self.assertEqual(finished, [])
        finally:
            release.set()
        with self.assertRaises(asyncio.CancelledError):
            await pending
        self.assertCountEqual(finished, started)
        self.assertEqual(len(started), 2, "Cancelled requests must not be reissued or continued")

    async def test_backend_image_failure_waits_for_other_images_and_logs_original_error(self):
        release = asyncio.Event()
        suffix_started = asyncio.Event()
        completed_suffixes = []

        class SlowImageServer(FakeServer):
            async def generate(self, **kwargs):
                if not kwargs["sampling_params"].get("stop"):
                    suffix_started.set()
                    await release.wait()
                    completed_suffixes.append(kwargs["request_id"])
                return await super().generate(**kwargs)

        class AbortedServer(FakeServer):
            async def generate(self, **kwargs):
                result = await super().generate(**kwargs)
                result.stop_reason = "aborted"
                return result

        bad_agent = make_agent(AbortedServer())
        slow_agent = make_agent(SlowImageServer())
        worker = object.__new__(DraftAgentLoopWorker)
        worker.config = configuration()
        worker.config.data = {}
        worker.tokenizer = TinyTokenizer()
        worker.processor = None
        worker.dataset_cls = object
        worker.server_manager = object()
        worker._agent_loop_postprocess = AsyncMock(side_effect=lambda trajectory, **kwargs: trajectory)
        batch = DataProto.from_dict(non_tensors={
            "uid": np.array(["bad-image", "slow-image"], dtype=object),
            "raw_prompt": np.array([[], []], dtype=object),
        }, meta_info={"draft_rollout": True, "global_steps": 15})
        with patch("emotion_grpo.draft_rollout.DraftImageAgent", side_effect=[bad_agent, slow_agent]):
            with self.assertLogs("emotion_grpo.draft_rollout", level="ERROR") as captured:
                pending = asyncio.create_task(worker.generate_draft_sequences(batch))
                try:
                    await asyncio.wait_for(suffix_started.wait(), timeout=2)
                    await asyncio.sleep(0)
                    self.assertFalse(pending.done(), "Worker failed while another image still generated")
                    self.assertEqual(completed_suffixes, [])
                finally:
                    release.set()
                with self.assertRaisesRegex(DraftRolloutError, "aborted"):
                    await pending
        self.assertEqual(len(completed_suffixes), 4)
        messages = "\n".join(captured.output)
        for detail in ("image_id=bad-image", "draft_id=0", "aborted", "global_step=15"):
            self.assertIn(detail, messages)

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
                "draft_fallback": False, "draft_fallback_reason": "",
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

    async def test_eos_without_draft_retains_original_and_samples_one_fresh_response(self):
        initial = [30, 99]  # A complete analysis/boxed answer, with no draft tags.
        fresh = [10, 11, 12, 31, 99]
        server = FakeServer(draft_ids=[initial, [10, 15, 12]], suffix_ids=fresh)
        agent = make_agent(server)
        with self.assertLogs("emotion_grpo.draft_rollout", level="WARNING") as captured:
            output = await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="i")
        self.assertEqual(len(output), 4)
        self.assertEqual(len(server.calls), 5)
        self.assertEqual(output[0].response_ids, initial)
        self.assertEqual(output[1].response_ids, fresh)
        self.assertEqual(server.calls[2]["prompt_ids"], [90, 91])
        self.assertEqual(server.calls[2]["sampling_params"]["max_tokens"], 12)
        self.assertNotIn("stop", server.calls[2]["sampling_params"])
        self.assertTrue(all(99 not in call["prompt_ids"] for call in server.calls))
        self.assertEqual([row.extra_fields["draft_token_count"] for row in output], [0, 0, 3, 3])
        self.assertEqual([row.extra_fields["draft_fallback"] for row in output], [True, True, False, False])
        for row in output:
            self.assertEqual(row.response_logprobs, [-float(token) / 100 for token in row.response_ids])
            self.assertEqual(row.prompt_ids, [90, 91])
        messages = "\n".join(captured.output)
        for detail in ("image_id=i", "draft_id=0", "reason=", "EOS", "token_count=2", "token_tail=[30, 99]"):
            self.assertIn(detail, messages)
        self.assertNotIn("Traceback", messages)

    async def test_malformed_unfinished_draft_continues_exact_prefix_twice_with_real_budget(self):
        cases = [[11, 12], [10, 12], [10, 11, 14], [10, 11], [10, 11, 12, 16]]
        for prefix in cases:
            with self.subTest(prefix=prefix):
                server = FakeServer(draft_ids=[prefix, [10, 15, 12]])
                agent = make_agent(server)
                with self.assertLogs("emotion_grpo.draft_rollout", level="WARNING"):
                    output = await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="i")
                self.assertEqual(len(output), 4)
                self.assertEqual(len(server.calls), 6)
                for branch in range(2):
                    row = output[branch]
                    self.assertEqual(row.response_ids[:len(prefix)], prefix)
                    self.assertEqual(row.extra_fields["draft_token_count"], 0)
                    self.assertTrue(row.extra_fields["draft_fallback"])
                    self.assertTrue(row.extra_fields["draft_fallback_reason"])
                    self.assertEqual(row.response_logprobs, [-float(token) / 100 for token in row.response_ids])
                    call = server.calls[2 + branch]
                    self.assertEqual(call["prompt_ids"], [90, 91] + prefix)
                    self.assertEqual(call["sampling_params"]["max_tokens"], 12 - len(prefix))
                    self.assertNotIn("stop", call["sampling_params"])

    async def test_empty_initial_draft_samples_two_full_responses_without_revalidation(self):
        server = FakeServer(draft_ids=[[], [10, 15, 12]], suffix_ids=[10, 11, 12, 30, 99])
        agent = make_agent(server)
        with self.assertLogs("emotion_grpo.draft_rollout", level="WARNING"):
            output = await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="empty")
        self.assertEqual(len(server.calls), 6)
        self.assertEqual(len(output), 4)
        for branch in range(2):
            self.assertEqual(server.calls[2 + branch]["prompt_ids"], [90, 91])
            self.assertEqual(server.calls[2 + branch]["sampling_params"]["max_tokens"], 12)
            self.assertEqual(output[branch].response_ids, [10, 11, 12, 30, 99])
            self.assertEqual(output[branch].extra_fields["draft_token_count"], 0)
            self.assertTrue(output[branch].extra_fields["draft_fallback"])
            self.assertEqual(output[branch].response_logprobs, [-float(token) / 100 for token in output[branch].response_ids])

    async def test_truncated_invalid_draft_preserves_all_tokens_and_one_token_suffix_budget(self):
        prefix = [10, 11, 15]
        server = FakeServer(logprobs=False, draft_ids=[prefix, [10, 15, 12]], suffix_ids=[30])
        agent = make_agent(server, response_length=4)
        with self.assertLogs("emotion_grpo.draft_rollout", level="WARNING"):
            output = await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="cutoff")
        self.assertEqual(len(server.calls), 6)
        for row in output[:2]:
            self.assertEqual(row.response_ids, prefix + [30])
            self.assertEqual(row.extra_fields["draft_token_count"], 0)
            self.assertIsNone(row.response_logprobs)
        self.assertTrue(all(call["sampling_params"]["max_tokens"] == 1 for call in server.calls[2:]))

    async def test_over_budget_initial_generation_is_a_backend_error_not_format_fallback(self):
        server = FakeServer(draft_ids=[[30] * 12, [10, 15, 12]])
        agent = make_agent(server)
        with self.assertLogs("emotion_grpo.draft_rollout", level="ERROR"), \
                self.assertRaisesRegex(DraftRolloutError, "exceeded.*max_tokens"):
            await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="overflow")
        self.assertEqual(len(server.calls), 2)

    async def test_fallback_with_empty_fresh_response_is_still_a_backend_error(self):
        server = FakeServer(draft_ids=[[], [10, 15, 12]])
        original = server.generate

        async def empty_continuation(**kwargs):
            result = await original(**kwargs)
            if not kwargs["sampling_params"].get("stop"):
                result.token_ids, result.log_probs = [], []
            return result

        server.generate = empty_continuation
        with self.assertLogs("emotion_grpo.draft_rollout", level="WARNING"), \
                self.assertRaisesRegex(DraftRolloutError, "Continuation returned no tokens"):
            await make_agent(server).run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="empty-backend")
        self.assertEqual(len(server.calls), 6)

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

    async def test_fallback_uses_existing_postprocess_with_variable_lengths_and_metadata(self):
        from emotion_grpo.draft_training import DraftTrainingController
        from emotion_grpo.reward import compute_draft_score

        class EmotionTokenizer(TinyTokenizer):
            pieces = {**TinyTokenizer.pieces, 30: "Analysis \\boxed{Amusement}", 31: "Analysis \\boxed{Sadness}"}

            def decode(self, ids, **kwargs):
                if kwargs.get("skip_special_tokens"):
                    ids = [token for token in ids if token not in (self.eos_token_id, self.pad_token_id)]
                return super().decode(ids, **kwargs)

        class AlternatingRewardServer(FakeServer):
            async def generate(self, **kwargs):
                suffix_index = self.suffix_count
                result = await super().generate(**kwargs)
                if not kwargs["sampling_params"].get("stop"):
                    # Fresh fallback: wrong; normal draft continuations: right,
                    # wrong. Together with the preserved answer: [1,0,1,0].
                    result.token_ids = [31 if suffix_index % 2 == 0 else 30, 99]
                    result.log_probs = [-float(token) / 100 for token in result.token_ids]
                return result

        server = AlternatingRewardServer(draft_ids=[[30, 99], [10, 15, 12]])
        agent = make_agent(server)
        agent.tokenizer = EmotionTokenizer()
        with self.assertLogs("emotion_grpo.draft_rollout", level="WARNING"):
            trajectories = await agent.run_draft_rollout({"temperature": 1.0}, raw_prompt=[], uid="i")
        worker = object.__new__(DraftAgentLoopWorker)
        worker.config = configuration()
        worker.tokenizer = agent.tokenizer
        worker.processor = None
        worker.reward_router_address = None
        worker.use_reward_loop = False
        processed = [await worker._agent_loop_postprocess(row, raw_prompt=[]) for row in trajectories]
        batch = worker._postprocess(processed)
        self.assertEqual(tuple(batch.batch["input_ids"].shape), (4, 17))
        self.assertEqual(batch.batch["prompts"].tolist(), [[0, 0, 0, 90, 91]] * 4)
        self.assertEqual(batch.batch["response_mask"].sum(dim=1).tolist(), [2, 2, 5, 5])
        for index, row in enumerate(trajectories):
            length = len(row.response_ids)
            self.assertEqual(batch.batch["responses"][index, :length].tolist(), row.response_ids)
            self.assertEqual(row.response_ids[-1], 99)
            self.assertEqual(batch.batch["response_mask"][index, length:].tolist(), [0] * (12 - length))
        self.assertEqual(batch.non_tensor_batch["draft_token_count"].tolist(), [0, 0, 3, 3])
        self.assertEqual(batch.non_tensor_batch["draft_fallback"].tolist(), [True, True, False, False])
        reasons = batch.non_tensor_batch["draft_fallback_reason"].tolist()
        self.assertEqual(reasons[0], reasons[1])
        self.assertTrue(reasons[0])
        self.assertEqual(reasons[2:], ["", ""])

        rewards = [
            compute_draft_score(
                "emoset", agent.tokenizer.decode(row.response_ids, skip_special_tokens=True), "Amusement",
                extra_info={"draft_grpo_warmup": False},
            )
            for row in trajectories
        ]
        self.assertEqual([reward["score"] for reward in rewards], [1, 0, 1, 0])
        self.assertEqual([reward["task_reward"] for reward in rewards], [1, 0, 1, 0])
        scores = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
        for index, row in enumerate(trajectories):
            scores[index, len(row.response_ids) - 1] = rewards[index]["task_reward"]
        batch.batch["token_level_scores"] = scores
        batch.batch["token_level_rewards"] = scores.clone()
        controller = DraftTrainingController({})
        controller.begin_session(0)
        batch, metrics = controller.compute_advantage(batch, global_step=21)
        advantages = batch.batch["advantages"]
        self.assertEqual(advantages.dtype, torch.bfloat16)
        for row, signed_value in enumerate((0.5, -0.5, 0.5, -0.5)):
            length = len(trajectories[row].response_ids)
            draft_length = trajectories[row].extra_fields["draft_token_count"]
            self.assertEqual(advantages[row, :draft_length].tolist(), [0.0] * draft_length)
            self.assertEqual(advantages[row, draft_length:length].tolist(), [signed_value] * (length - draft_length))
            self.assertEqual(advantages[row, length:].tolist(), [0.0] * (12 - length))
        self.assertEqual(metrics["draft/fallback_draft_count"], 1)
        self.assertEqual(metrics["draft/fallback_draft_fraction"], 0.5)
        self.assertEqual(metrics["draft/warmup"], 0.0)

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
        with patch("emotion_grpo.draft_rollout.ray.get", side_effect=DraftRolloutError("draft invalid")), \
                patch("emotion_grpo.draft_rollout.ray.wait", side_effect=lambda refs, **kwargs: (refs, [])), \
                self.assertLogs("emotion_grpo.draft_rollout", level="ERROR"):
            with self.assertRaisesRegex(DraftRolloutError, "invalid"):
                manager.generate_sequences(repeated_images())
        manager.sleep.assert_called_once()

    def test_failed_worker_cannot_sleep_until_sibling_completes_and_cleanup_preserves_original(self):
        manager = object.__new__(DraftAgentLoopManager)
        manager.config = configuration()
        manager.wake_up = Mock()
        manager.sleep = Mock(side_effect=RuntimeError("secondary rollout cleanup failure"))
        manager.reward_model_manager = SimpleNamespace(
            wake_up=Mock(), sleep=Mock(side_effect=RuntimeError("secondary reward cleanup failure"))
        )
        original = DraftRolloutError("first worker failure")
        failed, running = concurrent.futures.Future(), concurrent.futures.Future()
        failed.set_exception(original)
        manager.agent_loop_workers = [
            SimpleNamespace(generate_draft_sequences=SimpleNamespace(remote=Mock(return_value=ref)))
            for ref in (failed, running)
        ]
        drain_started = threading.Event()

        def get(refs):
            if isinstance(refs, list):
                return [ref.result() for ref in refs]
            return refs.result()

        def wait(refs, **kwargs):
            self.assertEqual(kwargs, {"num_returns": 2, "fetch_local": False})
            drain_started.set()
            concurrent.futures.wait(refs)
            return refs, []

        with patch("emotion_grpo.draft_rollout.ray.get", side_effect=get), \
                patch("emotion_grpo.draft_rollout.ray.wait", side_effect=wait), \
                self.assertLogs("emotion_grpo.draft_rollout", level="ERROR") as captured:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                pending = executor.submit(manager.generate_sequences, repeated_images(num_images=2))
                try:
                    self.assertTrue(drain_started.wait(timeout=2))
                    self.assertFalse(pending.done())
                    manager.sleep.assert_not_called()
                    manager.reward_model_manager.sleep.assert_not_called()
                finally:
                    running.set_result(object())
                with self.assertRaises(DraftRolloutError) as caught:
                    pending.result(timeout=2)
        self.assertIs(caught.exception, original)
        manager.sleep.assert_called_once()
        manager.reward_model_manager.sleep.assert_called_once()
        self.assertIn("secondary rollout cleanup failure", "\n".join(captured.output))
        self.assertIn("secondary reward cleanup failure", "\n".join(captured.output))

    def test_unconfirmed_worker_completion_skips_gpu_sleep(self):
        manager = object.__new__(DraftAgentLoopManager)
        manager.config = configuration()
        manager.wake_up = Mock()
        manager.sleep = Mock()
        manager.reward_model_manager = None
        manager.agent_loop_workers = [
            SimpleNamespace(generate_draft_sequences=SimpleNamespace(remote=Mock(return_value=object())))
        ]
        original = DraftRolloutError("original generation failure")
        with patch("emotion_grpo.draft_rollout.ray.get", side_effect=original), \
                patch("emotion_grpo.draft_rollout.ray.wait", side_effect=RuntimeError("Ray unavailable")), \
                self.assertLogs("emotion_grpo.draft_rollout", level="ERROR"):
            with self.assertRaises(DraftRolloutError) as caught:
                manager.generate_sequences(repeated_images(num_images=1))
        self.assertIs(caught.exception, original)
        manager.sleep.assert_not_called()

    def test_ready_ref_from_dead_worker_is_not_evidence_of_drained_server_requests(self):
        task_error = RayTaskError("generate", "Traceback: invalid draft", DraftRolloutError("invalid"), ip="127.0.0.1")
        for error, should_sleep in ((ActorDiedError(), False), (WorkerCrashedError(), False), (task_error, True)):
            with self.subTest(error=type(error).__name__):
                manager = object.__new__(DraftAgentLoopManager)
                manager.config = configuration()
                manager.wake_up = Mock()
                manager.sleep = Mock()
                manager.reward_model_manager = None
                manager.agent_loop_workers = [
                    SimpleNamespace(generate_draft_sequences=SimpleNamespace(remote=Mock(return_value=object())))
                ]
                with patch("emotion_grpo.draft_rollout.ray.get", side_effect=error), \
                        patch("emotion_grpo.draft_rollout.ray.wait", side_effect=lambda refs, **kwargs: (refs, [])), \
                        self.assertLogs("emotion_grpo.draft_rollout", level="ERROR"):
                    with self.assertRaises(type(error)) as caught:
                        manager.generate_sequences(repeated_images(num_images=1))
                self.assertIs(caught.exception, error)
                self.assertEqual(manager.sleep.call_count, int(should_sleep))


if __name__ == "__main__":
    unittest.main()
