"""Two sampled drafts, with two token-exact continuations per draft.

This is an AgentLoopManager extension, not a second rollout backend.  The
ordinary single-turn agent remains responsible for warm-up and validation.
Complete trajectories go through the same multimodal/padding/reward postprocessing as
ordinary rollouts.  All six generation requests for an image finish before
the manager returns control to the trainer (and hence before any update).

A malformed or truncated draft fails the batch explicitly.  We never append
a missing delimiter, retokenize a sampled prefix, silently resample a draft,
or train an image group with fewer than four trajectories.  Whitespace merged into the final
``</draft>`` token belongs to that indivisible draft boundary token.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import uuid4

import numpy as np
import ray

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
)
from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop
from verl.protocol import DataProto
from verl.utils.transferqueue_utils import tqbridge


NUM_DRAFTS = 2
NUM_BRANCHES = 2
TRAJECTORIES_PER_IMAGE = NUM_DRAFTS * NUM_BRANCHES


class DraftRolloutError(ValueError):
    """An incomplete image group cannot be used for the specified estimator."""


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def validate_draft_prefix(tokenizer, token_ids: list[int], response_length: int) -> int:
    """Validate the stopped draft without modifying any sampled token IDs."""
    if not token_ids or len(token_ids) >= response_length:
        raise DraftRolloutError("Draft exhausted the total response budget; no suffix tokens remain.")
    eos = tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    if any(token in eos_ids for token in token_ids):
        raise DraftRolloutError("Draft stopped at EOS before a valid </draft> boundary.")
    text = _decode(tokenizer, token_ids)
    opening, closing = "<draft>", "</draft>"
    if not text.lstrip().startswith(opening) or text.count(opening) != 1 or text.count(closing) != 1:
        raise DraftRolloutError("Draft must contain one opening <draft> and one closing </draft> tag.")
    start = text.index(opening) + len(opening)
    end = text.index(closing)
    if end < start or not text[start:end].strip():
        raise DraftRolloutError("Draft content is empty or its delimiters are reversed.")
    if text[end + len(closing) :].strip():
        raise DraftRolloutError("Draft stop token also contains analysis; cannot split a sampled token.")
    # vLLM stops after the first token completing the string. Extra whole
    # tokens after the delimiter are a server contract violation, not tokens
    # that we can silently trim or classify as sampled continuations.
    if len(token_ids) > 1 and closing in _decode(tokenizer, token_ids[:-1]):
        raise DraftRolloutError("Draft server returned extra tokens after the </draft> boundary.")
    return len(token_ids)


def split_draft_input(prompts: DataProto, num_workers: int) -> list[DataProto]:
    """Collapse the trainer's interleaved n=4 rows, then split whole images."""
    if num_workers < 1 or not len(prompts) or len(prompts) % TRAJECTORIES_PER_IMAGE:
        raise DraftRolloutError("Draft rollout requires a nonempty batch of four rows per image and a worker.")
    if "uid" not in prompts.non_tensor_batch:
        raise DraftRolloutError("Draft rollout requires the trainer's per-image uid.")
    uids = prompts.non_tensor_batch["uid"].tolist()
    image_uids = []
    for first in range(0, len(prompts), TRAJECTORIES_PER_IMAGE):
        image_uid = uids[first]
        if any(uid != image_uid for uid in uids[first : first + TRAJECTORIES_PER_IMAGE]):
            raise DraftRolloutError("Expected four interleaved copies of each image (same uid).")
        image_uids.append(image_uid)
    if len(set(image_uids)) != len(image_uids):
        raise DraftRolloutError("Each rollout image must have a unique uid.")
    image_rows = np.arange(0, len(prompts), TRAJECTORIES_PER_IMAGE)
    # np.array_split supports uneven B and B < worker count without padding
    # artificial images into the variance statistics.
    return [prompts[rows] for rows in np.array_split(image_rows, min(num_workers, len(image_rows)))]


class DraftImageAgent(SingleTurnAgentLoop):
    """Reuse single-turn prompt preparation; branch on sampled token IDs."""

    async def run_draft_rollout(self, sampling_params: dict[str, Any], **kwargs) -> list[AgentLoopOutput]:
        messages = list(kwargs["raw_prompt"])
        multi_modal_data = await self.process_vision_info(messages)
        images, videos = multi_modal_data.get("images"), multi_modal_data.get("videos")
        prompt_ids = await self.apply_chat_template(messages, tools=self.tool_schemas, images=images, videos=videos)
        response_length = self._get_response_length()
        if response_length < 2:
            raise DraftRolloutError("Draft rollout needs a total response budget of at least two tokens.")
        max_model_len = self.config.actor_rollout_ref.rollout.get("max_model_len")
        if max_model_len is not None and len(prompt_ids) + response_length > max_model_len:
            raise DraftRolloutError("Prompt plus complete trajectory response budget exceeds max_model_len.")
        if sampling_params.get("temperature", 1.0) <= 0:
            raise DraftRolloutError("Drafts and continuations must be sampled (temperature > 0).")
        if sampling_params.get("n", 1) != 1:
            raise DraftRolloutError("Each generation request samples one response; branching is explicit.")

        async def generate(ids, params, request_id):
            started = time.perf_counter()
            result = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=list(ids),
                sampling_params=dict(params),
                image_data=images,
                video_data=videos,
            )
            if result.stop_reason == "aborted":
                raise DraftRolloutError("Draft rollout generation was aborted.")
            if result.routed_experts is not None:
                raise DraftRolloutError("Draft rollout does not yet support rollout routing replay.")
            if result.log_probs is not None and len(result.log_probs) != len(result.token_ids):
                raise DraftRolloutError("Rollout token and log-probability lengths differ.")
            return result, time.perf_counter() - started

        draft_params = {
            **sampling_params,
            "max_tokens": response_length - 1,
            "stop": ["</draft>"],
            "include_stop_str_in_output": True,
            "ignore_eos": False,
        }
        request_ids = [uuid4().hex for _ in range(NUM_DRAFTS)]
        drafts = await asyncio.gather(
            *(generate(prompt_ids, draft_params, request_id) for request_id in request_ids)
        )
        # Validate both drafts before dispatching any continuations for this
        # image. The prefix is copied as IDs, never encoded from decoded text.
        draft_lengths = [validate_draft_prefix(self.tokenizer, d.token_ids, response_length) for d, _ in drafts]

        async def continue_draft(draft_id, branch_id):
            draft, draft_elapsed = drafts[draft_id]
            prefix_ids = list(draft.token_ids)
            remaining = response_length - draft_lengths[draft_id]
            suffix_params = {**sampling_params, "max_tokens": remaining, "ignore_eos": False}
            # The draft stop string applies only to the first stage.
            suffix_params.pop("stop", None)
            suffix_params.pop("include_stop_str_in_output", None)
            suffix, suffix_elapsed = await generate(
                prompt_ids + prefix_ids, suffix_params, request_ids[draft_id]
            )
            if not suffix.token_ids or len(suffix.token_ids) > remaining:
                raise DraftRolloutError("Suffix is empty or exceeds the shared response budget.")
            # Length-cutoff suffixes remain ordinary unsuccessful/parseable
            # task trajectories, exactly as in single-turn GRPO. Any sampled
            # EOS is retained and receives suffix advantage.
            logprobs = None
            if (draft.log_probs is None) != (suffix.log_probs is None):
                raise DraftRolloutError("Draft and suffix must both provide log probabilities or neither.")
            if draft.log_probs is not None:
                logprobs = list(draft.log_probs) + list(suffix.log_probs)
            full_ids = prefix_ids + list(suffix.token_ids)
            return AgentLoopOutput(
                prompt_ids=list(prompt_ids),
                response_ids=full_ids,
                response_mask=[1] * len(full_ids),
                response_logprobs=logprobs,
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics={"generate_sequences": draft_elapsed + suffix_elapsed},
                extra_fields={
                    "image_id": str(kwargs["uid"]),
                    "draft_id": draft_id,
                    "branch_id": branch_id,
                    "draft_token_count": draft_lengths[draft_id],
                },
            )

        return await asyncio.gather(
            *(continue_draft(j, k) for j in range(NUM_DRAFTS) for k in range(NUM_BRANCHES))
        )


class DraftAgentLoopWorker(AgentLoopWorker):
    """The ordinary worker plus a method accepting one row per image."""

    @tqbridge()
    async def generate_draft_sequences(self, batch: DataProto) -> DataProto:
        if batch.meta_info.get("validate", False) or not batch.meta_info.get("draft_rollout", False):
            raise DraftRolloutError("Draft worker is only valid for the draft training phase.")
        config = self.config.actor_rollout_ref.rollout
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.get("repetition_penalty", 1.0),
            logprobs=config.calculate_log_probs,
        )

        async def run_image(row):
            kwargs = {key: values[row] for key, values in batch.non_tensor_batch.items()}
            agent = DraftImageAgent(
                trainer_config=DictConfigWrap(self.config),
                server_manager=self.server_manager,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                dataset_config=self.config.data,
            )
            trajectories = await agent.run_draft_rollout(sampling_params, **kwargs)
            return [await self._agent_loop_postprocess(trajectory, __validate__=False, **kwargs) for trajectory in trajectories]

        outputs_by_image = await asyncio.gather(*(run_image(row) for row in range(len(batch))))
        return self._postprocess([trajectory for trajectories in outputs_by_image for trajectory in trajectories])


class DraftAgentLoopManager(AgentLoopManager):
    """Dispatch complete image groups, retaining the trainer's original order."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(DraftAgentLoopWorker)
        super().__init__(*args, **kwargs)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        if prompts.meta_info.get("validate", False) or not prompts.meta_info.get("draft_rollout", False):
            return super().generate_sequences(prompts)
        if self.config.actor_rollout_ref.rollout.n != TRAJECTORIES_PER_IMAGE:
            raise DraftRolloutError("Draft rollout requires rollout.n=4 (two drafts times two continuations).")
        chunks = split_draft_input(prompts, len(self.agent_loop_workers))
        self.wake_up()
        if self.reward_model_manager:
            self.reward_model_manager.wake_up()
        try:
            outputs = ray.get(
                [
                    worker.generate_draft_sequences.remote(chunk)
                    for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False)
                ]
            )
            outputs = self._align_output_prompt_length(outputs)
            output = DataProto.concat(outputs)
            if len(output) != len(prompts):
                raise DraftRolloutError("Draft rollout did not return exactly four trajectories per image.")
            metrics = [chunk.meta_info.pop("metrics") for chunk in outputs]
            timing = self._performance_metrics(metrics, output)
            output.meta_info = {"timing": timing, **outputs[0].meta_info}
            return output
        finally:
            self.sleep()
            if self.reward_model_manager:
                self.reward_model_manager.sleep()
