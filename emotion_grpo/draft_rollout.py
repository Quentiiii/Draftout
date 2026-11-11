"""Two sampled drafts, with two token-exact continuations per draft.

This is an AgentLoopManager extension, not a second rollout backend.  The
ordinary single-turn agent remains responsible for warm-up and validation.
Complete trajectories go through the same multimodal/padding/reward postprocessing as
ordinary rollouts. All generation requests for an image finish before
the manager returns control to the trainer (and hence before any update).

A malformed draft falls back to treating its complete trajectories as suffix
tokens. An unfinished response keeps its exact sampled prefix for both
continuations. An already terminated response is retained as one branch;
the other is independently sampled from the original prompt. Empty initial
responses receive two independent full completions. We never add delimiters,
retokenize sampled tokens, or repeatedly retry an invalid format.
Whitespace merged into the final
``</draft>`` token belongs to that indivisible draft boundary token.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from uuid import uuid4

import numpy as np
import ray
from ray.exceptions import RayActorError, RayError, RayTaskError, WorkerCrashedError

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
logger = logging.getLogger(__name__)


class DraftRolloutError(ValueError):
    """An incomplete image group cannot be used for the specified estimator."""


async def _gather_settled(*awaitables):
    """Drain every submitted request before propagating the first failure.

    Cancelling a local coroutine does not abort an already submitted vLLM
    request. Shield the group even from caller cancellation, and never cancel
    siblings merely because one request failed. Results retain input order;
    the first observed exception retains its original traceback.
    """
    first_error = []

    async def observe(awaitable):
        try:
            return await awaitable
        except BaseException as error:
            if not first_error:
                first_error.append(error)
            return error

    group = asyncio.gather(*(observe(awaitable) for awaitable in awaitables), return_exceptions=True)
    while not group.done():
        try:
            await asyncio.shield(group)
        except asyncio.CancelledError as error:
            if not first_error:
                first_error.append(error)
            # Keep awaiting the same requests; do not cancel and reissue them.
    results = group.result()
    if first_error:
        raise first_error[0]
    return results


def _decode(tokenizer, ids: list[int]) -> str:
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _draft_preview(tokenizer, ids: list[int]) -> str:
    """Bounded diagnostics must never replace the original rollout error."""
    try:
        text = _decode(tokenizer, ids)
    except Exception as error:
        return f"<decode failed: {type(error).__name__}: {error}>"[:512]
    return text if len(text) <= 512 else text[:256] + " ... " + text[-251:]


def validate_draft_prefix(tokenizer, token_ids: list[int], response_length: int) -> int:
    """Validate the stopped draft without modifying any sampled token IDs."""
    if not token_ids:
        raise DraftRolloutError("Draft stage returned no tokens.")
    if len(token_ids) >= response_length:
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
    # tokens after the delimiter prevent a clean sampled boundary. Retain
    # them in the fallback trajectory instead of silently trimming tokens.
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
        image_id = str(kwargs["uid"])
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

        async def generate(ids, params, request_id, *, draft_id, branch_id=None):
            started = time.perf_counter()
            try:
                result = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=list(ids),
                    sampling_params=dict(params),
                    image_data=images,
                    video_data=videos,
                )
                if result.stop_reason == "aborted":
                    raise DraftRolloutError("Draft rollout generation was aborted.")
                if len(result.token_ids) > params["max_tokens"]:
                    raise DraftRolloutError("Generation exceeded its requested max_tokens budget.")
                if result.routed_experts is not None:
                    raise DraftRolloutError("Draft rollout does not yet support rollout routing replay.")
                if result.log_probs is not None and len(result.log_probs) != len(result.token_ids):
                    raise DraftRolloutError("Rollout token and log-probability lengths differ.")
                return result, time.perf_counter() - started
            except BaseException:
                logger.exception(
                    "Draft generation failed: image_id=%s draft_id=%s branch_id=%s request_id=%s",
                    image_id, draft_id, branch_id, request_id,
                )
                raise

        draft_params = {
            **sampling_params,
            "max_tokens": response_length - 1,
            "stop": ["</draft>"],
            "include_stop_str_in_output": True,
            "ignore_eos": False,
        }
        request_ids = [uuid4().hex for _ in range(NUM_DRAFTS)]
        drafts = await _gather_settled(
            *(generate(prompt_ids, draft_params, request_id, draft_id=j) for j, request_id in enumerate(request_ids))
        )
        # Decide segment boundaries before dispatching continuations. A format
        # fallback keeps sampled IDs intact and assigns no tokens to draft.
        draft_lengths = []
        fallback_reasons = []
        for draft_id, (draft, _) in enumerate(drafts):
            try:
                draft_lengths.append(validate_draft_prefix(self.tokenizer, draft.token_ids, response_length))
                fallback_reasons.append("")
            except DraftRolloutError as error:
                draft_lengths.append(0)
                fallback_reasons.append(str(error))
                logger.warning(
                    "Draft format fallback: image_id=%s draft_id=%s reason=%s token_count=%s token_tail=%s text_preview=%r",
                    image_id, draft_id, error, len(draft.token_ids), draft.token_ids[-32:],
                    _draft_preview(self.tokenizer, draft.token_ids),
                )

        async def build_continuation(draft_id, branch_id):
            draft, draft_elapsed = drafts[draft_id]
            prefix_ids = list(draft.token_ids)
            fallback = bool(fallback_reasons[draft_id])
            eos = self.tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
            terminated = fallback and bool(prefix_ids) and (
                any(token in eos_ids for token in prefix_ids) or len(prefix_ids) >= response_length
            )
            if terminated and branch_id == 0:
                # Keep the actual terminated sample exactly once. Copying it
                # into both leaves would not be two independent samples.
                full_ids = prefix_ids
                logprobs = list(draft.log_probs) if draft.log_probs is not None else None
                elapsed = draft_elapsed
            else:
                if terminated:
                    prefix_ids = []
                # A zero draft advantage boundary is not a zero-length
                # sampled prefix: deduct its actual token length from budget.
                remaining = response_length - len(prefix_ids)
                suffix_params = {**sampling_params, "max_tokens": remaining, "ignore_eos": False}
                suffix_params.pop("stop", None)
                suffix_params.pop("include_stop_str_in_output", None)
                suffix, suffix_elapsed = await generate(
                    prompt_ids + prefix_ids, suffix_params, request_ids[draft_id], draft_id=draft_id, branch_id=branch_id
                )
                if not suffix.token_ids:
                    raise DraftRolloutError("Continuation returned no tokens.")
                # Fresh fallback completions are not revalidated as drafts,
                # even if their sampled text happens to contain draft tags.
                if prefix_ids:
                    if (draft.log_probs is None) != (suffix.log_probs is None):
                        raise DraftRolloutError("Draft and suffix must both provide log probabilities or neither.")
                    logprobs = (
                        list(draft.log_probs) + list(suffix.log_probs) if draft.log_probs is not None else None
                    )
                else:
                    logprobs = list(suffix.log_probs) if suffix.log_probs is not None else None
                full_ids = prefix_ids + list(suffix.token_ids)
                elapsed = draft_elapsed + suffix_elapsed
            return AgentLoopOutput(
                prompt_ids=list(prompt_ids),
                response_ids=full_ids,
                response_mask=[1] * len(full_ids),
                response_logprobs=logprobs,
                multi_modal_data=multi_modal_data,
                num_turns=2,
                metrics={"generate_sequences": elapsed},
                extra_fields={
                    "image_id": image_id,
                    "draft_id": draft_id,
                    "branch_id": branch_id,
                    "draft_token_count": draft_lengths[draft_id],
                    "draft_fallback": fallback,
                    "draft_fallback_reason": fallback_reasons[draft_id],
                },
            )

        async def continue_draft(draft_id, branch_id):
            try:
                return await build_continuation(draft_id, branch_id)
            except BaseException:
                logger.exception(
                    "Draft continuation failed: image_id=%s draft_id=%s branch_id=%s",
                    image_id, draft_id, branch_id,
                )
                raise

        return await _gather_settled(
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
            try:
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
            except BaseException:
                logger.exception(
                    "Draft image rollout failed: image_id=%s global_step=%s; draining other images before returning",
                    kwargs.get("uid"), batch.meta_info.get("global_steps"),
                )
                raise

        outputs_by_image = await _gather_settled(*(run_image(row) for row in range(len(batch))))
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
        worker_refs = []
        workers_settled = False
        primary_error = None
        try:
            # Keep references even if a later submission itself fails.
            for worker, chunk in zip(self.agent_loop_workers, chunks, strict=False):
                worker_refs.append(worker.generate_draft_sequences.remote(chunk))
            outputs = ray.get(worker_refs)
            workers_settled = True
            outputs = self._align_output_prompt_length(outputs)
            output = DataProto.concat(outputs)
            if len(output) != len(prompts):
                raise DraftRolloutError("Draft rollout did not return exactly four trajectories per image.")
            metrics = [chunk.meta_info.pop("metrics") for chunk in outputs]
            timing = self._performance_metrics(metrics, output)
            output.meta_info = {"timing": timing, **outputs[0].meta_info}
            return output
        except BaseException as error:
            primary_error = error
            logger.exception(
                "Draft rollout failed at global_step=%s; draining %s submitted workers before GPU sleep",
                prompts.meta_info.get("global_steps"), len(worker_refs),
            )
            raise
        finally:
            if worker_refs and not workers_settled:
                try:
                    # ray.get(list) can raise while sibling workers still
                    # generate. Failed references are also ready references;
                    # ray.wait does not propagate their task exceptions.
                    _, pending = ray.wait(worker_refs, num_returns=len(worker_refs), fetch_local=False)
                    if pending:
                        raise RuntimeError("Ray returned unfinished workers from an unbounded completion wait.")
                    workers_settled = True
                    for worker_index, reference in enumerate(worker_refs):
                        try:
                            ray.get(reference)
                        except BaseException as error:
                            # A failed task that returned through our worker
                            # drained its requests. A dead/unreachable worker
                            # cannot make that guarantee, even with a ready ref.
                            if (
                                isinstance(error, (RayActorError, WorkerCrashedError))
                                or (isinstance(error, RayError) and not isinstance(error, RayTaskError))
                                or not isinstance(error, Exception)
                            ):
                                workers_settled = False
                            logger.exception("Draft worker %s failed while draining completed workers", worker_index)
                except BaseException:
                    workers_settled = False
                    logger.exception("Unable to confirm that all draft workers completed; GPU sleep will be skipped")
            elif not worker_refs:
                workers_settled = True
            if workers_settled:
                cleanup_error = None
                cleanup_calls = [("rollout", self.sleep)]
                if self.reward_model_manager:
                    cleanup_calls.append(("reward", self.reward_model_manager.sleep))
                for name, cleanup in cleanup_calls:
                    try:
                        cleanup()
                    except BaseException as error:
                        logger.exception("Draft %s cleanup failed after worker completion", name)
                        if cleanup_error is None:
                            cleanup_error = error
                # Cleanup errors remain visible, but must not replace the
                # invalid-draft/server exception that triggered the cleanup.
                if primary_error is None and cleanup_error is not None:
                    raise cleanup_error
            else:
                logger.error("GPU sleep skipped because outstanding draft server requests cannot be ruled out")
