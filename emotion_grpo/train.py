"""Local verl entry point with a CPU preflight for the EmoSet GRPO recipe."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def check_config(config: DictConfig) -> None:
    """Validate the actual verl configs before creating any Ray/GPU workers."""
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import omega_conf_to_dataclass, validate_config

    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    if config.algorithm.adv_estimator != "grpo" or actor.policy_loss.loss_mode != "vanilla":
        raise ValueError("This entry point requires GRPO advantages and vanilla clipped policy loss.")
    if rollout.n < 2 or not rollout.do_sample or rollout.temperature <= 0:
        raise ValueError("GRPO requires at least two sampled answers per image and temperature > 0.")
    if need_reference_policy(config) or need_critic(config) or config.reward_model.enable:
        raise ValueError("This recipe uses correctness only, with KL, critic and reward model disabled.")
    if actor.entropy_coeff != 0:
        raise ValueError("This recipe uses no entropy bonus.")
    # Qwen3.5 needs separate batch rows for independent recurrent states.
    # Single-sample unpadding remains an option for smaller-memory devices.
    if (
        actor.use_dynamic_bsz
        or rollout.log_prob_use_dynamic_bsz
        or actor.ulysses_sequence_parallel_size != 1
        or actor.fsdp_config.ulysses_sequence_parallel_size != 1
    ):
        raise ValueError("This Qwen3.5 recipe requires fixed micro batches and SP=1.")
    actor_micro_batch = actor.ppo_micro_batch_size_per_gpu
    log_prob_micro_batch = rollout.log_prob_micro_batch_size_per_gpu
    if not actor_micro_batch or not log_prob_micro_batch or min(actor_micro_batch, log_prob_micro_batch) < 1:
        raise ValueError("Set positive per-GPU actor and log-prob micro batch sizes.")
    if config.actor_rollout_ref.model.use_remove_padding:
        if actor_micro_batch != 1 or log_prob_micro_batch != 1:
            raise ValueError("Qwen3.5 remove_padding=true requires micro batches=1 to avoid cross-sample packing.")
    elif min(actor_micro_batch, log_prob_micro_batch) < 2:
        # Transformers 5.5 only applies the recurrent padding mask for B > 1.
        raise ValueError("Padded Qwen3.5 forwards require micro batches >= 2; for batch 1 enable remove_padding.")
    if rollout.max_model_len < config.data.max_prompt_length + config.data.max_response_length:
        raise ValueError("rollout.max_model_len must cover max_prompt_length + max_response_length.")
    if config.trainer.n_gpus_per_node < 1:
        raise ValueError("At least one GPU must be configured, including when checking the recipe on CPU.")
    if config.trainer.n_gpus_per_node % rollout.tensor_model_parallel_size:
        raise ValueError("The GPU count must be divisible by rollout tensor parallel size.")
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    for name, size, micro_batch in (
        ("actor mini batch", actor.ppo_mini_batch_size * rollout.n, actor_micro_batch),
        ("log-prob batch", config.data.train_batch_size * rollout.n, log_prob_micro_batch),
    ):
        if size % (world_size * micro_batch):
            raise ValueError(
                f"{name} ({size} answers) must be divisible by GPU count ({world_size}) "
                f"* per-GPU micro batch ({micro_batch}); adjust the batch size or GPU count."
            )
    validate_config(config, use_reference_policy=False, use_critic=False)
    omega_conf_to_dataclass(rollout)


def check_data(config: DictConfig) -> dict:
    """Exercise real parquet loading, image processing, templates and rewards."""
    from emotion_eval.data import LABELS
    from verl.trainer.ppo.reward import get_custom_reward_fn
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import RLHFDataset

    model_path = Path(config.actor_rollout_ref.model.path).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Local model directory is missing config.json: {model_path}")
    model_config = json.loads((model_path / "config.json").read_text())
    if model_config.get("model_type") != "qwen3_5":
        raise ValueError("This recipe is configured for the local Qwen3.5 multimodal models.")
    tokenizer = hf_tokenizer(str(model_path), trust_remote_code=False)
    processor = hf_processor(str(model_path), trust_remote_code=False, use_fast=True)
    if processor is None or processor.image_processor.patch_size != config.data.image_patch_size:
        raise ValueError("The dataset image_patch_size must match the model's image processor.")
    reward_fn = get_custom_reward_fn(config)
    if reward_fn is None:
        raise ValueError("A correctness reward function must be configured.")

    result = {"model": str(model_path), "splits": {}}
    split_ids = {}
    for split, paths in (("train", config.data.train_files), ("val", config.data.val_files)):
        dataset = RLHFDataset(paths, tokenizer, config.data, processor)
        if split == "train" and len(dataset) != 2000:
            raise ValueError(f"Expected exactly 2,000 training examples; found {len(dataset)}.")
        if not len(dataset):
            raise ValueError(f"{split} is empty.")
        if split == "train" and len(dataset) % config.data.train_batch_size:
            raise ValueError("train_batch_size must divide 2,000 so verl's drop_last skips no training images.")
        seen_labels = set()
        ids = set()
        prompt_lengths = []
        # Read every row; process one real image for every class in each split.
        for index in range(len(dataset)):
            row = dataset[index]
            label = row["reward_model"]["ground_truth"]
            if label not in LABELS["emoset"]:
                raise ValueError(f"Unknown ground truth in {split}[{index}]: {label}")
            messages = row["raw_prompt"]
            images = [part for message in messages for part in message["content"] if part["type"] == "image"]
            if len(images) != 1:
                raise ValueError(f"{split}[{index}] must contain exactly one image.")
            image_path = Path(images[0]["image"]).resolve()
            if not image_path.is_file() or image_path in ids:
                raise ValueError(f"Missing or duplicate image in {split}: {image_path}")
            ids.add(image_path)
            if label in seen_labels:
                continue
            seen_labels.add(label)
            pil_images, videos = asyncio.run(
                dataset.process_vision_info(messages, image_patch_size=config.data.image_patch_size, config=config.data)
            )
            if videos:
                raise ValueError("EmoSet must contain images only.")
            rendered = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **config.data.apply_chat_template_kwargs
            )
            if str(image_path) in rendered or image_path.name in rendered:
                raise ValueError("Image paths must not be exposed in the model's text prompt.")
            inputs = processor(text=[rendered], images=pil_images, return_tensors="pt")
            length = inputs.input_ids.shape[-1]
            if length > config.data.max_prompt_length:
                raise ValueError(f"{split}[{index}] prompt length {length} exceeds max_prompt_length.")
            rope_kwargs = {"image_grid_thw": inputs.image_grid_thw}
            if "mm_token_type_ids" in inputs:
                rope_kwargs["mm_token_type_ids"] = inputs.mm_token_type_ids
            processor.get_rope_index(
                input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, **rope_kwargs
            )
            prompt_lengths.append(length)
            for answer, expected in ((label, 1.0), ("unknown", 0.0)):
                score = reward_fn(
                    data_source=row["data_source"], solution_str=answer, ground_truth=label, extra_info={}
                )
                if isinstance(score, dict):
                    score = score["score"]
                if score != expected:
                    raise ValueError("Correctness reward contract failed.")
        split_ids[split] = ids
        result["splits"][split] = {
            "count": len(dataset),
            "processed_images": len(prompt_lengths),
            "max_checked_prompt_tokens": max(prompt_lengths),
        }
    if split_ids["train"] & split_ids["val"]:
        raise ValueError("Training and validation image paths overlap.")
    return result


@hydra.main(config_path="../verl/trainer/config", config_name="emoset_grpo", version_base=None)
def main(config: DictConfig) -> None:
    import torch

    check_config(config)
    if not config.dry_run:
        expected = config.trainer.n_gpus_per_node
        if not torch.cuda.is_available() or torch.cuda.device_count() < expected:
            raise RuntimeError(
                f"Training needs {expected} visible CUDA GPUs; found {torch.cuda.device_count()}. "
                "Use scripts/run_emoset_grpo.sh --check for CPU validation."
            )
    result = check_data(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if config.dry_run:
        print("CPU preflight passed. Model weights, GPU rollout and distributed training were not exercised.")
        return
    checkpoint_dir = Path(config.trainer.default_local_dir)
    if config.trainer.resume_mode == "disable" and any(checkpoint_dir.glob("global_step_*")):
        raise FileExistsError(
            f"Checkpoints already exist in {checkpoint_dir}; set trainer.resume_mode=auto to resume "
            "or choose a new trainer.experiment_name."
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, checkpoint_dir / "resolved_config.yaml", resolve=True)
    from verl.trainer.main_ppo import run_ppo

    run_ppo(config)


if __name__ == "__main__":
    main()
