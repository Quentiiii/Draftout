"""Explicit decoding settings and reproducible batch sampling for evaluation."""

import hashlib
import json
import math


SEED_POLICY = "sha256_batch_sample_ids_v1"
PRESENCE_SCOPE = "generated_tokens_only"
SAMPLING_KEYS = {
    "temperature", "top_p", "top_k", "min_p", "presence_penalty",
    "repetition_penalty", "seed", "seed_policy", "seed_batch_size", "presence_penalty_scope",
}


def validate_decoding_config(config: dict) -> None:
    if not isinstance(config.get("do_sample"), bool):
        raise ValueError("do_sample must be a boolean")
    if not config["do_sample"] and not SAMPLING_KEYS.intersection(config):
        return
    missing = SAMPLING_KEYS - config.keys()
    if missing:
        raise ValueError(f"Missing decoding settings: {sorted(missing)}")
    for key in ("temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} must be finite")
    if config["temperature"] <= 0 or not 0 < config["top_p"] <= 1:
        raise ValueError("Require temperature > 0 and 0 < top_p <= 1")
    if not 0 <= config["min_p"] <= 1 or not -2 <= config["presence_penalty"] <= 2:
        raise ValueError("Require 0 <= min_p <= 1 and -2 <= presence_penalty <= 2")
    if config["repetition_penalty"] <= 0:
        raise ValueError("repetition_penalty must be positive")
    for key, lower, upper in (("top_k", 0, 2**63 - 1), ("seed", 0, 2**63 - 1),
                              ("seed_batch_size", 1, 2**63 - 1)):
        if type(config[key]) is not int or not lower <= config[key] <= upper:
            raise ValueError(f"Invalid {key}")
    if config["seed_policy"] != SEED_POLICY or config["presence_penalty_scope"] != PRESENCE_SCOPE:
        raise ValueError("Unsupported sampling seed policy or presence penalty scope")


def build_decoding_config(args) -> dict:
    config = {
        "do_sample": args.do_sample,
        **{key: getattr(args, key) for key in (
            "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "seed",
        )},
        "seed_policy": SEED_POLICY,
        "seed_batch_size": args.batch_size,
        "presence_penalty_scope": PRESENCE_SCOPE,
    }
    validate_decoding_config(config)
    # Preserve legacy greedy configuration exactly, including resume compatibility.
    if not args.do_sample and args.presence_penalty == 0 and args.repetition_penalty == 1:
        return {"do_sample": False}
    return config


def batch_seed(seed: int, samples) -> int:
    payload = json.dumps([seed, [[sample.dataset, sample.sample_id] for sample in samples]],
                         ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)


class PresencePenaltyLogitsProcessor:
    """Apply an additive penalty once per token seen in generated text only.

    Transformers 5.5 does not implement presence_penalty. Its GenerationConfig
    accepts unknown attributes, so passing the setting there would silently do
    nothing. This callable follows the generated-token semantics used by vLLM.
    """

    def __init__(self, penalty: float, prompt_length: int):
        if not math.isfinite(penalty) or not -2 <= penalty <= 2:
            raise ValueError("presence penalty must be finite and within [-2, 2]")
        if prompt_length < 0:
            raise ValueError("prompt_length must be nonnegative")
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        if self.penalty == 0 or input_ids.shape[1] <= self.prompt_length:
            return scores
        import torch

        seen = torch.zeros_like(scores, dtype=torch.bool)
        seen.scatter_(1, input_ids[:, self.prompt_length:], True)
        return scores - seen.to(scores.dtype) * self.penalty
