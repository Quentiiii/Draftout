"""Binary EmoSet correctness reward for verl's custom reward interface."""

import re
from collections.abc import Sequence

from emotion_eval.data import LABELS, canonical_label
from emotion_eval.metrics import parse_angles_prediction


def _parse_single_boxed_label(solution_str, labels: Sequence[str]) -> str | None:
    """Accept exactly one boxed canonical label, regardless of reasoning format."""
    if not isinstance(solution_str, str):
        return None
    # Count even malformed boxes so a second, broken answer is not ignored.
    if len(re.findall(r"\\boxed\b", solution_str)) != 1:
        return None
    match = re.search(r"\\boxed\s*\{([^{}]*)\}", solution_str)
    return canonical_label(match.group(1), labels) if match is not None else None


def compute_grpo_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Reward ordinary GRPO by boxed accuracy without requiring draft tags.

    Reasoning may use any format. Only one unambiguous, canonical boxed answer
    determines correctness; absent, malformed, multiple, or wrong answers earn
    zero, using the same task-accuracy parser as the Draft stage.
    """
    labels = LABELS["emoset"]
    truth = canonical_label(ground_truth, labels)
    if truth is None:
        raise ValueError(f"Unknown ground-truth EmoSet emotion: {ground_truth!r}")
    return float(_parse_single_boxed_label(solution_str, labels) == truth)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Return 1 for the correct emotion, otherwise 0; never add a format bonus.

    Require a nonempty draft plan, separate analysis, and one final boxed label,
    using the same strict parser as angles evaluation. Emotions mentioned in
    the text do not determine the answer. No angle-count or explanation bonus
    is applied; malformed outputs and wrong labels earn zero.
    Ground truth must itself be one of the eight EmoSet labels.
    """
    labels = LABELS["emoset"]
    truth = canonical_label(ground_truth, labels)
    if truth is None:
        raise ValueError(f"Unknown ground-truth EmoSet emotion: {ground_truth!r}")
    if not isinstance(solution_str, str):
        return 0.0
    return float(parse_angles_prediction(solution_str, labels) == truth)


def compute_draft_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> dict[str, float]:
    """Use strict-format correctness during warmup, otherwise boxed accuracy.

    Task accuracy ignores the draft and analysis format, but requires exactly
    one unambiguous boxed label. Its separate ``task_reward`` field is always
    binary classification correctness, including during format warmup.
    """
    labels = LABELS["emoset"]
    truth = canonical_label(ground_truth, labels)
    if truth is None:
        raise ValueError(f"Unknown ground-truth EmoSet emotion: {ground_truth!r}")
    prediction = _parse_single_boxed_label(solution_str, labels)
    format_valid = False
    if isinstance(solution_str, str):
        format_valid = parse_angles_prediction(solution_str, labels) is not None
    task_reward = float(prediction == truth)
    warmup = bool((extra_info or {}).get("draft_grpo_warmup", False))
    score = (
        compute_score(data_source, solution_str, truth, extra_info=extra_info, **kwargs)
        if warmup
        else task_reward
    )
    return {"score": score, "task_reward": task_reward, "format_valid": float(format_valid)}
