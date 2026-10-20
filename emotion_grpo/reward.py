"""Binary EmoSet correctness reward for verl's custom reward interface."""

from emotion_eval.data import LABELS, canonical_label
from emotion_eval.metrics import parse_prediction


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    """Return 1 for the correct emotion, otherwise 0; never add a format bonus.

    Use the evaluation parser so a single case-insensitive label, an answer
    wrapper, or a final answer after a thinking block are scored consistently.
    Alternatives, explanations, unknown labels, and empty answers earn zero.
    Ground truth must itself be one of the eight EmoSet labels.
    """
    labels = LABELS["emoset"]
    truth = canonical_label(ground_truth, labels)
    if truth is None:
        raise ValueError(f"Unknown ground-truth EmoSet emotion: {ground_truth!r}")
    if not isinstance(solution_str, str):
        return 0.0
    return float(parse_prediction(solution_str, labels) == truth)
