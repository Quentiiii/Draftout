"""Strict label parsing and dependency-free classification metrics."""

import re
from collections.abc import Sequence

from .data import canonical_label


def parse_prediction(text: str, labels: Sequence[str]) -> str | None:
    """Accept one label ignoring case; return the configured canonical spelling."""
    text = text.strip().lower()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1].strip()
    answer = re.fullmatch(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL)
    if answer:
        text = answer.group(1)
    text = text.strip(" \t\r\n\"'`.*")
    return canonical_label(text, labels)


def classification_metrics(records: Sequence[dict], labels: Sequence[str]) -> dict:
    """Invalid answers remain in the denominator and count as false negatives."""
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    # Last column is invalid; rows are ground truth, columns are predictions.
    matrix = [[0] * (len(labels) + 1) for _ in labels]
    for record in records:
        truth_label = canonical_label(record["label"], labels)
        if truth_label is None:
            raise ValueError(f"Unknown ground-truth emotion: {record['label']!r}")
        truth = index[truth_label]
        prediction = index.get(canonical_label(record["prediction"], labels), len(labels))
        matrix[truth][prediction] += 1
    per_class = {}
    for label, i in index.items():
        tp = matrix[i][i]
        support = sum(matrix[i])
        predicted = sum(row[i] for row in matrix)
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    count = len(records)
    correct = sum(matrix[i][i] for i in range(len(labels)))
    return {
        "num_samples": count,
        "num_correct": correct,
        "num_invalid": sum(row[-1] for row in matrix),
        "accuracy": correct / count if count else 0.0,
        "macro_f1": sum(row["f1"] for row in per_class.values()) / len(labels),
        "per_class": per_class,
        "confusion_matrix": {"rows": labels, "columns": [*labels, "invalid"], "values": matrix},
    }
