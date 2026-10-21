"""Load evaluation images without exposing their labels to the model."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


LABELS = {
    "emoset": (
        "Amusement", "Awe", "Contentment", "Excitement",
        "Anger", "Disgust", "Fear", "Sadness",
    ),
    "emotion6": ("Anger", "Disgust", "Fear", "Joy", "Sadness", "Surprise"),
    "emo8": (
        "Amusement", "Awe", "Contentment", "Excitement",
        "Anger", "Disgust", "Fear", "Sadness",
    ),
}

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"})


def canonical_label(value: Any, labels: Sequence[str]) -> str | None:
    """Match case-insensitively, returning the spelling configured in LABELS."""
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    for label in labels:
        if label.casefold() == normalized:
            return label
    return None


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_id: str
    image_path: Path
    label: str


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _relative_file(root: Path, value: Any, context: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context}: expected a nonempty relative POSIX path, got {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or value != relative.as_posix():
        raise ValueError(f"{context}: invalid relative path {value!r}")
    path = root.joinpath(*relative.parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"{context}: path escapes dataset root: {value!r}")
    if not path.is_file():
        raise FileNotFoundError(f"{context}: missing file {path}")
    return relative.as_posix(), path


def load_emoset(root: Path, test_list: Path | None = None) -> list[Sample]:
    """Read test.json (or an explicit list) and use annotation emotions as answers.

    The redundant emotion in test.json must match the annotation. Filename and
    directory names never determine the answer for EmoSet. Callers providing
    a subset list are responsible for checking its official split membership.
    """
    root = Path(root).expanduser().resolve()
    split_path = Path(test_list).expanduser().resolve() if test_list is not None else root / "test.json"
    rows = _read_json(split_path)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{split_path}: expected a nonempty list of test samples")
    samples: list[Sample] = []
    seen: set[Path] = set()
    for index, row in enumerate(rows, start=1):
        context = f"{split_path}, entry {index}"
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{context}: expected [label, image_path, annotation_path]")
        split_value, image_value, annotation_value = row
        split_label = canonical_label(split_value, LABELS["emoset"])
        if split_label is None:
            raise ValueError(f"{context}: unknown emotion {split_value!r}")
        sample_id, image_path = _relative_file(root, image_value, context)
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{context}: unsupported image extension: {image_value!r}")
        if image_path in seen:
            raise ValueError(f"{context}: duplicate image {sample_id!r}")
        _, annotation_path = _relative_file(root, annotation_value, context)
        annotation = _read_json(annotation_path)
        if not isinstance(annotation, dict):
            raise ValueError(f"{annotation_path}: expected a JSON object")
        annotation_value = annotation.get("emotion")
        label = canonical_label(annotation_value, LABELS["emoset"])
        if label is None:
            raise ValueError(f"{annotation_path}: unknown or missing emotion {annotation_value!r}")
        if label != split_label:
            raise ValueError(
                f"{context}: label mismatch: test.json={split_label!r}, "
                f"annotation emotion={label!r} ({annotation_path})"
            )
        seen.add(image_path)
        samples.append(Sample("emoset", sample_id, image_path, label))
    return samples


def _emotion6_image_root(root: Path) -> Path:
    root = Path(root).expanduser().resolve()
    candidates = [root / "images", root / "Emotion6" / "images"]
    if root.name == "images":
        candidates.insert(0, root)
    existing = [path for path in candidates if path.is_dir()]
    if not existing:
        raise FileNotFoundError(
            f"Emotion6 image directory not found under {root}; "
            "expected images/ or Emotion6/images/"
        )
    if len(existing) > 1:
        raise ValueError(f"Ambiguous Emotion6 image directories: {existing}")
    return existing[0].resolve()


def load_emotion6(root: Path, test_list: Path | None = None) -> list[Sample]:
    """Use the six image parent-directory names as Emotion6 answers.

    The local dataset has no test split. By default every image is evaluated.
    A test_list can select an existing split using one path (e.g. anger/1.jpg)
    per line, relative to images/. ground_truth.txt is intentionally not read.
    """
    image_root = _emotion6_image_root(root)
    if test_list is not None:
        test_list = Path(test_list).expanduser().resolve()
        entries = [
            (f"{test_list}, line {number}", line.strip())
            for number, line in enumerate(test_list.read_text(encoding="utf-8-sig").splitlines(), start=1)
            if line.strip()
        ]
    else:
        entries = []
        for class_dir in sorted(image_root.iterdir()):
            if not class_dir.is_dir():
                raise ValueError(f"Expected an Emotion6 class directory, found {class_dir}")
            if canonical_label(class_dir.name, LABELS["emotion6"]) is None:
                raise ValueError(f"Unknown Emotion6 class directory: {class_dir}")
            for image_path in sorted(class_dir.iterdir()):
                entries.append((str(image_root), image_path.relative_to(image_root).as_posix()))
    if not entries:
        raise ValueError(f"No Emotion6 evaluation images found for {image_root}")
    samples: list[Sample] = []
    seen: set[Path] = set()
    for context, value in entries:
        relative = PurePosixPath(value)
        if len(relative.parts) != 2:
            raise ValueError(f"{context}: expected <emotion>/<image>, got {value!r}")
        label = canonical_label(relative.parts[0], LABELS["emotion6"])
        if label is None:
            raise ValueError(f"{context}: expected <emotion>/<image>, got {value!r}")
        sample_id, image_path = _relative_file(image_root, value, context)
        if image_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{context}: unsupported image extension: {value!r}")
        if image_path in seen:
            raise ValueError(f"{context}: duplicate image {sample_id!r}")
        seen.add(image_path)
        samples.append(Sample("emotion6", sample_id, image_path, label))
    return samples


def load_emo8(root: Path) -> list[Sample]:
    """Evaluate every image from EMO8's train/, val/, and test/ directories.

    Answers come from class directory names, with EMO8's ``sad`` mapped to
    ``Sadness``. IDs retain the original split/class/filename so identical
    filenames in different splits remain distinct samples.
    """
    root = Path(root).expanduser().resolve()
    split_directories = [root / split for split in ("train", "val", "test")]
    for split_dir in split_directories:
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing EMO8 split directory: {split_dir}")
        if not split_dir.resolve().is_relative_to(root):
            raise ValueError(f"EMO8 split directory escapes dataset root: {split_dir}")

    samples: list[Sample] = []
    seen: set[Path] = set()
    for split_dir in split_directories:
        split_start = len(samples)
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                raise ValueError(f"Expected an EMO8 class directory, found {class_dir}")
            class_name = "Sadness" if class_dir.name.strip().casefold() == "sad" else class_dir.name
            label = canonical_label(class_name, LABELS["emo8"])
            if label is None:
                raise ValueError(f"Unknown EMO8 class directory: {class_dir}")
            if not class_dir.resolve().is_relative_to(root):
                raise ValueError(f"EMO8 class directory escapes dataset root: {class_dir}")
            for candidate in sorted(class_dir.iterdir()):
                value = candidate.relative_to(root).as_posix()
                sample_id, image_path = _relative_file(root, value, str(split_dir))
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    raise ValueError(f"{split_dir}: unsupported image extension: {value!r}")
                if image_path in seen:
                    raise ValueError(f"{split_dir}: duplicate image {sample_id!r}")
                seen.add(image_path)
                samples.append(Sample("emo8", sample_id, image_path, label))
        if len(samples) == split_start:
            raise ValueError(f"No EMO8 evaluation images found for split {split_dir}")
    return samples
