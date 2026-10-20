"""Prepare reproducible EmoSet subsets in verl's multimodal parquet format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from emotion_eval.data import IMAGE_SUFFIXES, LABELS, canonical_label
from emotion_eval.prompts import EMOTION_PROMPT

EMOSET_LABELS = LABELS["emoset"]
PROMPT = "<image>\n" + EMOTION_PROMPT.format(labels=", ".join(EMOSET_LABELS))
DEFAULT_DATASET_ROOT = Path("/data/xiyan.zhao/work/dataset/EMOSET")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "emoset_grpo" / "data"
ARTIFACT_NAMES = (
    "train.parquet", "val.parquet", "train_manifest.jsonl", "val_manifest.jsonl", "metadata.json",
)


@dataclass(frozen=True)
class SplitEntry:
    label: str
    image: str
    annotation: str
    source_index: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{context}: expected a relative POSIX path, got {value!r}")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != value:
        raise ValueError(f"{context}: invalid relative path {value!r}")
    return value


def read_split(root: Path, split: str) -> list[SplitEntry]:
    """Read an official split without inferring answers from image filenames."""
    path = root / f"{split}.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a nonempty list")
    entries: list[SplitEntry] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        context = f"{path}, entry {index}"
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError(f"{context}: expected [label, image_path, annotation_path]")
        label = canonical_label(row[0], EMOSET_LABELS)
        if label is None:
            raise ValueError(f"{context}: unknown emotion {row[0]!r}")
        image = _relative_path(row[1], context)
        annotation = _relative_path(row[2], context)
        if PurePosixPath(image).suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"{context}: unsupported image extension: {image!r}")
        if image in seen:
            raise ValueError(f"{context}: duplicate image {image!r}")
        seen.add(image)
        entries.append(SplitEntry(label, image, annotation, index))
    return entries


def check_disjoint(splits: dict[str, list[SplitEntry]]) -> None:
    """Check every official split, including test, before drawing subsets."""
    seen: dict[str, str] = {}
    for split, entries in splits.items():
        for entry in entries:
            if entry.image in seen:
                raise ValueError(f"Split overlap: {entry.image!r} is in {seen[entry.image]} and {split}")
            seen[entry.image] = split


def select_entries(
    entries: list[SplitEntry], count: int, seed: int, sampling: str = "balanced",
) -> list[SplitEntry]:
    """Sample without replacement; stable even if the manifest is reordered."""
    if count <= 0:
        raise ValueError("Subset sizes must be positive")
    if sampling not in ("balanced", "random"):
        raise ValueError(f"Unknown sampling method: {sampling!r}")
    rng = random.Random(seed)
    population = sorted(entries, key=lambda entry: entry.image)
    if sampling == "random":
        if count > len(population):
            raise ValueError(f"Requested {count} examples, but only {len(population)} are available")
        selected = rng.sample(population, count)
    else:
        if count % len(EMOSET_LABELS):
            raise ValueError(f"Balanced subset size must be divisible by {len(EMOSET_LABELS)}")
        per_class = count // len(EMOSET_LABELS)
        selected = []
        for label in EMOSET_LABELS:
            candidates = [entry for entry in population if entry.label == label]
            if len(candidates) < per_class:
                raise ValueError(f"{label}: requested {per_class} examples, only {len(candidates)} available")
            selected.extend(rng.sample(candidates, per_class))
    rng.shuffle(selected)
    return selected


def _existing_file(root: Path, relative: str) -> Path:
    path = root.joinpath(*PurePosixPath(relative).parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {relative!r}")
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return path


def make_rows(
    root: Path, split: str, entries: list[SplitEntry], min_pixels: int = 65536, max_pixels: int = 262144,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate selected annotations and keep per-example answers out of prompts."""
    rows, manifest = [], []
    seen_paths: set[Path] = set()
    for index, entry in enumerate(entries):
        image_path = _existing_file(root, entry.image)
        annotation_path = _existing_file(root, entry.annotation)
        if image_path in seen_paths:
            raise ValueError(f"Duplicate resolved image: {image_path}")
        seen_paths.add(image_path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if not isinstance(annotation, dict):
            raise ValueError(f"{annotation_path}: expected an annotation object")
        label = canonical_label(annotation.get("emotion"), EMOSET_LABELS)
        if label is None:
            raise ValueError(f"{annotation_path}: unknown or missing emotion")
        if label != entry.label:
            raise ValueError(
                f"{annotation_path}: label mismatch: {split}.json={entry.label!r}, annotation={label!r}"
            )
        # This opaque ID is metadata only. Image paths are consumed as image
        # inputs by verl; neither paths nor the answer enter message text.
        sample_id = hashlib.sha256(entry.image.encode("utf-8")).hexdigest()
        rows.append({
            "data_source": "emoset",
            "prompt": [{"role": "user", "content": PROMPT}],
            "images": [{"image": str(image_path), "min_pixels": min_pixels, "max_pixels": max_pixels}],
            "ability": "emotion_classification",
            "reward_model": {"style": "rule", "ground_truth": label},
            "extra_info": {"split": split, "index": index, "sample_id": sample_id},
        })
        manifest.append({
            "index": index,
            "sample_id": sample_id,
            "source_split": split,
            "source_index": entry.source_index,
            "image": entry.image,
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
            "annotation": entry.annotation,
            "annotation_sha256": sha256_file(annotation_path),
            "ground_truth": label,
        })
    return rows, manifest


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_dataset(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_size: int = 2000,
    val_size: int = 256,
    seed: int = 42,
    sampling: str = "balanced",
    min_pixels: int = 65536,
    max_pixels: int = 262144,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build all artifacts, refusing to silently replace a different dataset."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(dataset_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if min_pixels <= 0 or max_pixels < min_pixels:
        raise ValueError("Pixel limits must satisfy 0 < min_pixels <= max_pixels")
    if output_dir == root or root.is_relative_to(output_dir):
        raise ValueError("Output directory must not be the dataset root or one of its ancestors")
    splits = {split: read_split(root, split) for split in ("train", "val", "test")}
    check_disjoint(splits)
    config = {
        "dataset_root": str(root),
        "train_size": train_size,
        "val_size": val_size,
        "seed": seed,
        "sampling": sampling,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "emoset",
        "config": config,
        "labels": list(EMOSET_LABELS),
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "selection": "Sort by relative image path, sample without replacement, then shuffle with random.Random(seed).",
        "sources": {
            split: {
                "path": str(root / f"{split}.json"),
                "sha256": sha256_file(root / f"{split}.json"),
                "count": len(entries),
                "class_counts": dict(sorted(Counter(entry.label for entry in entries).items())),
            }
            for split, entries in splits.items()
        },
        "checks": {
            "official_train_val_test_paths_disjoint": True,
            "selected_annotation_labels_match_split": True,
            "selected_files_exist": True,
            "selected_resolved_paths_disjoint": True,
            "selected_train_val_image_contents_disjoint": True,
        },
        "subsets": {},
        "artifacts": {},
    }
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".emoset-grpo-", dir=output_dir.parent) as temporary:
        stage = Path(temporary)
        manifests = {}
        for split, count in (("train", train_size), ("val", val_size)):
            selected = select_entries(splits[split], count, seed, sampling)
            rows, manifest = make_rows(root, split, selected, min_pixels, max_pixels)
            manifests[split] = manifest
            pq.write_table(pa.Table.from_pylist(rows), stage / f"{split}.parquet", compression="snappy")
            (stage / f"{split}_manifest.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
                encoding="utf-8",
            )
            metadata["subsets"][split] = {
                "count": len(rows),
                "class_counts": dict(sorted(Counter(row["ground_truth"] for row in manifest).items())),
            }
        train_paths = {row["image_path"] for row in manifests["train"]}
        val_paths = {row["image_path"] for row in manifests["val"]}
        if train_paths & val_paths:
            raise ValueError("Selected train and val subsets share resolved image paths")
        train_hashes = {row["image_sha256"] for row in manifests["train"]}
        val_hashes = {row["image_sha256"] for row in manifests["val"]}
        if train_hashes & val_hashes:
            raise ValueError("Selected train and val subsets share identical image contents")
        for name in ARTIFACT_NAMES[:-1]:
            metadata["artifacts"][name] = {"sha256": sha256_file(stage / name)}
        _write_json(stage / "metadata.json", metadata)
        existing = [name for name in ARTIFACT_NAMES if (output_dir / name).exists()]
        identical = len(existing) == len(ARTIFACT_NAMES) and all(
            sha256_file(output_dir / name) == sha256_file(stage / name) for name in ARTIFACT_NAMES
        )
        if existing and not identical and not overwrite:
            raise FileExistsError(
                f"{output_dir} already contains different or incomplete GRPO artifacts; "
                "choose a new --output-dir or pass --overwrite"
            )
        if not identical:
            output_dir.mkdir(parents=True, exist_ok=True)
            for name in ARTIFACT_NAMES:
                os.replace(stage / name, output_dir / name)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=2000)
    parser.add_argument("--val-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sampling", choices=("balanced", "random"), default="balanced")
    parser.add_argument("--min-pixels", type=int, default=65536)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--overwrite", action="store_true", help="Replace existing, different GRPO artifacts.")
    args = parser.parse_args()
    try:
        metadata = prepare_dataset(**vars(args))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "output_dir": str(args.output_dir.expanduser().resolve()),
        "seed": metadata["config"]["seed"],
        "sampling": metadata["config"]["sampling"],
        "subsets": metadata["subsets"],
    }, indent=2))


if __name__ == "__main__":
    main()
