#!/usr/bin/env python3
"""Validate complete emotion-evaluation shards and restore official sample order."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emotion_eval.data import LABELS, Sample, load_emo8, load_emotion6, load_emoset
from emotion_eval.infer import DEFAULT_DATA, data_fingerprint, read_predictions
from emotion_eval.metrics import (
    classification_metrics, parse_angles_prediction, parse_angles_response,
    parse_draft_prediction, parse_draft_response,
)
from emotion_eval.prompts import DEEPSEEK_EMOTION_PROMPT, EMOTION_ANALYSIS_ANGLES_PROMPT
from emotion_eval.sampling import SAMPLING_KEYS, validate_decoding_config


SPLITS = {
    "emoset": "official_test",
    "emotion6": "all_images_no_official_test_split",
    "emo8": "all_train_val_test_as_test",
}
VARIABLE_CONFIG_KEYS = {"num_samples", "samples_sha256", "device", "shard_index"}
REQUIRED_CONFIG_KEYS = {
    "format_version", "dataset", "split", "num_samples", "samples_sha256", "model", "prompt",
    "labels", "enable_thinking", "response_format", "do_sample", "max_new_tokens", "min_pixels",
    "max_pixels", "dtype", "device", "batch_size", "num_shards", "shard_index",
}


def _read_predictions_snapshot(
    path: Path, samples: list[Sample], labels: tuple[str, ...], *, angles: bool = False,
) -> tuple[list[dict], str]:
    """Reuse resume validation on a copy, without repairing or writing source shards."""
    payload = path.read_bytes()
    for number, line in enumerate(payload.splitlines(), 1):
        try:
            value = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid or incomplete JSONL in {path}, line {number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Expected a JSON object in {path}, line {number}")
    with tempfile.TemporaryDirectory(prefix="emotion-shard-read-") as temporary:
        snapshot = Path(temporary) / "predictions.jsonl"
        snapshot.write_bytes(payload)
        records = read_predictions(
            snapshot, samples, labels,
            prediction_parser=parse_angles_prediction if angles else parse_draft_prediction,
        )
    for record in records:
        if angles:
            draft, analysis, _ = parse_angles_response(record["raw_output"], labels)
            if "analysis" not in record or record["analysis"] != analysis:
                raise ValueError(f"Saved analysis disagrees with raw output in {path}: {record['sample_id']}")
        else:
            draft, _ = parse_draft_response(record["raw_output"], labels)
        if "draft" not in record or record["draft"] != draft:
            raise ValueError(f"Saved explanation disagrees with raw output in {path}: {record['sample_id']}")
    return records, hashlib.sha256(payload).hexdigest()


def aggregate(input_dir: Path, data_root: Path = DEFAULT_DATA, num_shards: int = 2) -> dict:
    """Validate every dataset before publishing complete aggregate artifacts."""
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    input_dir = Path(input_dir).expanduser().resolve()
    data_root = Path(data_root).expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")
    datasets = {
        "emoset": load_emoset(data_root / "EMOSET"),
        "emotion6": load_emotion6(data_root / "Emotion6"),
        "emo8": load_emo8(data_root / "EMO8"),
    }
    validated = {}
    summary = {}
    for dataset, samples in datasets.items():
        if not samples or len({sample.sample_id for sample in samples}) != len(samples):
            raise ValueError(f"Empty or duplicate official samples for {dataset}")
        records_by_id = {}
        shard_configs = []
        provenance = []
        reference_core = None
        for index in range(num_shards):
            expected = samples[index::num_shards]
            directory = input_dir / f"shard_{index}" / dataset
            config_path = directory / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or REQUIRED_CONFIG_KEYS - config.keys():
                raise ValueError(f"Missing required configuration in {config_path}")
            version = config.get("format_version")
            if version == 2:
                if config.get("do_sample") is not False or SAMPLING_KEYS.intersection(config):
                    raise ValueError(f"Invalid legacy greedy configuration in {config_path}")
            elif version in (3, 4):
                if version == 3 and SAMPLING_KEYS - config.keys():
                    raise ValueError(f"Missing sampling configuration in {config_path}")
                validate_decoding_config(config)
                if "seed_batch_size" in config and config["seed_batch_size"] != config["batch_size"]:
                    raise ValueError(f"Sampling batch size mismatch in {config_path}")
            else:
                raise ValueError(f"Unsupported format version in {config_path}: {version}")
            angles = version == 4
            template = EMOTION_ANALYSIS_ANGLES_PROMPT if angles else DEEPSEEK_EMOTION_PROMPT
            if not angles and config.get("prompt_style", "draft") != "draft":
                raise ValueError(f"Invalid legacy prompt style configuration in {config_path}")
            expected_fields = {
                "dataset": dataset,
                "split": SPLITS[dataset],
                "num_samples": len(expected),
                "samples_sha256": data_fingerprint(expected),
                "prompt": template.format(labels=", ".join(LABELS[dataset])),
                "labels": list(LABELS[dataset]),
                "enable_thinking": False,
                "response_format": "angles_then_analysis_then_boxed_label" if angles else "draft_then_boxed_label",
                "num_shards": num_shards,
                "shard_index": index,
            }
            if angles:
                expected_fields["prompt_style"] = "angles"
            differences = [key for key, value in expected_fields.items() if config.get(key) != value]
            if differences:
                raise ValueError(f"Unexpected shard configuration in {config_path}: {differences}")
            core = {key: value for key, value in config.items() if key not in VARIABLE_CONFIG_KEYS}
            if reference_core is None:
                reference_core = core
            elif core != reference_core:
                differences = sorted(key for key in core.keys() | reference_core.keys()
                                     if core.get(key) != reference_core.get(key))
                raise ValueError(f"Incompatible shard settings in {config_path}: {differences}")
            records, digest = _read_predictions_snapshot(
                directory / "predictions.jsonl", expected, LABELS[dataset], angles=angles,
            )
            if len(records) != len(expected):
                raise ValueError(f"Incomplete shard {index} for {dataset}: {len(records)}/{len(expected)} records")
            for record in records:
                if record["sample_id"] in records_by_id:
                    raise ValueError(f"Duplicate sample across shards: {record['sample_id']}")
                records_by_id[record["sample_id"]] = record
            shard_configs.append(config)
            provenance.append({
                "shard_index": index,
                "directory": str(directory),
                "predictions_sha256": digest,
                "config": config,
            })
        if set(records_by_id) != {sample.sample_id for sample in samples}:
            raise ValueError(f"Incomplete official sample coverage for {dataset}")
        records = [records_by_id[sample.sample_id] for sample in samples]
        config = {
            **reference_core,
            "num_samples": len(samples),
            "samples_sha256": data_fingerprint(samples),
            "device": "multiple_shards",
            "devices": [config["device"] for config in shard_configs],
            "aggregation_complete": True,
            "shards_provenance": provenance,
        }
        metrics = {
            "dataset": dataset,
            "split": SPLITS[dataset],
            "complete": True,
            **classification_metrics(records, LABELS[dataset]),
        }
        validated[dataset] = (config, records, metrics)
        summary[dataset] = metrics

    # All source shards have passed validation. Publish files atomically, with
    # summary.json last; never target files inside shard_N directories.
    for dataset in validated:
        directory = input_dir / dataset
        if directory.is_symlink() or any((directory / name).is_symlink() for name in (
            "config.json", "predictions.jsonl", "metrics.json",
        )):
            raise ValueError(f"Refusing symlinked aggregate output: {directory}")
    if (input_dir / "summary.json").is_symlink():
        raise ValueError("Refusing symlinked aggregate summary.json")
    with tempfile.TemporaryDirectory(prefix=".emotion-aggregate-", dir=input_dir) as temporary:
        stage = Path(temporary)
        for dataset, (config, records, metrics) in validated.items():
            directory = stage / dataset
            directory.mkdir()
            for name, value in (("config.json", config), ("metrics.json", metrics)):
                (directory / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (directory / "predictions.jsonl").write_text(
                "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records), encoding="utf-8"
            )
        (stage / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for dataset in validated:
            directory = input_dir / dataset
            directory.mkdir(exist_ok=True)
            for name in ("config.json", "predictions.jsonl", "metrics.json"):
                (stage / dataset / name).replace(directory / name)
        (stage / "summary.json").replace(input_dir / "summary.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Root containing shard_0, shard_1, ...")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA, help="Parent of EMOSET, Emotion6 and EMO8")
    parser.add_argument("--num-shards", type=int, default=2)
    args = parser.parse_args(argv)
    try:
        summary = aggregate(args.input_dir, args.data_root, args.num_shards)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Aggregation failed: {exc}\n")
    for dataset, metrics in summary.items():
        print(f"[{dataset}] n={metrics['num_samples']} accuracy={metrics['accuracy']:.4f} "
              f"macro_f1={metrics['macro_f1']:.4f} invalid={metrics['num_invalid']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
