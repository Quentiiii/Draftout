#!/usr/bin/env python3
"""Audit complete angles/draft runs and write supplementary comparison reports.

Reads all original result files without modifying them. Requires complete output
for all three datasets and writes only INPUT_DIR/analysis/ after validation.
Uses no model, GPU, network, or additional inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT))

from emotion_eval.data import LABELS, canonical_label, load_emo8, load_emotion6, load_emoset
from emotion_eval.infer import DEFAULT_DATA, data_fingerprint
from emotion_eval.metrics import classification_metrics, parse_angles_response, parse_draft_response
from emotion_eval.prompts import DEEPSEEK_EMOTION_PROMPT, EMOTION_ANALYSIS_ANGLES_PROMPT
from emotion_eval.sampling import validate_decoding_config


SPLITS = {
    "emoset": "official_test",
    "emotion6": "all_images_no_official_test_split",
    "emo8": "all_train_val_test_as_test",
}
BOXED_RULE = (
    "Supplementary only: exactly one \\boxed occurrence containing one complete "
    "\\boxed{label}, with a known label (shared case-insensitive canonicalization). "
    "Other text is allowed; multiple/malformed/unknown-label boxes count as wrong. "
    "The same parser and full sample denominator are used for both runs."
)
SETTING_KEYS = (
    "model", "format_version", "response_format", "enable_thinking", "do_sample",
    "temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty",
    "seed", "seed_policy", "seed_batch_size", "presence_penalty_scope", "batch_size",
    "max_new_tokens", "dtype", "min_pixels", "max_pixels",
)
METRIC_KEYS = ("num_samples", "num_correct", "num_invalid", "accuracy", "macro_f1")


def parse_boxed_prediction(text: str, labels) -> str | None:
    """Apply the same single-box rule to both formats, without requiring a draft."""
    if len(re.findall(r"\\boxed\b", text)) != 1:
        return None
    boxes = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return canonical_label(boxes[0], labels) if len(boxes) == 1 else None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path, hashes: dict[str, str]) -> bytes:
    payload = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(payload).hexdigest()
    return payload


def _read_json(path: Path, hashes: dict[str, str]) -> dict:
    value = json.loads(_read(path, hashes))
    _require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _require_complete_files(root: Path) -> None:
    expected = [root / "summary.json"]
    expected.extend(root / dataset / name for dataset in LABELS
                    for name in ("config.json", "predictions.jsonl", "metrics.json"))
    missing = [str(path.relative_to(root)) for path in expected if not path.is_file()]
    _require(not missing, f"Incomplete run {root}: missing {', '.join(missing)}")


def _validate_metrics(saved: dict, computed: dict, dataset: str, path: str) -> None:
    expected = {"dataset": dataset, "split": SPLITS[dataset], **computed}
    _require(isinstance(saved, dict), f"Expected metrics object: {path}")
    differences = [key for key, value in expected.items() if saved.get(key) != value]
    _require(not differences, f"Metrics disagree with predictions in {path}: {differences}")
    _require(saved.get("complete", True) is True, f"Incomplete metrics: {path}")


def validate_run(root: Path, datasets: dict, style: str) -> dict:
    """Validate a complete full-order run; never call repair-capable resume code."""
    hashes: dict[str, str] = {}
    summary = _read_json(root / "summary.json", hashes)
    _require(set(summary) == set(datasets), f"Summary dataset coverage mismatch: {root}")
    results = {}
    common_settings = None
    for dataset, samples in datasets.items():
        labels = LABELS[dataset]
        directory = root / dataset
        config = _read_json(directory / "config.json", hashes)
        template = EMOTION_ANALYSIS_ANGLES_PROMPT if style == "angles" else DEEPSEEK_EMOTION_PROMPT
        expected = {
            "dataset": dataset, "split": SPLITS[dataset], "num_samples": len(samples),
            "samples_sha256": data_fingerprint(samples), "labels": list(labels),
            "prompt": template.format(labels=", ".join(labels)), "enable_thinking": False,
            "response_format": "angles_then_analysis_then_boxed_label" if style == "angles"
                               else "draft_then_boxed_label",
        }
        if style == "angles":
            expected.update(format_version=4, prompt_style="angles")
        else:
            _require(config.get("format_version") in (2, 3), f"Unsupported draft format: {directory}")
        differences = [key for key, value in expected.items() if config.get(key) != value]
        _require(not differences, f"Configuration mismatch in {directory}: {differences}")
        validate_decoding_config(config)
        if config.get("do_sample"):
            _require(config.get("seed_batch_size") == config.get("batch_size"),
                     f"Sampling batch size mismatch: {directory}")
        shard_count = config.get("num_shards", 1)
        _require(type(shard_count) is int and shard_count > 0, f"Invalid shard count: {directory}")
        if shard_count > 1:
            _require(config.get("aggregation_complete") is True and "shard_index" not in config,
                     f"Expected complete aggregate, not an individual shard: {directory}")
        settings = {key: config.get(key) for key in SETTING_KEYS}
        settings.update(prompt_style=style, num_shards=shard_count)
        if common_settings is None:
            common_settings = settings
        _require(settings == common_settings, f"Inconsistent settings across datasets in {root}")

        prediction_path = directory / "predictions.jsonl"
        payload = _read(prediction_path, hashes)
        records = []
        for number, line in enumerate(payload.splitlines(), 1):
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f"Incomplete or malformed JSONL: {prediction_path}:{number}") from exc
            _require(isinstance(record, dict), f"Expected prediction object: {prediction_path}:{number}")
            records.append(record)
        _require(len(records) == len(samples),
                 f"Incomplete run {root}/{dataset}: {len(records)}/{len(samples)} predictions")
        boxed_records = []
        structure_valid = 0
        for index, (record, sample) in enumerate(zip(records, samples, strict=True), 1):
            where = f"{prediction_path}:{index}"
            identity = {"dataset": dataset, "sample_id": sample.sample_id,
                        "image_path": str(sample.image_path), "label": sample.label}
            _require(all(record.get(key) == value for key, value in identity.items()),
                     f"Official sample ID/order/ground truth/path mismatch: {where}")
            raw = record.get("raw_output")
            _require(isinstance(raw, str), f"Missing raw model output: {where}")
            if style == "angles":
                draft, analysis, prediction = parse_angles_response(raw, labels)
                _require("analysis" in record and record["analysis"] == analysis,
                         f"Saved analysis differs from raw output: {where}")
            else:
                draft, prediction = parse_draft_response(raw, labels)
            _require("draft" in record and record["draft"] == draft,
                     f"Saved draft differs from raw output: {where}")
            _require("prediction" in record and record["prediction"] == prediction,
                     f"Saved prediction differs from strict parser: {where}")
            _require(type(record.get("correct")) is bool and record["correct"] == (prediction == sample.label),
                     f"Saved correctness differs from strict parser: {where}")
            structure_valid += draft is not None
            boxed_records.append({"label": sample.label, "prediction": parse_boxed_prediction(raw, labels)})
        strict_metrics = classification_metrics(records, labels)
        metrics = _read_json(directory / "metrics.json", hashes)
        _validate_metrics(metrics, strict_metrics, dataset, str(directory / "metrics.json"))
        _validate_metrics(summary[dataset], strict_metrics, dataset, f"{root}/summary.json/{dataset}")
        results[dataset] = {
            "records": records, "boxed_records": boxed_records,
            "strict": strict_metrics, "boxed_only": classification_metrics(boxed_records, labels),
            "samples_sha256": expected["samples_sha256"],
            "format": {"num_structure_valid": structure_valid,
                       "num_structure_invalid": len(records) - structure_valid,
                       "num_valid_structure_unknown_label": structure_valid -
                           (len(records) - strict_metrics["num_invalid"])},
        }
    return {"directory": str(root), "settings": common_settings, "datasets": results, "source_sha256": hashes}


def _comparison(current: dict, baseline: dict, dataset: str, scoring: str) -> dict:
    after, before = current["datasets"][dataset], baseline["datasets"][dataset]
    metrics_after, metrics_before = after[scoring], before[scoring]
    rows_key = "records" if scoring == "strict" else "boxed_records"
    improved = regressed = changed = 0
    for new, old in zip(after[rows_key], before[rows_key], strict=True):
        new_correct, old_correct = new["prediction"] == new["label"], old["prediction"] == old["label"]
        improved += new_correct and not old_correct
        regressed += old_correct and not new_correct
        changed += new["prediction"] != old["prediction"]
    return {
        "angles": {key: metrics_after[key] for key in METRIC_KEYS},
        "draft_baseline": {key: metrics_before[key] for key in METRIC_KEYS},
        "accuracy_delta_pp": 100 * (metrics_after["accuracy"] - metrics_before["accuracy"]),
        "macro_f1_delta_pp": 100 * (metrics_after["macro_f1"] - metrics_before["macro_f1"]),
        "num_correct_delta": metrics_after["num_correct"] - metrics_before["num_correct"],
        "num_invalid_delta": metrics_after["num_invalid"] - metrics_before["num_invalid"],
        "paired": {"wrong_to_correct": improved, "correct_to_wrong": regressed, "changed_prediction": changed},
    }


def analyze(input_dir: Path, baseline_dir: Path, data_root: Path = DEFAULT_DATA) -> dict:
    input_dir, baseline_dir, data_root = (path.expanduser().resolve() for path in (input_dir, baseline_dir, data_root))
    _require(input_dir != baseline_dir, "Input and baseline directories must differ")
    for root in (input_dir, baseline_dir):
        _require_complete_files(root)
    datasets = {
        "emoset": load_emoset(data_root / "EMOSET"),
        "emotion6": load_emotion6(data_root / "Emotion6"),
        "emo8": load_emo8(data_root / "EMO8"),
    }
    current = validate_run(input_dir, datasets, "angles")
    baseline = validate_run(baseline_dir, datasets, "draft")
    _require(current["settings"]["model"] == baseline["settings"]["model"], "Model paths differ between runs")
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    limitation = (
        "This is not a controlled prompt-only ablation. Compare the recorded settings: "
        "the planned angles run uses batch_size=32, max_new_tokens=1024 and one shard, "
        "whereas the draft baseline uses batch_size=64, max_new_tokens=512 and two shards. "
        "Changing batch membership or sharding changes batch-derived random seeds even with seed=42. "
        "Strict scoring also requires different output structures; boxed-only scoring applies one shared rule."
    )
    differences = {key: {"angles": value, "draft_baseline": baseline["settings"].get(key)}
                   for key, value in current["settings"].items() if value != baseline["settings"].get(key)}
    comparison = {
        "validated_at": now, "angles_directory": str(input_dir), "baseline_directory": str(baseline_dir),
        "settings": {"angles": current["settings"], "draft_baseline": baseline["settings"]},
        "setting_differences": differences, "interpretation": limitation,
        "datasets": {dataset: {scoring: _comparison(current, baseline, dataset, scoring)
                               for scoring in ("strict", "boxed_only")} for dataset in datasets},
    }
    validation = {
        "validated_at": now, "complete": True, "total_samples_per_run": sum(map(len, datasets.values())),
        "checks": ["full official sample ID/order/ground truth/image path and configured fingerprint",
                   "exact prompt and output format", "saved draft/analysis/prediction/correctness vs raw output",
                   "all metrics and summary recomputed", "original result files unchanged during audit"],
        "scope": "Fingerprints cover IDs, paths and labels; image bytes and model weights are not hashed. "
                 "Emotion6 uses every local image; EMO8 includes train, val and test; EmoSet uses official test.",
        "format_statistics_scope": "Outer syntax only; no semantic judgment of plan quality, angle count or analysis.",
        "runs": {name: {"directory": run["directory"], "source_sha256": run["source_sha256"],
                        "datasets": {dataset: {"num_samples": len(datasets[dataset]),
                                              "samples_sha256": values["samples_sha256"],
                                              "format": values["format"]}
                                     for dataset, values in run["datasets"].items()}}
                 for name, run in (("angles", current), ("draft_baseline", baseline))},
    }
    boxed = {
        "validated_at": now, "supplementary_only": True, "scoring_rule": BOXED_RULE,
        "runs": {name: {"directory": run["directory"], "datasets": {
            dataset: values["boxed_only"] for dataset, values in run["datasets"].items()}}
                 for name, run in (("angles", current), ("draft_baseline", baseline))},
    }
    lines = ["Angles examples: first official-order sample for each ground-truth class.",
             "Selection does not depend on prediction, correctness, or output quality.", ""]
    for dataset, values in current["datasets"].items():
        for label in LABELS[dataset]:
            record = next(row for row in values["records"] if row["label"] == label)
            boxed_prediction = parse_boxed_prediction(record["raw_output"], LABELS[dataset])
            lines.extend([f"===== {dataset} / {label} =====", f"Sample ID: {record['sample_id']}",
                          f"Image: {record['image_path']}", f"Ground truth: {label}",
                          f"Strict prediction: {record['prediction']}", f"Boxed-only prediction: {boxed_prediction}",
                          f"Correct (strict): {record['correct']}", "Raw output:", record["raw_output"], ""])
    # Ensure an actively changing or replaced source cannot be published as validated.
    for run in (current, baseline):
        for path, digest in run["source_sha256"].items():
            _require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,
                     f"Source changed while auditing: {path}; rerun after completion")
    destination = input_dir / "analysis"
    reports = {"comparison.json": comparison, "boxed_only_summary.json": boxed, "validation.json": validation}
    _require(not destination.is_symlink(), f"Refusing symlinked analysis directory: {destination}")
    for name in (*reports, "examples.txt"):
        _require(not (destination / name).is_symlink(), f"Refusing symlinked analysis output: {destination / name}")
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".analysis-", dir=destination) as temporary:
        stage = Path(temporary)
        for name, report in reports.items():
            (stage / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "examples.txt").write_text("\n".join(lines), encoding="utf-8")
        # Publish the validation marker last, after every other report is ready.
        for name in ("comparison.json", "boxed_only_summary.json", "examples.txt", "validation.json"):
            (stage / name).replace(destination / name)
    return comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT / "outputs/qwen35_4b_emotion_angles_sampled_seed42")
    parser.add_argument("--baseline-dir", type=Path,
                        default=PROJECT / "outputs/qwen35_4b_emotion_draft_sampled_seed42")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args(argv)
    try:
        comparison = analyze(args.input_dir, args.baseline_dir, args.data_root)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        parser.exit(1, f"Analysis refused: {exc}\n")
    for dataset, values in comparison["datasets"].items():
        for scoring in ("strict", "boxed_only"):
            row = values[scoring]
            print(f"[{dataset}/{scoring}] accuracy={row['angles']['accuracy']:.4%} "
                  f"delta={row['accuracy_delta_pp']:+.4f} pp "
                  f"macro_f1={row['angles']['macro_f1']:.4%} delta={row['macro_f1_delta_pp']:+.4f} pp")
    print(f"Reports: {args.input_dir.resolve() / 'analysis'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
