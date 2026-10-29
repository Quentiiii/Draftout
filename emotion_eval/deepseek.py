"""Evaluate EmoSet, Emotion6 and EMO8 using DeepSeek's image API (no GPU)."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from http.client import IncompleteRead
from pathlib import Path
from urllib.parse import urlsplit

from .data import LABELS, Sample, canonical_label, load_emo8, load_emotion6, load_emoset
from .infer import read_predictions, write_json
from .metrics import classification_metrics, parse_draft_prediction, parse_draft_response
from .prompts import DEEPSEEK_EMOTION_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = LABELS["emoset"]
PROMPT = DEEPSEEK_EMOTION_PROMPT.format(labels=", ".join(EMOTIONS))
ARTIFACTS = ("config.json", "samples.json", "predictions.jsonl", "errors.jsonl", "metrics.json")
DATASET_DIRECTORIES = {"emoset": "EMOSET", "emotion6": "Emotion6", "emo8": "EMO8"}


def dataset_prompt(dataset: str) -> str:
    return DEEPSEEK_EMOTION_PROMPT.format(labels=", ".join(LABELS[dataset]))


@dataclass
class EvaluationRun:
    args: argparse.Namespace
    samples: list[Sample]
    manifest: list[dict] = field(default_factory=list)
    records: list[dict] = field(default_factory=list)


def select_samples(root: Path, test_list: Path | None, limit: int | None = None) -> list[Sample]:
    """Require every selected row to belong to the official, annotation-checked test set."""
    if test_list is None:
        return limit_samples(load_emoset(root), limit)
    official_rows = {row[1]: row for row in json.loads((root / "test.json").read_text(encoding="utf-8"))}
    rows = json.loads(test_list.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{test_list}: expected a nonempty list of [label, image, annotation]")
    seen = set()
    for row in rows:
        if not isinstance(row, list) or len(row) != 3 or not all(isinstance(v, str) for v in row):
            raise ValueError(f"{test_list}: malformed test row")
        label, sample_id, annotation = row
        official = official_rows.get(sample_id)
        if official is None or sample_id in seen:
            raise ValueError(f"Unknown or duplicate official test image: {sample_id}")
        if canonical_label(label, EMOTIONS) != canonical_label(official[0], EMOTIONS) or annotation != official[2]:
            raise ValueError(f"Subset row does not match the official test annotation: {sample_id}")
        seen.add(sample_id)
    return limit_samples(load_emoset(root, test_list), limit)


def limit_samples(samples: list[Sample], limit: int | None) -> list[Sample]:
    if limit is not None and limit > len(samples):
        raise ValueError(f"--limit {limit} exceeds the {len(samples)} available {samples[0].dataset} images")
    return samples if limit is None else samples[:limit]


def encode_image(path: Path, max_side: int) -> str:
    """Apply EXIF orientation, resize, and send pixels without filenames or metadata."""
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        try:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            image.info.clear()
            with io.BytesIO() as buffer:
                image.save(buffer, format="JPEG", quality=95)
                encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        finally:
            image.close()
    return "data:image/jpeg;base64," + encoded


def build_payload(args: argparse.Namespace, image_url: str, dataset: str = "emoset") -> dict:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            {"type": "text", "text": dataset_prompt(dataset)},
        ]}],
        "thinking": {"type": "disabled"},
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "stream": False,
    }


class APIError(RuntimeError):
    """A safe error message that excludes credentials and request/response bodies."""


def request_completion(args: argparse.Namespace, api_key: str, payload: dict) -> dict:
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(args.max_retries + 1):
        delay = min(2 ** attempt, 30)
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise APIError("API returned a non-object JSON response")
            return result
        except urllib.error.HTTPError as exc:
            status = exc.code
            retry_after = exc.headers.get("Retry-After", "") if exc.headers else ""
            exc.close()
            retryable = status in (408, 409, 429) or 500 <= status < 600
            hint = {
                400: "check model, image support and request parameters",
                401: "check DEEPSEEK_API_KEY",
                402: "check your DeepSeek balance",
                403: "check API permissions",
                404: "check --base-url and --model",
                422: "check request parameters",
                429: "rate limit reached; try fewer --workers",
            }.get(status, "request failed")
            message = f"DeepSeek HTTP {status}: {hint}"
            if not retryable or attempt == args.max_retries:
                raise APIError(message) from None
            try:
                retry_seconds = float(retry_after)
                if math.isfinite(retry_seconds):
                    delay = max(delay, min(max(retry_seconds, 0), 60))
            except ValueError:
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError, IncompleteRead) as exc:
            message = f"DeepSeek connection failed ({type(exc).__name__})"
            if attempt == args.max_retries:
                raise APIError(message) from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise APIError("DeepSeek returned invalid JSON") from None
        print(f"{message}; retry {attempt + 1}/{args.max_retries} in {delay:g}s", file=sys.stderr, flush=True)
        time.sleep(delay)
    raise AssertionError("unreachable")


def predict_sample(args: argparse.Namespace, api_key: str, sample: Sample) -> dict:
    started = time.monotonic()
    payload = build_payload(args, encode_image(sample.image_path, args.max_image_side), sample.dataset)
    result = request_completion(args, api_key, payload)
    try:
        choice = result["choices"][0]
        raw = choice["message"]["content"]
        finish_reason = choice["finish_reason"]
        if raw is None:
            raw = ""
        if not isinstance(raw, str) or not isinstance(finish_reason, str):
            raise TypeError
    except (KeyError, IndexError, TypeError):
        raise APIError("DeepSeek returned a malformed chat completion") from None
    draft, prediction = parse_draft_response(raw, LABELS[sample.dataset])
    return {
        "dataset": sample.dataset, "sample_id": sample.sample_id, "image_path": str(sample.image_path),
        "label": sample.label, "prediction": prediction, "draft": draft, "raw_output": raw,
        "correct": prediction == sample.label, "finish_reason": finish_reason,
        "response_model": result.get("model"), "response_id": result.get("id"),
        "usage": result.get("usage") or {}, "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def run_config(args: argparse.Namespace, samples: list[Sample], manifest: list[dict]) -> dict:
    return {
        "format_version": 2, "dataset": args.dataset, "split": args.split,
        "data_root": str(args.data_root), "test_list": str(args.test_list) if args.test_list else None,
        "num_samples": len(samples),
        "samples_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest(),
        "model": args.model, "base_url": args.base_url,
        "prompt": dataset_prompt(args.dataset), "labels": list(LABELS[args.dataset]),
        "thinking": "disabled", "temperature": 0, "max_tokens": args.max_tokens,
        "response_format": "draft_then_boxed_label",
        "image": {"max_side": args.max_image_side, "format": "JPEG", "quality": 95, "detail": "high"},
    }


def validate_run(args: argparse.Namespace, samples: list[Sample], manifest: list[dict]) -> dict:
    """Check every selected run before initializing files or making paid requests."""
    config = run_config(args, samples, manifest)
    directory = args.output_dir
    existing = [name for name in ARTIFACTS if (directory / name).exists()]
    config_path = directory / "config.json"
    if existing and not args.resume:
        raise ValueError(f"Results already exist in {directory}; use --resume or a new --output-dir")
    if existing:
        if not config_path.exists() or not (directory / "samples.json").exists():
            raise ValueError("Cannot resume without config.json and samples.json; use a new --output-dir")
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            changes = [key for key in config if old.get(key) != config[key]]
            raise ValueError(f"Cannot resume with changed settings/data: {changes}; use a new --output-dir")
        if json.loads((directory / "samples.json").read_text(encoding="utf-8")) != manifest:
            raise ValueError("Saved sample manifest does not match current images/annotations")
    return config


def prepare_run(args: argparse.Namespace, samples: list[Sample], manifest: list[dict]) -> list[dict]:
    config = validate_run(args, samples, manifest)
    directory = args.output_dir
    if not (directory / "config.json").exists():
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / "config.json", config)
        write_json(directory / "samples.json", manifest)
    return read_predictions(
        directory / "predictions.jsonl", samples, LABELS[args.dataset], prediction_parser=parse_draft_prediction
    )


def save_metrics(
    directory: Path, samples: list[Sample], records: list[dict], *, split: str = "official_test_subset"
) -> dict:
    dataset = samples[0].dataset
    metrics = {
        "dataset": dataset, "split": split,
        "complete": len(records) == len(samples), "num_selected": len(samples),
        "num_pending": len(samples) - len(records),
        "accuracy_denominator": "completed API responses, including invalid answers",
        **classification_metrics(records, LABELS[dataset]),
        "num_truncated": sum(record.get("finish_reason") == "length" for record in records),
        "usage": {
            key: sum((record.get("usage") or {}).get(key, 0) or 0 for record in records)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
    }
    write_json(directory / "metrics.json", metrics)
    return metrics


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--datasets", nargs="+", choices=("all", *LABELS),
        help="Datasets to evaluate; defaults to all three (EMO8 includes train/val/test)",
    )
    parser.add_argument(
        "--data-root", type=Path,
        help="Parent of EMOSET/, Emotion6/ and EMO8/; defaults to ../dataset",
    )
    parser.add_argument(
        "--test-list", type=Path,
        help="Optional EmoSet official-test subset JSON; without --datasets, use legacy single-dataset layout",
    )
    parser.add_argument("--emotion6-test-list", type=Path, help="Optional Emotion6 image list, one relative path per line")
    parser.add_argument(
        "--output-dir", type=Path,
        help="Results root; defaults to outputs/deepseek_v41_flash_emotion, with one directory per dataset",
    )
    parser.add_argument("--model", default="deepseek-flash", help="Official API name for DeepSeek-V4.1-Flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests; one image per request")
    parser.add_argument("--max-tokens", type=int, default=512, help="Output budget for explanation and boxed answer")
    parser.add_argument("--max-image-side", type=int, default=1024, help="Shrink large images preserving aspect ratio")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries after transient HTTP/network errors")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N images of EACH dataset, for a smoke test")
    parser.add_argument("--resume", action="store_true", help="Skip saved responses after checking settings and images")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate data/images without API calls or output writes"
    )
    return parser


def select_runs(args: argparse.Namespace) -> list[EvaluationRun]:
    runs = []
    for dataset in args.datasets:
        run_args = argparse.Namespace(**vars(args))
        run_args.dataset = dataset
        run_args.data_root = args.data_root if args.legacy_subset else args.data_root / DATASET_DIRECTORIES[dataset]
        run_args.output_dir = args.output_dir if args.legacy_subset else args.output_dir / dataset
        run_args.test_list = args.test_list if dataset == "emoset" else None
        if dataset == "emoset":
            samples = select_samples(run_args.data_root, run_args.test_list, args.limit)
            run_args.split = "official_test_subset" if run_args.test_list else "official_test"
        elif dataset == "emotion6":
            run_args.test_list = args.emotion6_test_list
            samples = limit_samples(load_emotion6(run_args.data_root, run_args.test_list), args.limit)
            run_args.split = "custom_test_list" if run_args.test_list else "all_images_no_official_test_split"
        else:
            samples = limit_samples(load_emo8(run_args.data_root), args.limit)
            run_args.split = "all_train_val_test_as_test"
        print(json.dumps({
            "dataset": dataset, "split": run_args.split, "selected": len(samples),
            "class_counts": dict(Counter(s.label for s in samples)), "model": args.model,
            "test_list": str(run_args.test_list) if run_args.test_list else None,
        }, ensure_ascii=False), flush=True)
        runs.append(EvaluationRun(run_args, samples))
    return runs


def evaluate_run(run: EvaluationRun, api_key: str) -> int:
    args, samples, records = run.args, run.samples, run.records
    seen = {record["sample_id"] for record in records}
    pending = [sample for sample in samples if sample.sample_id not in seen]
    failed = False
    correct = sum(row["correct"] for row in records)
    print(f"[{args.dataset}] saved={len(records)} pending={len(pending)}", flush=True)
    try:
        with (
            (args.output_dir / "predictions.jsonl").open("a", encoding="utf-8", buffering=1) as predictions,
            (args.output_dir / "errors.jsonl").open("a", encoding="utf-8", buffering=1) as errors,
            ThreadPoolExecutor(max_workers=args.workers) as executor,
        ):
            # Keep at most one batch in flight. On an error, save successful peers and stop.
            for offset in range(0, len(pending), args.workers):
                futures = {
                    executor.submit(predict_sample, args, api_key, sample): sample
                    for sample in pending[offset:offset + args.workers]
                }
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        message = str(exc) if isinstance(exc, APIError) else f"Local failure ({type(exc).__name__})"
                        message = message.replace(api_key, "[REDACTED]")
                        errors.write(json.dumps({
                            "dataset": args.dataset, "sample_id": sample.sample_id, "error": message,
                        }) + "\n")
                        print(f"[{args.dataset}] {sample.sample_id}: {message}", file=sys.stderr, flush=True)
                        failed = True
                    else:
                        predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records.append(record)
                        correct += record["correct"]
                        print(f"[{args.dataset} {len(records)}/{len(samples)}] "
                              f"accuracy={correct / len(records):.4f}", flush=True)
                save_metrics(args.output_dir, samples, records, split=args.split)
                if failed:
                    print("Stopped after request errors. Fix the issue and use --resume.", file=sys.stderr)
                    break
    finally:
        metrics = save_metrics(args.output_dir, samples, records, split=args.split)
    print(f"[{args.dataset}] complete={metrics['complete']} accuracy={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f} invalid={metrics['num_invalid']} "
          f"pending={metrics['num_pending']}\nResults: {args.output_dir}", flush=True)
    return 1 if failed else 0


def save_summary(args: argparse.Namespace, runs: list[EvaluationRun]) -> None:
    summary = {
        run.args.dataset: save_metrics(run.args.output_dir, run.samples, run.records, split=run.args.split)
        for run in runs
    }
    if not args.legacy_subset:
        write_json(args.output_dir / "summary.json", summary)


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    for key in ("workers", "max_tokens", "max_image_side", "limit"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not math.isfinite(args.timeout) or args.timeout <= 0 or args.max_retries < 0:
        parser.error("--timeout must be finite and positive; --max-retries must be nonnegative")
    if args.max_image_side > 8192:
        parser.error("--max-image-side must be <= 8192 (API image dimension limit)")
    args.base_url = args.base_url.rstrip("/")
    endpoint = urlsplit(args.base_url)
    if endpoint.scheme not in ("http", "https") or not endpoint.netloc or endpoint.username or endpoint.password:
        parser.error("--base-url must be an HTTP(S) URL without credentials")
    if endpoint.query or endpoint.fragment:
        parser.error("--base-url must not contain a query or fragment")

    # Explicit old --test-list commands can still resume the saved flat 2k run.
    args.legacy_subset = args.test_list is not None and args.datasets is None
    args.datasets = args.datasets or (["emoset"] if args.legacy_subset else list(LABELS))
    if "all" in args.datasets:
        if len(args.datasets) != 1:
            parser.error("--datasets all must be used on its own")
        args.datasets = list(LABELS)
    if len(args.datasets) != len(set(args.datasets)):
        parser.error("--datasets must not contain duplicates")
    if args.test_list is not None and "emoset" not in args.datasets:
        parser.error("--test-list requires the emoset dataset")
    if args.emotion6_test_list is not None and "emotion6" not in args.datasets:
        parser.error("--emotion6-test-list requires the emotion6 dataset")
    default_data = PROJECT_ROOT.parent / "dataset"
    args.data_root = args.data_root or (default_data / "EMOSET" if args.legacy_subset else default_data)
    args.output_dir = args.output_dir or PROJECT_ROOT / "outputs" / (
        "deepseek_v41_flash_emoset_2k" if args.legacy_subset else "deepseek_v41_flash_emotion"
    )
    for key in ("data_root", "test_list", "emotion6_test_list", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if not args.legacy_subset and any((args.output_dir / name).exists() for name in ARTIFACTS):
        raise ValueError("Output root contains a legacy single-dataset run; use a new --output-dir "
                         "or provide --test-list without --datasets to resume it")
    if args.legacy_subset and (
        (args.output_dir / "summary.json").exists()
        or any((args.output_dir / dataset / name).exists() for dataset in LABELS for name in ARTIFACTS)
    ):
        raise ValueError("Output root contains a multi-dataset run; use a new --output-dir "
                         "for the legacy subset, or pass --datasets to keep the multi-dataset layout")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        parser.error("Set DEEPSEEK_API_KEY, or fill .deepseek.env and use scripts/run_deepseek_emotion.sh")

    print("Checking all selected datasets and annotations...", flush=True)
    runs = select_runs(args)
    total = sum(len(run.samples) for run in runs)
    print(f"Selected {total} images across {len(runs)} dataset(s).", flush=True)
    for run in runs:
        print(f"[{run.args.dataset}] Checking image decoding and content fingerprints...", flush=True)
        for index, sample in enumerate(run.samples, 1):
            encode_image(sample.image_path, args.max_image_side)
            run.manifest.append({
                "sample_id": sample.sample_id, "image_path": str(sample.image_path), "label": sample.label,
                "image_sha256": hashlib.sha256(sample.image_path.read_bytes()).hexdigest(),
            })
            if index % 500 == 0 or index == len(run.samples):
                print(f"[{run.args.dataset}] checked {index}/{len(run.samples)} images", flush=True)
    if args.dry_run:
        scope = "official test images" if args.legacy_subset else "images across selected datasets"
        print(f"Dry run passed: {total} {scope}; no API requests or output files.", flush=True)
        return 0

    for run in runs:
        validate_run(run.args, run.samples, run.manifest)
    for run in runs:
        run.records = prepare_run(run.args, run.samples, run.manifest)
    save_summary(args, runs)
    try:
        for run in runs:
            status = evaluate_run(run, api_key)
            save_summary(args, runs)
            if status:
                return status
    finally:
        save_summary(args, runs)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; saved responses can be resumed with --resume.", file=sys.stderr)
        raise SystemExit(130) from None
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
