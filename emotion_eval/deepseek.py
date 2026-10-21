"""Evaluate the fixed EmoSet test subset using DeepSeek's image API (no GPU)."""

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
from http.client import IncompleteRead
from pathlib import Path
from urllib.parse import urlsplit

from .data import LABELS, Sample, canonical_label, load_emoset
from .infer import read_predictions, write_json
from .metrics import classification_metrics, parse_draft_prediction, parse_draft_response
from .prompts import DEEPSEEK_EMOTION_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMOTIONS = LABELS["emoset"]
PROMPT = DEEPSEEK_EMOTION_PROMPT.format(labels=", ".join(EMOTIONS))
ARTIFACTS = ("config.json", "samples.json", "predictions.jsonl", "errors.jsonl", "metrics.json")


def select_samples(root: Path, test_list: Path, limit: int | None = None) -> list[Sample]:
    """Require every selected row to belong to the official, annotation-checked test set."""
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
    samples = load_emoset(root, test_list)
    if limit is not None and limit > len(samples):
        raise ValueError(f"--limit {limit} exceeds the {len(samples)} available subset images")
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


def build_payload(args: argparse.Namespace, image_url: str) -> dict:
    return {
        "model": args.model,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url, "detail": "high"}},
            {"type": "text", "text": PROMPT},
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
    payload = build_payload(args, encode_image(sample.image_path, args.max_image_side))
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
    draft, prediction = parse_draft_response(raw, EMOTIONS)
    return {
        "dataset": "emoset", "sample_id": sample.sample_id, "image_path": str(sample.image_path),
        "label": sample.label, "prediction": prediction, "draft": draft, "raw_output": raw,
        "correct": prediction == sample.label, "finish_reason": finish_reason,
        "response_model": result.get("model"), "response_id": result.get("id"),
        "usage": result.get("usage") or {}, "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def prepare_run(args: argparse.Namespace, samples: list[Sample], manifest: list[dict]) -> list[dict]:
    config = {
        "format_version": 2, "dataset": "emoset", "split": "official_test_subset",
        "data_root": str(args.data_root), "test_list": str(args.test_list), "num_samples": len(samples),
        "samples_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode("utf-8")).hexdigest(),
        "model": args.model, "base_url": args.base_url, "prompt": PROMPT, "labels": list(EMOTIONS),
        "thinking": "disabled", "temperature": 0, "max_tokens": args.max_tokens,
        "response_format": "draft_then_boxed_label",
        "image": {"max_side": args.max_image_side, "format": "JPEG", "quality": 95, "detail": "high"},
    }
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
    else:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(config_path, config)
        write_json(directory / "samples.json", manifest)
    return read_predictions(
        directory / "predictions.jsonl", samples, EMOTIONS, prediction_parser=parse_draft_prediction
    )


def save_metrics(directory: Path, samples: list[Sample], records: list[dict]) -> dict:
    metrics = {
        "dataset": "emoset", "split": "official_test_subset",
        "complete": len(records) == len(samples), "num_selected": len(samples),
        "num_pending": len(samples) - len(records),
        "accuracy_denominator": "completed API responses, including invalid answers",
        **classification_metrics(records, EMOTIONS),
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
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT.parent / "dataset" / "EMOSET")
    parser.add_argument("--test-list", type=Path, default=PROJECT_ROOT / "splits" / "emoset_test_2000.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "deepseek_v41_flash_emoset_2k")
    parser.add_argument("--model", default="deepseek-flash", help="Official API name for DeepSeek-V4.1-Flash")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests; one image per request")
    parser.add_argument("--max-tokens", type=int, default=512, help="Output budget for explanation and boxed answer")
    parser.add_argument("--max-image-side", type=int, default=1024, help="Shrink large images preserving aspect ratio")
    parser.add_argument("--timeout", type=float, default=120, help="HTTP timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries after transient HTTP/network errors")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N subset entries, for a smoke test")
    parser.add_argument("--resume", action="store_true", help="Skip saved responses after checking settings and images")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate data/images without API calls or output writes"
    )
    return parser


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
    for key in ("data_root", "test_list", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not args.dry_run and not api_key:
        parser.error("Set DEEPSEEK_API_KEY, or fill .deepseek.env and use scripts/run_deepseek_emoset.sh")

    print("Checking test subset membership and annotations...", flush=True)
    samples = select_samples(args.data_root, args.test_list, args.limit)
    print(json.dumps({"selected": len(samples), "class_counts": dict(Counter(s.label for s in samples)),
                      "model": args.model, "test_list": str(args.test_list)}, ensure_ascii=False), flush=True)
    manifest = []
    print("Checking selected image decoding and content fingerprints...", flush=True)
    for sample in samples:
        encode_image(sample.image_path, args.max_image_side)
        manifest.append({"sample_id": sample.sample_id, "image_path": str(sample.image_path), "label": sample.label,
                         "image_sha256": hashlib.sha256(sample.image_path.read_bytes()).hexdigest()})
    if args.dry_run:
        print(f"Dry run passed: {len(samples)} official test images; no API requests or output files.", flush=True)
        return 0

    records = prepare_run(args, samples, manifest)
    seen = {record["sample_id"] for record in records}
    pending = [sample for sample in samples if sample.sample_id not in seen]
    failed = False
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
                        errors.write(json.dumps({"sample_id": sample.sample_id, "error": message}) + "\n")
                        print(f"{sample.sample_id}: {message}", file=sys.stderr, flush=True)
                        failed = True
                    else:
                        predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                        records.append(record)
                        correct = sum(row["correct"] for row in records)
                        print(f"[{len(records)}/{len(samples)}] accuracy={correct / len(records):.4f}", flush=True)
                save_metrics(args.output_dir, samples, records)
                if failed:
                    print("Stopped after request errors. Fix the issue and use --resume.", file=sys.stderr)
                    break
    finally:
        metrics = save_metrics(args.output_dir, samples, records)
    print(f"complete={metrics['complete']} accuracy={metrics['accuracy']:.4f} "
          f"macro_f1={metrics['macro_f1']:.4f} invalid={metrics['num_invalid']} "
          f"pending={metrics['num_pending']}\nResults: {args.output_dir}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; saved responses can be resumed with --resume.", file=sys.stderr)
        raise SystemExit(130) from None
    except (ValueError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
