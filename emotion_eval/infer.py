"""Zero-shot Qwen3.5 inference on local EmoSet, Emotion6 and EMO8 images."""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from .data import LABELS, Sample, load_emo8, load_emotion6, load_emoset
from .metrics import (
    classification_metrics, parse_angles_prediction, parse_angles_response,
    parse_draft_prediction, parse_draft_response, parse_prediction,
)
from .prompts import DEEPSEEK_EMOTION_PROMPT, EMOTION_ANALYSIS_ANGLES_PROMPT
from .sampling import PresencePenaltyLogitsProcessor, batch_seed, build_decoding_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path("/data/xiyan.zhao/work/model/Qwen3.5-2B")
DEFAULT_DATA = Path("/data/xiyan.zhao/work/dataset")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def data_fingerprint(samples: list[Sample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(json.dumps([sample.sample_id, str(sample.image_path), sample.label]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def read_predictions(
    path: Path,
    samples: list[Sample],
    labels: tuple[str, ...],
    *,
    prediction_parser: Callable[[str, Sequence[str]], str | None] = parse_prediction,
) -> list[dict]:
    """Validate saved results before skipping them on a resumed run."""
    if not path.exists():
        return []
    expected = {sample.sample_id: sample for sample in samples}
    records = []
    seen = set()
    with path.open("rb+") as handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                # A killed process can leave one incomplete final write.
                if not line.endswith(b"\n") and not handle.read(1):
                    print(f"Recovering incomplete final JSONL record in {path}", file=sys.stderr)
                    handle.truncate(start)
                    break
                raise ValueError(f"Invalid JSONL in {path} at byte {start}") from exc
            sample = expected.get(record.get("sample_id"))
            if sample is None or sample.sample_id in seen:
                raise ValueError(f"Unknown or duplicate sample in {path}: {record.get('sample_id')}")
            if (
                record.get("dataset") != sample.dataset
                or record.get("label") != sample.label
                or record.get("image_path") != str(sample.image_path)
                or not isinstance(record.get("raw_output"), str)
                or "prediction" not in record
                or record["prediction"] != prediction_parser(record["raw_output"], labels)
            ):
                raise ValueError(f"Saved prediction does not match current data/parser: {sample.sample_id}")
            record["correct"] = record["prediction"] == sample.label
            seen.add(sample.sample_id)
            records.append(record)
            if not line.endswith(b"\n"):
                # A complete JSON object with a missing final newline is safe.
                handle.write(b"\n")
    return records


class QwenPredictor:
    def __init__(self, args: argparse.Namespace):
        if args.device == "cpu":
            # Transformers 5.5 chooses optional CUDA kernels at import time.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            import torch
            from transformers import AutoProcessor, GenerationConfig, Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Use the vision-opd environment (Transformers 5.5.0 with Qwen3.5 support). "
                "The base Python environment is too old."
            ) from exc

        self.torch = torch
        self.args = args
        self.decoding = build_decoding_config(args)
        if args.device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = args.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Use a GPU node, or explicitly select --device cpu.")
        if device == "cpu":
            print("Using CPU; full-dataset inference will be slow.", file=sys.stderr)
        if args.dtype == "auto":
            if device == "cpu":
                dtype = torch.float32
            else:
                with torch.cuda.device(device):
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = getattr(torch, args.dtype)

        self.processor = AutoProcessor.from_pretrained(str(args.model), local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(args.model),
            dtype=dtype,
            device_map=device,
            attn_implementation="sdpa",
            local_files_only=True,
        ).eval()
        generation_kwargs = {"do_sample": args.do_sample, "repetition_penalty": args.repetition_penalty}
        if args.do_sample:
            generation_kwargs.update({key: getattr(args, key) for key in (
                "temperature", "top_p", "top_k", "min_p",
            )})
        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            **generation_kwargs,
            num_beams=1,
            use_cache=True,
            eos_token_id=self.model.generation_config.eos_token_id,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        print(f"Loaded {args.model} on {self.model.device}, dtype={dtype}", flush=True)

    def predict(self, samples: list[Sample], prompt: str) -> list[str]:
        from PIL import Image, ImageOps

        # The template contains a placeholder, never a path, filename or answer.
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        images = []
        try:
            for sample in samples:
                with Image.open(sample.image_path) as source:
                    images.append(ImageOps.exif_transpose(source).convert("RGB"))
            inputs = self.processor(
                text=[text] * len(samples),
                images=images,
                padding=True,
                return_tensors="pt",
                size={"shortest_edge": self.args.min_pixels, "longest_edge": self.args.max_pixels},
            ).to(self.model.device)
            prompt_length = inputs["input_ids"].shape[1]
            processors = []
            if self.args.presence_penalty:
                processors.append(PresencePenaltyLogitsProcessor(self.args.presence_penalty, prompt_length))
            if self.args.do_sample:
                # Derive a fresh seed from the original complete batch. Resuming
                # earlier batches or changing model load order cannot shift it.
                self.torch.manual_seed(batch_seed(self.args.seed, samples))
            with self.torch.inference_mode():
                outputs = self.model.generate(**inputs, generation_config=self.generation_config,
                                              logits_processor=processors)
            # Slice by the padded input width, so the prompt never enters scoring.
            generated = outputs[:, inputs["input_ids"].shape[1]:]
            return self.processor.batch_decode(generated, skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
        finally:
            for image in images:
                image.close()


def prepare_run(args: argparse.Namespace, dataset: str, samples: list[Sample], split: str, prompt: str) -> dict:
    """Start afresh by default; reuse saved predictions only with --resume."""
    directory = args.output_dir / dataset
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "config.json"
    predictions_path = directory / "predictions.jsonl"
    decoding = build_decoding_config(args)
    angles = getattr(args, "prompt_style", "draft") == "angles"
    config = {
        "format_version": 4 if angles else (3 if len(decoding) > 1 else 2),
        "dataset": dataset,
        "split": split,
        "num_samples": len(samples),
        "samples_sha256": data_fingerprint(samples),
        "model": str(args.model),
        "prompt": prompt,
        "labels": list(LABELS[dataset]),
        "enable_thinking": False,
        "response_format": "angles_then_analysis_then_boxed_label" if angles else "draft_then_boxed_label",
        **decoding,
        "max_new_tokens": args.max_new_tokens,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "dtype": args.dtype,
        "device": args.device,
        "batch_size": args.batch_size,
    }
    if angles:
        config["prompt_style"] = "angles"
    if getattr(args, "num_shards", 1) > 1:
        config.update(num_shards=args.num_shards, shard_index=getattr(args, "shard_index", 0))
    if not args.resume:
        # Reset only this dataset's generated artifacts, never its whole directory.
        predictions_path.write_text("", encoding="utf-8")
        (directory / "metrics.json").unlink(missing_ok=True)
        write_json(config_path, config)
    elif config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old != config:
            differences = [key for key in config if old.get(key) != config[key]]
            raise ValueError(
                f"Cannot resume with changed settings/data: {differences}. "
                "Remove --resume to start over, or use a new --output-dir."
            )
    else:
        if predictions_path.exists():
            raise ValueError(
                f"{predictions_path} has no matching config.json; "
                "remove --resume to start over, or use a new --output-dir."
            )
        write_json(config_path, config)
    records = read_predictions(
        predictions_path, samples, LABELS[dataset],
        prediction_parser=parse_angles_prediction if angles else parse_draft_prediction,
    ) if args.resume else []
    seen = {record["sample_id"] for record in records}
    return {"directory": directory, "records": records, "pending": [s for s in samples if s.sample_id not in seen]}


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="Local Qwen3.5 model directory")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA, help="Parent of EMOSET, Emotion6 and EMO8")
    parser.add_argument(
        "--datasets", nargs="+", choices=list(LABELS), default=list(LABELS), help="Datasets to evaluate"
    )
    parser.add_argument("--emotion6-test-list", type=Path, help="Optional list of paths like anger/1.jpg")
    parser.add_argument(
        "--prompt-style", choices=("draft", "angles"), default="draft",
        help="draft: explanation inside draft; angles: draft outline, separate analysis, then boxed answer",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "qwen35_emotion",
        help="Output directory; selected dataset results are overwritten unless --resume is used",
    )
    parser.add_argument("--device", default="auto", help="auto selects one GPU if available; also cpu or cuda:N")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument(
        "--batch-size", "--batch_size", type=int, default=1, help="Number of images per generation batch"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512, help="Output budget for explanation and boxed answer per image"
    )
    parser.add_argument("--min-pixels", type=int, default=65536, help="Minimum resized image area")
    parser.add_argument("--max-pixels", type=int, default=1048576, help="Maximum resized image area")
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable random sampling instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (>0)")
    parser.add_argument("--top-p", type=float, default=0.8, help="Sampling nucleus probability")
    parser.add_argument("--top-k", type=int, default=20, help="Sampling top-k cutoff; 0 disables it")
    parser.add_argument("--min-p", type=float, default=0.0, help="Sampling relative probability cutoff")
    parser.add_argument("--presence-penalty", type=float, default=0.0,
                        help="Subtract this logit penalty once per token already generated; excludes prompt")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Transformers multiplicative repetition penalty; 1 disables it")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for deterministic original-batch sampling")
    parser.add_argument("--limit", type=int, help="Take first N samples per dataset for a smoke test")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of interleaved dataset shards")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard to evaluate; use a separate output directory")
    parser.add_argument("--resume", action="store_true", help="Resume a matching run from saved JSONL records")
    parser.add_argument(
        "--dry-run", action="store_true", help="Check all selected paths/labels without loading the model"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = argument_parser()
    args = parser.parse_args(argv)
    for key in ("batch_size", "max_new_tokens", "min_pixels", "max_pixels", "limit", "num_shards"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("Require 0 <= --shard-index < --num-shards")
    if args.min_pixels > args.max_pixels or args.min_pixels < 1024:
        parser.error("Require 1024 <= --min-pixels <= --max-pixels")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("--datasets must not contain duplicates")
    try:
        build_decoding_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    args.model = args.model.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if not args.model.is_dir():
        parser.error(f"Model directory does not exist: {args.model}")
    template = EMOTION_ANALYSIS_ANGLES_PROMPT if args.prompt_style == "angles" else DEEPSEEK_EMOTION_PROMPT
    prompts = {
        dataset: template.format(labels=", ".join(LABELS[dataset]))
        for dataset in args.datasets
    }

    datasets = {}
    for dataset in args.datasets:
        if dataset == "emoset":
            samples = load_emoset(args.data_root / "EMOSET")
            split = "official_test"
        elif dataset == "emotion6":
            samples = load_emotion6(args.data_root / "Emotion6", args.emotion6_test_list)
            split = "provided_test_list" if args.emotion6_test_list else "all_images_no_official_test_split"
        else:
            samples = load_emo8(args.data_root / "EMO8")
            split = "all_train_val_test_as_test"
        total = len(samples)
        if args.limit:
            samples = samples[:args.limit]
        samples = samples[args.shard_index::args.num_shards]
        if not samples:
            raise ValueError(f"No images selected for {dataset} (shard {args.shard_index}/{args.num_shards})")
        print(json.dumps({"dataset": dataset, "split": split, "available": total, "selected": len(samples),
                          "class_counts": dict(Counter(s.label for s in samples))}, ensure_ascii=False), flush=True)
        datasets[dataset] = (samples, split)
    if args.dry_run:
        return 0

    from tqdm.auto import tqdm

    if not args.resume:
        # The previous summary no longer describes the new run, even if it fails.
        (args.output_dir / "summary.json").unlink(missing_ok=True)
    runs = {
        name: prepare_run(args, name, samples, split, prompts[name])
        for name, (samples, split) in datasets.items()
    }
    predictor = QwenPredictor(args) if any(run["pending"] for run in runs.values()) else None
    summary = {}
    for dataset, run in runs.items():
        samples, split = datasets[dataset]
        labels = LABELS[dataset]
        records, pending = run["records"], run["pending"]
        num_correct = sum(record["correct"] for record in records)
        with (
            (run["directory"] / "predictions.jsonl").open("a", encoding="utf-8", buffering=1) as handle,
            tqdm(
                total=len(samples),
                initial=len(records),
                desc=dataset,
                unit="image",
                dynamic_ncols=True,
                postfix={"accuracy": f"{num_correct / len(records):.4f}" if records else "N/A"},
            ) as progress,
        ):
            seen = {record["sample_id"] for record in records}
            # Keep original batch membership on resume, including a batch whose
            # JSONL append was interrupted. Re-generate it and append only missing
            # records, preserving the batch seed and padding seen by the model.
            for offset in range(0, len(samples), args.batch_size):
                batch = samples[offset:offset + args.batch_size]
                if all(sample.sample_id in seen for sample in batch):
                    continue
                raw_outputs = predictor.predict(batch, prompt=prompts[dataset])
                if len(raw_outputs) != len(batch):
                    raise RuntimeError("The model did not return exactly one answer per image")
                added = 0
                for sample, raw in zip(batch, raw_outputs, strict=True):
                    if sample.sample_id in seen:
                        continue
                    if args.prompt_style == "angles":
                        draft, analysis, prediction = parse_angles_response(raw, labels)
                    else:
                        draft, prediction = parse_draft_response(raw, labels)
                    record = {"dataset": dataset, "sample_id": sample.sample_id,
                              "image_path": str(sample.image_path), "label": sample.label,
                              "prediction": prediction, "draft": draft, "raw_output": raw,
                              "correct": prediction == sample.label}
                    if args.prompt_style == "angles":
                        record["analysis"] = analysis
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records.append(record)
                    seen.add(sample.sample_id)
                    added += 1
                    num_correct += record["correct"]
                progress.set_postfix(accuracy=f"{num_correct / len(records):.4f}", refresh=False)
                progress.update(added)
        metrics = {"dataset": dataset, "split": split, **classification_metrics(records, labels)}
        write_json(run["directory"] / "metrics.json", metrics)
        summary[dataset] = metrics
        print(f"[{dataset}] accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
              f"invalid={metrics['num_invalid']}/{metrics['num_samples']}", flush=True)
    write_json(args.output_dir / "summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
