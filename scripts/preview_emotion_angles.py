#!/usr/bin/env python3
"""Preview the analysis-angles prompt on a small balanced EmoSet test sample.

Example (vision-opd environment):
    CUDA_VISIBLE_DEVICES=1 python scripts/preview_emotion_angles.py

The draft contains an outline, followed by analysis outside the draft. Save raw
outputs without applying the older draft-then-boxed strict evaluation parser.
"""

import argparse
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from emotion_eval.data import LABELS, canonical_label, load_emoset
from emotion_eval.infer import QwenPredictor, data_fingerprint, write_json
from emotion_eval.prompts import EMOTION_ANALYSIS_ANGLES_PROMPT
from emotion_eval.sampling import build_decoding_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=PROJECT.parent / "model/Qwen3.5-4B")
    parser.add_argument("--data-root", type=Path, default=PROJECT.parent / "dataset/EMOSET")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--samples-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0", help="Index within CUDA_VISIBLE_DEVICES")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dry-run", action="store_true", help="Print selection/config without loading weights or writing results")
    args = parser.parse_args()
    if args.samples_per_class < 1 or args.max_new_tokens < 1:
        parser.error("samples-per-class and max-new-tokens must be positive")
    args.model = args.model.expanduser().resolve()
    if not args.model.is_dir():
        parser.error(f"Model directory does not exist: {args.model}")

    labels = LABELS["emoset"]
    official = load_emoset(args.data_root)
    rng = random.Random(args.seed)
    samples = []
    for label in labels:
        candidates = sorted((sample for sample in official if sample.label == label), key=lambda s:s.sample_id)
        if args.samples_per_class > len(candidates):
            parser.error(f"Not enough test samples for {label}")
        samples.extend(rng.sample(candidates, args.samples_per_class))
    prompt = EMOTION_ANALYSIS_ANGLES_PROMPT.format(labels=", ".join(labels))
    inference = argparse.Namespace(
        model=args.model, device=args.device, dtype="bfloat16", batch_size=1,
        max_new_tokens=args.max_new_tokens, min_pixels=65536, max_pixels=1048576,
        do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
        presence_penalty=1.5, repetition_penalty=1.0, seed=args.seed,
    )
    config = {
        "model":str(args.model), "dataset":"emoset", "split":"official_test",
        "prompt_name":"EMOTION_ANALYSIS_ANGLES_PROMPT", "prompt":prompt,
        "selection":"Sort official test IDs within each class, then random.Random(seed).sample per class, before inference.",
        "selection_seed":args.seed, "samples_per_class":args.samples_per_class,
        "num_samples":len(samples), "samples_sha256":data_fingerprint(samples),
        "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device":args.device, "dtype":inference.dtype, "batch_size":1,
        "enable_thinking":False, "max_new_tokens":inference.max_new_tokens,
        "min_pixels":inference.min_pixels, "max_pixels":inference.max_pixels,
        "decoding":build_decoding_config(inference),
        "scoring":"Preview only; boxed_prediction extracts one valid box. The previous strict draft parser is not used.",
    }
    manifest = [{"dataset":s.dataset, "sample_id":s.sample_id, "image_path":str(s.image_path), "label":s.label} for s in samples]
    if args.dry_run:
        print(json.dumps({"config":config,"samples":manifest},ensure_ascii=False,indent=2))
        return 0

    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    model_name = re.sub(r"[^a-z0-9]+", "_", args.model.name.lower().replace(".", "")).strip("_")
    directory = (args.output_dir or PROJECT / "outputs" / f"{model_name}_angles_preview_{now:%Y%m%d_%H%M%S}").expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    config.update(status="running", started_at=now.isoformat(), completed_samples=0)
    write_json(directory / "config.json", config)
    (directory / "samples.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in manifest),encoding="utf-8")
    print(f"Output directory: {directory}", flush=True)
    try:
        predictor = QwenPredictor(inference)
        with (directory / "predictions.jsonl").open("w",encoding="utf-8") as output, (directory / "examples.txt").open("w",encoding="utf-8") as readable:
            readable.write(f"Model: {args.model}\nPrompt:\n{prompt}\n\n")
            for index, (sample, metadata) in enumerate(zip(samples,manifest),start=1):
                raw = predictor.predict([sample],prompt)[0]
                boxes = re.findall(r"\\boxed\{([^{}]*)\}",raw)
                prediction = canonical_label(boxes[0],labels) if len(boxes)==1 and len(re.findall(r"\\boxed\b",raw))==1 else None
                record = {**metadata,"raw_output":raw,"boxed_prediction":prediction,"boxed_correct":prediction==sample.label}
                output.write(json.dumps(record,ensure_ascii=False)+"\n")
                output.flush()
                readable.write(f"===== Example {index} =====\nImage: {sample.image_path}\nGround truth: {sample.label}\nBoxed prediction: {prediction}\n\n{raw}\n\n")
                readable.flush()
                config["completed_samples"] = index
                write_json(directory / "config.json",config)
                print(f"[{index}/{len(samples)}] {sample.sample_id}: label={sample.label}, boxed={prediction}",flush=True)
        config.update(status="complete",finished_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat())
        write_json(directory / "config.json",config)
    except Exception as error:
        config.update(status="failed",error=str(error))
        write_json(directory / "config.json",config)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
