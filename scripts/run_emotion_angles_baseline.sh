#!/usr/bin/env bash
# Qwen3.5-4B: draft angles, evidence analysis, then a boxed emotion label.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python}"
model_dir="${MODEL_DIR:-/data/xiyan.zhao/work/model/Qwen3.5-4B}"
result_dir="${OUTPUT_DIR:-$project_dir/outputs/qwen35_4b_emotion_angles_sampled_seed42}"
batch_size="${BATCH_SIZE:-32}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
num_shards="${#gpu_ids[@]}"
export LD_LIBRARY_PATH="/data/xiyan.zhao/anaconda3/envs/vision-opd/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
cd "$project_dir"
mkdir -p "$result_dir/logs"

record_state() {
  "$python_bin" - "$result_dir" "$model_dir" "$1" "$CUDA_VISIBLE_DEVICES" "$batch_size" "$num_shards" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root, model, state, devices, batch, shards = sys.argv[1:]
root = Path(root)
path = root / 'run_metadata.json'
data = json.loads(path.read_text()) if path.exists() else {}
now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
data.setdefault('started_at', now)
data.update(model=model, status=state, updated_at=now, cuda_visible_devices=devices,
            shards=int(shards), batch_size_per_gpu=int(batch), dtype='bfloat16',
            prompt_style='angles', max_new_tokens=1024,
            min_pixels=65536, max_pixels=1048576, enable_thinking=False,
            do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
            presence_penalty=1.5, repetition_penalty=1.0, seed=42,
            seed_policy='sha256_batch_sample_ids_v1', presence_penalty_scope='generated_tokens_only',
            expected_samples={'emoset': 17716, 'emotion6': 1980, 'emo8': 8930},
            reproduce='bash scripts/run_emotion_angles_baseline.sh')
if state == 'complete':
    summary = json.loads((root / 'summary.json').read_text())
    data.update(finished_at=now, total_evaluated=sum(m['num_samples'] for m in summary.values()),
                metrics={ds: {key:m[key] for key in ('num_samples','num_correct','num_invalid','accuracy','macro_f1')}
                         for ds,m in summary.items()},
                coverage_verified='Official IDs, order, fingerprints and parsed outputs validated by scripts/analyze_emotion_angles.py')
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
temporary.replace(path)
PY
}

pids=()
stop_run() {
  if (( ${#pids[@]} )); then kill "${pids[@]}" 2>/dev/null || true; fi
  record_state interrupted
  exit 130
}
trap stop_run INT TERM
record_state running
for ((shard=0; shard<num_shards; shard++)); do
  shard_dir="$result_dir/shard_$shard"
  if (( num_shards == 1 )); then shard_dir="$result_dir"; fi
  "$python_bin" -u -m emotion_eval.infer \
    --model "$model_dir" --datasets emoset emotion6 emo8 --prompt-style angles \
    --dtype bfloat16 --batch-size "$batch_size" --max-new-tokens 1024 \
    --min-pixels 65536 --max-pixels 1048576 \
    --do-sample --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
    --presence-penalty 1.5 --repetition-penalty 1 --seed 42 \
    --device "cuda:$shard" --num-shards "$num_shards" --shard-index "$shard" --resume \
    --output-dir "$shard_dir" > "$result_dir/logs/shard_$shard.log" 2>&1 &
  pids+=("$!")
done
printf '%s\n' "${pids[@]}" > "$result_dir/worker_pids.txt"
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
pids=()
if (( status != 0 )); then
  record_state failed
  echo "Inference failed; inspect $result_dir/logs and rerun with the same settings to resume." >&2
  exit "$status"
fi
if (( num_shards > 1 )); then
  if ! "$python_bin" scripts/aggregate_emotion_shards.py --input-dir "$result_dir" --num-shards "$num_shards"; then
    record_state aggregation_failed
    exit 1
  fi
fi
record_state validating
if ! "$python_bin" scripts/analyze_emotion_angles.py --input-dir "$result_dir"; then
  record_state validation_failed
  exit 1
fi
record_state complete
