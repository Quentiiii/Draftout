#!/usr/bin/env bash
# Resume the two draft baselines with Qwen3.5's non-thinking VL sampling settings.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python}"
cd "$project_dir"
export CUDA_VISIBLE_DEVICES=0,3
export LD_LIBRARY_PATH="/data/xiyan.zhao/anaconda3/envs/vision-opd/lib:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

if (( $# == 0 )); then set -- 2B 4B; fi
for model_size in "$@"; do
  case "$model_size" in
    2B|4B) ;;
    *) echo "Usage: bash scripts/run_emotion_sampling_baselines.sh [2B] [4B]" >&2; exit 2 ;;
  esac
done

record_state() {
  "$python_bin" - "$result_dir" "$model_dir" "$1" <<'PY'
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
root, model, state = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
path = root / 'run_metadata.json'
data = json.loads(path.read_text()) if path.exists() else {}
now = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
data.setdefault('started_at', now)
data.update(model=model, status=state, updated_at=now, cuda_visible_devices='0,3',
            shards=2, batch_size_per_gpu=64, dtype='bfloat16', max_new_tokens=512,
            min_pixels=65536, max_pixels=1048576, enable_thinking=False,
            do_sample=True, temperature=0.7, top_p=0.8, top_k=20, min_p=0.0,
            presence_penalty=1.5, repetition_penalty=1.0, seed=42,
            seed_policy='sha256_batch_sample_ids_v1', presence_penalty_scope='generated_tokens_only',
            expected_samples={'emoset': 17716, 'emotion6': 1980, 'emo8': 8930},
            reproduce='bash scripts/run_emotion_sampling_baselines.sh')
if state == 'complete':
    summary = json.loads((root / 'summary.json').read_text())
    data.update(finished_at=now, total_evaluated=sum(m['num_samples'] for m in summary.values()),
                metrics={ds: {key:m[key] for key in ('num_samples','num_correct','num_invalid','accuracy','macro_f1')}
                         for ds,m in summary.items()},
                coverage_verified='All official IDs, order, data fingerprints and parsed outputs validated by aggregation')
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
temporary.replace(path)
PY
}

pids=()
trap 'if (( ${#pids[@]} )); then kill "${pids[@]}" 2>/dev/null || true; fi; exit 130' INT TERM
for model_size in "$@"; do
  model_dir="/data/xiyan.zhao/work/model/Qwen3.5-$model_size"
  result_dir="$project_dir/outputs/qwen35_${model_size,,}_emotion_draft_sampled_seed42"
  mkdir -p "$result_dir/logs"
  record_state running
  pids=()
  for shard in 0 1; do
    "$python_bin" -u -m emotion_eval.infer \
      --model "$model_dir" --datasets emoset emotion6 emo8 --dtype bfloat16 \
      --batch-size 64 --max-new-tokens 512 --min-pixels 65536 --max-pixels 1048576 \
      --do-sample --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
      --presence-penalty 1.5 --repetition-penalty 1 --seed 42 \
      --device "cuda:$shard" --num-shards 2 --shard-index "$shard" --resume \
      --output-dir "$result_dir/shard_$shard" > "$result_dir/logs/shard_$shard.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do wait "$pid" || status=1; done
  pids=()
  if (( status != 0 )); then
    record_state failed
    echo "$model_size failed; inspect $result_dir/logs and rerun to resume." >&2
    exit "$status"
  fi
  if ! "$python_bin" scripts/aggregate_emotion_shards.py --input-dir "$result_dir" --num-shards 2; then
    record_state aggregation_failed
    exit 1
  fi
  record_state complete
done
