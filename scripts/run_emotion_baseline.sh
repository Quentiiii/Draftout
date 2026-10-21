#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

model_path="/data/xiyan.zhao/work/model/Qwen3.5-2B"

# 修改此数组选择数据集（小写名称；多个名称之间用空格分隔）：
# datasets=(emoset)          只跑 EmoSet 官方测试集。
# datasets=(emotion6)        只跑 Emotion6 全部 1,980 张图片。
# datasets=(emoset emotion6) 两个数据集都跑（默认）。
datasets=(emoset emotion6 emo8)

batch_size=64
dtype=bfloat16

# just for fun

python_bin="${PYTHON_BIN:-/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python}"
cd "$project_dir"
exec "$python_bin" -m emotion_eval.infer \
  --model "$model_path" \
  --datasets "${datasets[@]}" \
  --batch-size "$batch_size" \
  --dtype "$dtype" \
  "$@"
