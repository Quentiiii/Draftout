#!/usr/bin/env bash
set -euo pipefail

# 默认使用物理 GPU 0；可在命令前设置 CUDA_VISIBLE_DEVICES 覆盖，空值表示隐藏全部 GPU。
# 例如 CUDA_VISIBLE_DEVICES=1 bash scripts/run_emotion_test.sh --device cuda:0
# 此时 cuda:0 对应可见的第一张卡，即物理 GPU 1。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

# 模型路径：默认 2B；跑 4B 时改成 /data/xiyan.zhao/work/model/Qwen3.5-4B。
# 命令行 --model /path/to/model 可以覆盖此处设置。
model_path="/data/xiyan.zhao/work/model/Qwen3.5-2B"

# 修改此数组选择数据集（小写名称；多个名称之间用空格分隔）：
# datasets=(emoset)          只跑 EmoSet 官方测试集。
# datasets=(emotion6)        只跑 Emotion6 全部 1,980 张图片。
# datasets=(emoset emotion6) 两个数据集都跑（默认）。
datasets=(emoset emotion6 emo8)

batch_size=64
# 三个数据集统一使用 BF16 加载模型。
dtype=bfloat16

python_bin="${PYTHON_BIN:-/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python}"
cd "$project_dir"
exec "$python_bin" -m emotion_eval.infer \
  --model "$model_path" \
  --datasets "${datasets[@]}" \
  --batch-size "$batch_size" \
  --dtype "$dtype" \
  "$@"
