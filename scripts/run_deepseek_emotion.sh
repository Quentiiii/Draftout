#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python}"
cd "$project_dir"

# 与现有 Qwen 评测口径一致，共 28,626 张。
# EmoSet：官方 test 17,716；Emotion6：六类目录全量 1,980；
# EMO8：train + val + test 全量 8,930。
# 可改成 datasets=(emoset)，或命令行传 --datasets emotion6 emo8。
datasets=(emoset emotion6 emo8)
model_name="deepseek-flash"
workers=4
max_tokens=512
max_image_side=1024
output_dir="outputs/deepseek_v41_flash_emotion"

# 优先使用环境变量；也可复制 .deepseek.env.example 为 .deepseek.env 并填写 key。
if [[ -z "${DEEPSEEK_API_KEY:-}" && -f .deepseek.env ]]; then
  set -a
  source .deepseek.env
  set +a
fi

# 保留旧单 EmoSet 子集命令：--test-list 且未显式 --datasets 时，
# Python 沿用直接指向 EMOSET 的 data-root 和平铺输出，支持旧 2k 结果续跑。
has_test_list=false
has_datasets=false
for argument in "$@"; do
  case "$argument" in
    --test-list|--test-list=*) has_test_list=true ;;
    --datasets|--datasets=*) has_datasets=true ;;
  esac
done
run_options=()
if [[ "$has_test_list" != true || "$has_datasets" == true ]]; then
  run_options=(--datasets "${datasets[@]}" --output-dir "$output_dir")
fi

exec "$python_bin" -m emotion_eval.deepseek \
  "${run_options[@]}" \
  --model "$model_name" \
  --workers "$workers" \
  --max-tokens "$max_tokens" \
  --max-image-side "$max_image_side" \
  "$@"
