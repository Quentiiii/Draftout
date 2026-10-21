#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python}"
cd "$project_dir"

# 优先使用环境变量；也可复制 .deepseek.env.example 为 .deepseek.env 并填写 key。
if [[ -z "${DEEPSEEK_API_KEY:-}" && -f .deepseek.env ]]; then
  set -a
  source .deepseek.env
  set +a
fi

exec "$python_bin" -m emotion_eval.deepseek "$@"
