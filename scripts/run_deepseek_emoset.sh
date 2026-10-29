#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# 兼容旧入口；无参运行现在默认评测三个数据集的全部图片。
exec bash "$project_dir/scripts/run_deepseek_emotion.sh" "$@"
