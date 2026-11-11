#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export BATCH_SIZE="${BATCH_SIZE:-8}"
mode=()
smoke_overrides=()
case "${1-}" in
  --help|-h)
    cat <<'EOF'
用法：bash scripts/run_emoset_draft_grpo.sh [--prepare-only|--check|--smoke] [Hydra overrides...]
  复用 run_emoset_grpo.sh 的模型、数据和 PPO 配置。
  默认 GPU 2,3，单卡 batch 8、梯度累计 2；每步 8 图 / 32 条完整回答。
  可通过 BATCH_SIZE 和 GRADIENT_ACCUMULATION_STEPS 覆盖默认值。
  每次启动（包括恢复）先做 10 步普通 GRPO，再做 2 draft × 2 continuation。
  Draft GRPO 阶段使用 batch 全局收缩系数与分段 advantage，不做 group normalization。
  独立实验：outputs/emoset_grpo/checkpoints/emoset_draft_grpo_2b_2k。
  --check：仅 CPU 预检，不加载模型权重。
  --smoke：独立目录中运行 11 步（10 步 warmup + 1 步 Draft GRPO），需要 GPU。
  训练输出与分支 ID：实验目录下 rollout/<step>.jsonl。
  参数：draft_grpo.ema_beta=0.9 draft_grpo.shrink_min=0.1 draft_grpo.eps=1e-6。
  保留每次启动固定 10 步 warmup；总轮数仍表示累计轮数。
EOF
    exit 0 ;;
  --prepare-only|--check) mode=("$1"); shift ;;
  --smoke)
    mode=("$1"); shift
    smoke_overrides=(
      "trainer.experiment_name=emoset_draft_grpo_2b_2k_smoke"
      "trainer.total_training_steps=11"
    ) ;;
esac

exec bash "$project_dir/scripts/run_emoset_grpo.sh" "${mode[@]}" \
  --config-name emoset_draft_grpo "${smoke_overrides[@]}" "$@"
