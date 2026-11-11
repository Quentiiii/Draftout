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
  从头训练先做 20 步普通 GRPO；恢复 checkpoint 时直接继续 Draft GRPO，不重新预热。
  Draft 阶段每图采样 2 draft × 2 continuation。
  Draft GRPO 阶段使用 batch 全局收缩系数与分段 advantage，不做 group normalization。
  无有效 draft 时将整条输出作为后缀训练，并记录 fallback 次数。
  独立实验：outputs/emoset_grpo/checkpoints/emoset_draft_grpo_2b_2k_warmup20_fallback。
  --check：仅 CPU 预检，不加载模型权重。
  --smoke：独立目录中运行 21 步（20 步 warmup + 1 步 Draft GRPO），需要 GPU。
  训练输出与分支 ID：实验目录下 rollout/<step>.jsonl。
  完整终端日志：outputs/emoset_grpo/logs/train_draft_日期时间_PID.log。
  可用 DRAFT_TRAIN_LOG 指定日志文件；默认启用 HYDRA_FULL_ERROR=1。
  参数：draft_grpo.ema_beta=0.9 draft_grpo.shrink_min=0.1 draft_grpo.eps=1e-6。
  总轮数表示累计轮数；默认 draft_grpo.warmup_on_resume=false。
EOF
    exit 0 ;;
  --prepare-only|--check) mode=("$1"); shift ;;
  --smoke)
    mode=("$1"); shift
    smoke_overrides=(
      "trainer.experiment_name=emoset_draft_grpo_2b_2k_warmup20_fallback_smoke"
      "trainer.total_training_steps=21"
    ) ;;
esac

if [[ "${mode[0]-}" != --check && "${mode[0]-}" != --prepare-only ]]; then
  log_file="${DRAFT_TRAIN_LOG:-$project_dir/outputs/emoset_grpo/logs/train_draft_$(date +%Y%m%d_%H%M%S)_$$.log}"
  mkdir -p -- "$(dirname -- "$log_file")"
  export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
  export PYTHONUNBUFFERED=1
  exec > >(tee -a -- "$log_file") 2>&1
  echo "完整训练日志：$log_file"
fi

exec bash "$project_dir/scripts/run_emoset_grpo.sh" "${mode[@]}" \
  --config-name emoset_draft_grpo "${smoke_overrides[@]}" "$@"
