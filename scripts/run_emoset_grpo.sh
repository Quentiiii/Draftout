#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python}"
model_path="${MODEL_PATH:-/data/xiyan.zhao/work/model/Qwen3.5-2B}"
dataset_root="${EMOSET_ROOT:-/data/xiyan.zhao/work/dataset/EMOSET}"
data_dir="${DATA_DIR:-$project_dir/outputs/emoset_grpo/data}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-1,2}"

batch_size="${BATCH_SIZE:-16}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-1}"
rollout_n="${ROLLOUT_N:-4}"

export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$project_dir/outputs/emoset_grpo/cache/datasets}"

mode=train
case "${1-}" in
  --prepare-only) mode=prepare; shift ;;
  --check) mode=check; shift ;;
  --smoke) mode=smoke; shift ;;
  --help|-h)
    cat <<'EOF'
用法：bash scripts/run_emoset_grpo.sh [--prepare-only|--check|--smoke] [Hydra overrides...]
  默认：准备固定 2,000 条训练数据，从最新 checkpoint 续训到总共 10 epochs。
  当前 global_step_250 已完成 1 epoch；默认 batch 下继续到 step 2500，再跑 9 epochs。
  没有 checkpoint 时从头训练；总轮数由 trainer.total_epochs 控制（不是追加轮数）。
  --prepare-only：只准备 train/val parquet 和样本清单。
  --check：CPU 检查配置、奖励函数、数据和真实多模态处理器，不加载模型权重。
  --smoke：完整 2k 数据上执行 1 个训练 step，不跑验证；保存到独立 smoke 目录。

训练参数（也可直接修改脚本顶部）：
  BATCH_SIZE=16                   单卡一次前向/反向的回答数
  GRADIENT_ACCUMULATION_STEPS=1    单卡梯度累积次数；1 表示不累积
  ROLLOUT_N=4                     每图采样回答数
  每步图片数 = BATCH_SIZE × 累积次数 × GPU 数 / ROLLOUT_N。
  结果须为整数且能整除 2,000，避免丢弃最后一批数据。
  batch=1 时自动启用单样本去 padding，其余 batch 保留独立批次维度。

其他环境变量：PYTHON_BIN、MODEL_PATH、EMOSET_ROOT、DATA_DIR、CUDA_VISIBLE_DEVICES。
默认 vision-opd、Qwen3.5-2B、GPU 0,1。GPU 数由 CUDA_VISIBLE_DEVICES 自动计算。
批次、累积次数及 GPU 数请通过以上变量设置；其他 Hydra 参数可直接追加。
例如：
  CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_emoset_grpo.sh
  BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/run_emoset_grpo.sh
  BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 bash scripts/run_emoset_grpo.sh
  MODEL_PATH=/data/xiyan.zhao/work/model/Qwen3.5-2B bash scripts/run_emoset_grpo.sh
  bash scripts/run_emoset_grpo.sh trainer.total_epochs=10
  bash scripts/run_emoset_grpo.sh trainer.resume_mode=disable trainer.experiment_name=emoset_grpo_2k_new
EOF
    exit 0 ;;
esac

batch_overrides=()
if [[ "$mode" != prepare ]]; then
  for parameter in batch_size gradient_accumulation_steps rollout_n; do
    if [[ ! "${!parameter}" =~ ^[1-9][0-9]*$ ]]; then
      echo "${parameter^^} 必须是正整数，当前为：${!parameter}" >&2
      exit 1
    fi
    value="${!parameter}"
    if [[ ${#value} -gt 19 || ( ${#value} -eq 19 && "$value" > "9223372036854775807" ) ]]; then
      echo "${parameter^^} 超出脚本整数范围：$value" >&2
      exit 1
    fi
  done
  if (( rollout_n < 2 )); then
    echo "ROLLOUT_N 必须至少为 2，才能计算 GRPO 组内优势。" >&2
    exit 1
  fi
  if [[ -z "$CUDA_VISIBLE_DEVICES" || "$CUDA_VISIBLE_DEVICES" == "-1" ||
        "$CUDA_VISIBLE_DEVICES" == ,* || "$CUDA_VISIBLE_DEVICES" == *, ||
        "$CUDA_VISIBLE_DEVICES" == *,,* ]]; then
    echo "请通过 CUDA_VISIBLE_DEVICES 指定用于训练的 GPU，例如 0,1；--check 也按此计算 batch。" >&2
    exit 1
  fi
  IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
  n_gpus="${#gpu_ids[@]}"

  # These fields are derived together; overriding only one would silently
  # change the requested accumulation count or the loader's actual batch.
  for argument in "$@"; do
    key="${argument%%=*}"
    key="${key#++}"
    key="${key#+}"
    key="${key#\~}"
    case "$key" in
      data.train_batch_size|data.gen_batch_size|\
      actor_rollout_ref.actor.ppo_mini_batch_size|\
      actor_rollout_ref.actor.ppo_micro_batch_size|\
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu|\
      actor_rollout_ref.rollout.log_prob_micro_batch_size|\
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu|\
      actor_rollout_ref.rollout.n|actor_rollout_ref.model.use_remove_padding|\
      actor_rollout_ref.actor.use_dynamic_bsz|actor_rollout_ref.rollout.log_prob_use_dynamic_bsz|\
      actor_rollout_ref.actor.ulysses_sequence_parallel_size|\
      actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size|\
      trainer.n_gpus_per_node|trainer.nnodes)
        echo "$key 由脚本统一计算；请设置 BATCH_SIZE、GRADIENT_ACCUMULATION_STEPS、ROLLOUT_N 或 CUDA_VISIBLE_DEVICES。" >&2
        exit 1 ;;
    esac
  done
  if (( gradient_accumulation_steps > 9223372036854775807 / batch_size ||
        n_gpus > 9223372036854775807 / batch_size / gradient_accumulation_steps )); then
    echo "BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS × GPU 数超出脚本整数范围。" >&2
    exit 1
  fi
  response_batch_size=$((batch_size * gradient_accumulation_steps * n_gpus))
  if (( response_batch_size <= 0 || response_batch_size % rollout_n != 0 )); then
    echo "单卡 batch × 累积次数 × GPU 数必须能被 ROLLOUT_N 整除：$batch_size × $gradient_accumulation_steps × $n_gpus / $rollout_n。" >&2
    exit 1
  fi
  prompt_batch_size=$((response_batch_size / rollout_n))
  if (( prompt_batch_size > 2000 || 2000 % prompt_batch_size != 0 )); then
    echo "当前每步 $prompt_batch_size 张图，不能整除 2,000；请调整 BATCH_SIZE 或 GRADIENT_ACCUMULATION_STEPS。" >&2
    echo "例如两卡可用 BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4；四卡可用 BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1。" >&2
    exit 1
  fi
  remove_padding=false
  if (( batch_size == 1 )); then
    remove_padding=true
  fi
  batch_overrides=(
    "trainer.nnodes=1"
    "trainer.n_gpus_per_node=$n_gpus"
    "data.train_batch_size=$prompt_batch_size"
    "actor_rollout_ref.actor.ppo_mini_batch_size=$prompt_batch_size"
    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$batch_size"
    "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$batch_size"
    "actor_rollout_ref.rollout.n=$rollout_n"
    "actor_rollout_ref.model.use_remove_padding=$remove_padding"
  )
  echo "Batch 设置：单卡 $batch_size 条回答，累积 $gradient_accumulation_steps 次，$n_gpus 张 GPU；每步 $prompt_batch_size 张图 / $response_batch_size 条回答。"
fi

if [[ ! -x "$python_bin" ]]; then
  echo "Python 不可执行：$python_bin。请设置 PYTHON_BIN 为 vision-opd 的解释器。" >&2
  exit 1
fi
cd "$project_dir"
"$python_bin" -m emotion_grpo.prepare_data \
  --dataset-root "$dataset_root" --output-dir "$data_dir" \
  --train-size 2000 --val-size 256 --seed 42
if [[ "$mode" == prepare ]]; then
  exit 0
fi

overrides=(
  "project_root=$project_dir"
  "data_dir=$data_dir"
  "actor_rollout_ref.model.path=$model_path"
  "${batch_overrides[@]}"
)
if [[ "$mode" == check ]]; then
  overrides+=("dry_run=true")
elif [[ "$mode" == smoke ]]; then
  overrides+=(
    "trainer.experiment_name=emoset_grpo_2k_smoke"
    "trainer.resume_mode=disable"
    "trainer.total_training_steps=1"
    "trainer.val_before_train=false"
    "trainer.test_freq=-1"
    "trainer.save_freq=1"
  )
fi
exec "$python_bin" -m emotion_grpo.train "${overrides[@]}" "$@"
