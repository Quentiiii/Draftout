#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python}"
model_path="${MODEL_PATH:-/data/xiyan.zhao/work/model/Qwen3.5-2B}"
dataset_root="${EMOSET_ROOT:-/data/xiyan.zhao/work/dataset/EMOSET}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-2,3}"

batch_size="${BATCH_SIZE:-8}"
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"
rollout_n="${ROLLOUT_N:-4}"

export PYTHONPATH="$project_dir${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/data/xiyan.zhao/anaconda3/envs/vision-opd/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$project_dir/outputs/emoset_grpo/cache/datasets}"

# The Draft wrapper chooses its prompt, data and trainer config together.
prompt_style=grpo
config_name=emoset_grpo
smoke_experiment=emoset_grpo_2b_8k_smoke
smoke_steps=1
config_arguments=()
if [[ "${1-}" == --draft ]]; then
  shift
  prompt_style=draft
  config_name=emoset_draft_grpo
  smoke_experiment=emoset_draft_grpo_2b_8k_warmup20_fallback_smoke
  smoke_steps=21
  config_arguments=(--config-name "$config_name")
fi
data_dir="${DATA_DIR:-$project_dir/outputs/emoset_grpo/data_${prompt_style}_8k_seed42}"

mode=train
case "${1-}" in
  --prepare-only) mode=prepare; shift ;;
  --check) mode=check; shift ;;
  --smoke) mode=smoke; shift ;;
  --help|-h)
    cat <<'EOF'
用法：bash scripts/run_emoset_grpo.sh [--prepare-only|--check|--smoke] [Hydra overrides...]
  默认：使用 GRPO_PROMPT，逐步分析并输出 boxed 答案；Qwen3.5-2B，8,000 张训练图，总共 2 epochs。
  数据目录：outputs/emoset_grpo/data_grpo_8k_seed42。
  当前实验：outputs/emoset_grpo/checkpoints/emoset_grpo_2b_8k。
  自动恢复当前实验的 checkpoint；没有 checkpoint 时从原版模型开始，总轮数表示累计轮数。
  --prepare-only：只准备 train/val parquet 和样本清单。
  --check：CPU 检查配置、奖励函数、数据和真实多模态处理器，不加载模型权重。
  --smoke：完整 8k 数据上执行 1 个训练 step，不跑验证；保存到独立 smoke 目录。

训练参数（也可直接修改脚本顶部）：
  BATCH_SIZE=8                    单卡一次前向/反向的回答数
  GRADIENT_ACCUMULATION_STEPS=2    单卡梯度累积次数；1 表示不累积
  ROLLOUT_N=4                     每图采样回答数
  每步图片数 = BATCH_SIZE × 累积次数 × GPU 数 / ROLLOUT_N。
  结果须为整数且能整除 8,000，避免丢弃最后一批数据。
  batch=1 时自动启用单样本去 padding，其余 batch 保留独立批次维度。

其他环境变量：PYTHON_BIN、MODEL_PATH、EMOSET_ROOT、DATA_DIR、CUDA_VISIBLE_DEVICES。
默认 vision-opd、Qwen3.5-2B、GPU 2,3。GPU 数由 CUDA_VISIBLE_DEVICES 自动计算。
输出最多 1,024 tokens；默认两卡每步仍是 8 张图，2 epochs 共 2,000 steps。
批次、累积次数及 GPU 数请通过以上变量设置；其他 Hydra 参数可直接追加。
例如：
  CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_emoset_grpo.sh --smoke
  CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_emoset_grpo.sh
  CUDA_VISIBLE_DEVICES=2 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/run_emoset_grpo.sh
  BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=16 bash scripts/run_emoset_grpo.sh
  MODEL_PATH=/data/xiyan.zhao/work/model/Qwen3.5-2B bash scripts/run_emoset_grpo.sh
  bash scripts/run_emoset_grpo.sh trainer.total_epochs=2
  bash scripts/run_emoset_grpo.sh trainer.resume_mode=disable trainer.experiment_name=emoset_grpo_2b_new
EOF
    exit 0 ;;
esac

# Consume matching Hydra config flags, but reject flags that would select a
# trainer recipe inconsistent with the data prompt chosen by this entry point.
hydra_arguments=()
while (( $# )); do
  case "$1" in
    --config-name)
      if [[ "${2-}" != "$config_name" ]]; then
        echo "此入口固定使用 $config_name；请使用对应的 run_emoset_grpo.sh 或 run_emoset_draft_grpo.sh。" >&2
        exit 1
      fi
      shift 2 ;;
    --config-name=*)
      if [[ "${1#*=}" != "$config_name" ]]; then
        echo "此入口固定使用 $config_name；请使用对应的 run_emoset_grpo.sh 或 run_emoset_draft_grpo.sh。" >&2
        exit 1
      fi
      shift ;;
    *) hydra_arguments+=("$1"); shift ;;
  esac
done
set -- "${hydra_arguments[@]}"

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
  if (( prompt_batch_size > 8000 || 8000 % prompt_batch_size != 0 )); then
    echo "当前每步 $prompt_batch_size 张图，不能整除 8,000；请调整 BATCH_SIZE 或 GRADIENT_ACCUMULATION_STEPS。" >&2
    echo "例如两卡可用 BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=8；四卡可用 BATCH_SIZE=2 GRADIENT_ACCUMULATION_STEPS=4。" >&2
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
  --train-size 8000 --val-size 256 --seed 42 --prompt-style "$prompt_style"
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
  # Batch dimensions above still describe the requested training GPUs.
  # Preflight itself must not initialize or use any GPU.
  export CUDA_VISIBLE_DEVICES=""
  overrides+=("dry_run=true")
elif [[ "$mode" == smoke ]]; then
  overrides+=(
    "trainer.experiment_name=$smoke_experiment"
    "trainer.resume_mode=disable"
    "trainer.total_training_steps=$smoke_steps"
    "trainer.val_before_train=false"
    "trainer.test_freq=-1"
    "trainer.save_freq=1"
  )
fi
exec "$python_bin" -m emotion_grpo.train "${config_arguments[@]}" "${overrides[@]}" "$@"
