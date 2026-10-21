# Draftout

Local development workspace for `verl`, copied from Vision-OPD. The existing
modifications inside `verl/` are preserved; this is not a fresh upstream checkout.

## Development

The existing conda environment is `vision-opd`. It uses the source in this directory
through an editable installation:

```bash
conda activate vision-opd
cd /data/xiyan.zhao/work/Draftout
python -m pip install --no-deps --no-build-isolation -e .
```

Changes to Python source are used by newly started processes. Restart running
Python processes or notebook kernels after editing code.

To verify the installed source location from outside the repository:

```bash
cd /tmp
python -c "import verl; print(verl.__file__)"
```

The expected path is `/data/xiyan.zhao/work/Draftout/verl/__init__.py`.
`requirements.txt` retains the dependency snapshot inherited from Vision-OPD.

## Checkpoint merging

The retained helper invokes `verl.model_merger`. Supply a checkpoint directory
containing an `actor/` subdirectory:

```bash
bash /data/xiyan.zhao/work/Draftout/scripts/merge_checkpoint.sh /path/to/checkpoint
```

The helper replaces existing top-level files in the checkpoint directory with
merged model files and preserves the `actor/` shards.

## License

Apache-2.0; see `LICENSE` and the notices in individual source files.

## EmoSet / Emotion6 / EMO8 纯推理评估

入口为 `emotion_eval/infer.py`，默认加载本地
`/data/xiyan.zhao/work/model/Qwen3.5-2B`，只执行图像分类推理和评分。
默认关闭 thinking、贪心生成，不训练、不更新权重。模型调用使用
[Transformers 的 Qwen3.5 多模态接口](https://huggingface.co/docs/transformers/v5.5.0/en/model_doc/qwen3_5)。
三个数据集统一以 BF16 加载模型：Python 入口默认 `--dtype bfloat16`，启动脚本也显式传入该设置。

- **EmoSet**：读取 `dataset/EMOSET/test.json`，共 **17,716** 张；
  真值来自每张图片对应的 `annotation/*.json` 的 `emotion`，并核对测试清单中的类别。
- **Emotion6**：按已确认的范围，评估 `dataset/Emotion6/Emotion6/images/` 下全部 **1,980** 张；
  真值使用图片父目录名：`anger, disgust, fear, joy, sadness, surprise`。
  本地没有独立测试划分，因此结果标记为全量评估；不使用 `ground_truth.txt` 的情绪分布。
- **EMO8**：将 `dataset/EMO8/train/`（7,141 张）、`val/`（889 张）、`test/`（900 张）
  合并为评估集，共 **8,930** 张，全部只用于推理。真值来自类别目录名，其中 `sad` 映射为
  `Sadness`，其余七类与 EmoSet 同名。结果中的 split 为 `all_train_val_test_as_test`；
  样本 ID 保留 `train/`、`val/`、`test/` 前缀，避免不同划分下同名文件相互覆盖。
- 模型只接收 RGB 图片和固定分类提示词；图片文件名、路径、真值不会进入提示词。

类别统一使用 `emotion_eval/data.py` 中 `LABELS` 的写法，目前为首字母大写。
读取标注、目录类别及解析模型回答时忽略大小写，例如 `awe`、`Awe`、`AWE` 都映射为 `Awe`；
图片路径和文件名保持原样。Prompt 要求按候选列表的大小写输出，评分也兼容大小写差异。

启动脚本默认使用已有的 `vision-opd` 解释器（当前为 Python 3.12、
PyTorch 2.10.0、Transformers 5.5.0、torchvision 0.25.0、Pillow 12.0.0、Accelerate 1.12.0）。
无需安装本项目或训练依赖；默认 base Python 的 Transformers 版本不支持此模型。
可用 `PYTHON_BIN=/path/to/python` 覆盖解释器。模型和处理器均只从本地加载。

在启动脚本中修改 `model_path` 可以选择模型，默认是
`/data/xiyan.zhao/work/model/Qwen3.5-2B`；运行 4B 时改为
`/data/xiyan.zhao/work/model/Qwen3.5-4B`。也可用命令行 `--model` 覆盖，例如：

```bash
bash scripts/run_emotion_baseline.sh \
  --model /data/xiyan.zhao/work/model/Qwen3.5-4B \
  --output-dir outputs/qwen35_4b_emotion
```

这里指定独立输出目录可保留 2B 的结果；不指定时沿用默认目录并覆盖所选数据集的旧结果。

在 `scripts/run_emotion_baseline.sh` 中修改 `datasets=(...)` 即可选择要运行的数据集：

| 脚本参数 | 评估范围 |
| --- | --- |
| `datasets=(emoset)` | 只运行 EmoSet 官方测试集，17,716 张 |
| `datasets=(emotion6)` | 只运行 Emotion6 全部 1,980 张 |
| `datasets=(emo8)` | 只运行 EMO8 全部 8,930 张，合并 train、val、test |
| `datasets=(emoset emotion6)` | 只运行 EmoSet 和 Emotion6 |
| `datasets=(emoset emotion6 emo8)` | 三个数据集都运行（默认） |

也可通过命令行 `--datasets emoset`、`--datasets emotion6`、`--datasets emo8` 或
`--datasets emoset emotion6 emo8` 覆盖脚本中的选择，也可以任意组合。

每批处理的图片数由脚本中的 `batch_size` 控制（当前为 `64`），可直接修改这个变量，
也可用命令行 `--batch-size 8` 覆盖；`--batch_size 8` 是等价写法。
例如：

```bash
bash scripts/run_emotion_baseline.sh --batch-size 8 --output-dir outputs/emotion_bs8
```

批量大小需按可用显存调整。默认从头运行并覆盖所选数据集的旧结果；如需保留旧结果，
请使用新的 `--output-dir`。显式使用 `--resume` 时，`batch_size` 必须与原运行一致。

脚本默认设置 `CUDA_VISIBLE_DEVICES=0`，可直接编辑这行的默认卡号，
也可在启动命令前设置环境变量覆盖。`--device cuda:0` 表示可见 GPU 中的第一张；
例如 `CUDA_VISIBLE_DEVICES=1` 时，`cuda:0` 对应物理 GPU 1。

分类提示词独立保存在 `emotion_eval/prompts.py` 的 `EMOTION_PROMPT` 字符串中，可直接修改。
`{labels}` 是唯一会被自动替换的占位符，请保留它；程序按当前数据集填入合法类别列表。
推理代码导入该字符串，填入类别后作为 `prompt` 参数传给 `predictor.predict()`。
修改提示词后直接运行会从头覆盖旧结果；如需保留旧结果，请使用新的 `--output-dir`。
`--resume` 要求提示词与原运行一致。

```bash
cd /data/xiyan.zhao/work/Draftout

# 仅检查三个数据集的路径、标注和类别数量，不加载模型、不修改输出
bash scripts/run_emotion_baseline.sh --dry-run

# GPU 小样本试跑：每个数据集前 5 张，使用独立输出目录
CUDA_VISIBLE_DEVICES=0 bash scripts/run_emotion_baseline.sh \
  --device cuda:0 --limit 5 --output-dir outputs/emotion_smoke

# 完整评估三个数据集，从头运行并覆盖旧结果
CUDA_VISIBLE_DEVICES=0 bash scripts/run_emotion_baseline.sh \
  --device cuda:0 --batch-size 1

# 仅评估 Emotion6；改成 --datasets emoset 即只评估 EmoSet
bash scripts/run_emotion_baseline.sh --datasets emotion6 --device cuda:0

# 仅评估 EMO8：train、val、test 全部作为测试集
bash scripts/run_emotion_baseline.sh --datasets emo8 --device cuda:0

# 中断后使用相同参数续跑，跳过已保存的图片
CUDA_VISIBLE_DEVICES=0 bash scripts/run_emotion_baseline.sh \
  --device cuda:0 --batch-size 1 --resume
```

不使用启动脚本时，激活 `vision-opd` 后在项目根目录运行
`python -m emotion_eval.infer ...`。默认 `--device auto` 选择一个可用 GPU，
没有 GPU 则使用 CPU；全量 CPU 推理会很慢。CPU 小样本验证可用：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/run_emotion_baseline.sh \
  --device cpu --limit 2 --batch-size 2 --max-pixels 65536 \
  --output-dir outputs/emotion_cpu_smoke
```

推理时每个数据集显示一个 `tqdm` 进度条，包含已完成图片数、总图片数、处理速度、
预计剩余时间和累计准确率。每完成一批图片后刷新；使用 `--resume` 时从已保存的数量继续显示。

默认输出目录是项目下的 `outputs/qwen35_emotion/`，每个数据集分别保存：

- `predictions.jsonl`：逐图路径、真值、预测类别、原始模型输出和是否正确，每条立即写出。
- `metrics.json`：Accuracy、Macro-F1、每类 precision/recall/F1/support、混淆矩阵。
- `config.json`：模型路径、提示词、推理参数及样本清单摘要，用于核对断点续跑。

所选数据集完成后额外写出 `summary.json`。指标范围为 0–1；无法解析为单个合法类别的
输出记为 `prediction: null` 并计错，不从分母剔除。Macro-F1 始终覆盖数据集全部类别，
无样本类别记为 0，所以 `--limit` 仅适合验证流程，不能当作完整测试指标。
混淆矩阵行为真值、列为预测，最后一列是无效输出。

直接运行 Python 入口时，`--batch-size` 默认为 1；启动脚本使用其 `batch_size` 变量（当前为 `64`）。
`--max-pixels` 默认为 1,048,576，
`--min-pixels` 默认为 65,536，均指缩放后的图片像素面积。
`--max-new-tokens` 默认为 16。完整参数见 `bash scripts/run_emotion_baseline.sh --help`。
如另有 Emotion6 测试划分，可传 `--emotion6-test-list /path/to/test.txt`，
每行是相对 `images/` 的路径，例如 `anger/1.jpg`。

默认从头运行，覆盖所选数据集的 `config.json`、`predictions.jsonl` 和 `metrics.json`，
并更新 `summary.json`。如需保留旧结果，请使用新的 `--output-dir`。`--dry-run` 不修改输出。
只有显式传入 `--resume` 才会续跑；续跑须保持数据清单、提示词和推理参数一致。
加载/推理错误会停止程序，已写出的预测可使用 `--resume` 续跑；
中断导致的最后一条不完整 JSONL 会在续跑时移除并重新推理。

数据加载、答案解析、指标和续跑逻辑可通过标准库测试验证：

```bash
python -m unittest discover -s tests -v
```

## DeepSeek-V4.1-Flash：EmoSet 测试集 2,000 张

入口为 `scripts/run_deepseek_emoset.sh`，Python 模块为 `emotion_eval.deepseek`。
模型参数默认使用官方 API 名称 `deepseek-flash`，对应 V4.1-Flash 系列；
这是服务端维护的别名，实际返回的模型名称会随结果保存。
模型名称参考 [DeepSeek 官方发布说明](https://api-docs.deepseek.com/news/news260910/)，
图像请求格式参考 [Vision 文档](https://api-docs.deepseek.com/guides/vision/)。
脚本设置 `temperature=0`，使用 `emotion_eval/prompts.py` 中的 `DEEPSEEK_EMOTION_PROMPT`：
先用 2–4 句话解释图片中的视觉依据，将解释放在 `<draft>...</draft>` 内，
然后以 `\boxed{类别}` 给出唯一答案，例如：

```text
<draft>
The dark, threatening scene and tense posture convey a sense of danger.
</draft>
\boxed{Fear}
```

解释和答案一起生成在 API 的 `content` 中并保存；API 的独立 thinking 模式保持关闭，
相关参数见 [Thinking Mode 文档](https://api-docs.deepseek.com/guides/thinking_mode/)。
评分仅使用 `</draft>` 后的唯一 `\boxed{}` 类别，解释中提到的其他情绪不参与评分。

默认使用项目中的 `splits/emoset_test_2000.json`，从此前
`Qwen3-VL/splits/emoset_test_2000.json` 原样复制，保持实验样本一致。
该清单来自官方 `test.json` 的 17,716 张测试图片，按原始类别比例分层抽样，
seed 为 `20260612`，包含 2,000 张互不重复的图片，不包含训练样本。
运行时检查清单属于官方测试集、图片与 annotation 存在、annotation 的 emotion 与清单一致。
这里的测试子集与下方 GRPO 训练用的 2,000 张样本相互独立。

依赖仅为 **Python 3.10+ 和 Pillow**，无需本地模型或 GPU。
启动脚本默认使用已有的 `vision-opd` Python，可通过 `PYTHON_BIN=/path/to/python` 覆盖。
API key 通过环境变量 `DEEPSEEK_API_KEY`，或启动脚本读取的项目根目录 `.deepseek.env` 提供；
该本地文件已加入 Git 忽略规则。首次使用：

```bash
cd /data/xiyan.zhao/work/Draftout
cp .deepseek.env.example .deepseek.env
# 编辑 .deepseek.env，填入自己的 DEEPSEEK_API_KEY。

# 不需要 API key：校验数据与图片解码，不请求 API、不写输出文件。
bash scripts/run_deepseek_emoset.sh --dry-run

# 填好 key 后，先用清单前 2 张检查真实 API；结果放在独立目录。
bash scripts/run_deepseek_emoset.sh \
  --limit 2 --output-dir outputs/deepseek_emoset_smoke

# 完整运行清单中的 2,000 张。
bash scripts/run_deepseek_emoset.sh

# 中断或 API 错误后，在同一输出目录继续。
bash scripts/run_deepseek_emoset.sh --resume
```

仅有 API key 配置不会自动发起请求；执行非 `--dry-run` 命令才会调用 API。
若使用其他 Python，可先执行 `/path/to/python -m pip install Pillow`，再运行
`PYTHON_BIN=/path/to/python bash scripts/run_deepseek_emoset.sh`。
也可在项目根目录直接执行 `python -m emotion_eval.deepseek`，此时自行设置环境变量
`DEEPSEEK_API_KEY`，不会通过启动脚本加载 `.deepseek.env`。

常用参数如下，完整选项见 `bash scripts/run_deepseek_emoset.sh --help`：

| 参数 | 默认值与用途 |
| --- | --- |
| `--data-root` | `/data/xiyan.zhao/work/dataset/EMOSET`，直接指向含 `test.json` 的目录 |
| `--test-list` | 项目内 `splits/emoset_test_2000.json`，可指定相同三列格式的测试子集清单 |
| `--limit` | 默认不截断；指定时仅运行清单前 N 张，用于检查流程 |
| `--workers` | `4`，并发请求数 |
| `--model` | `deepseek-flash` |
| `--base-url` | `https://api.deepseek.com` |
| `--max-tokens` | `512`，单张图片解释与最终答案的总生成 token 上限 |
| `--max-image-side` | `1024`，发送图片的最长边上限 |
| `--timeout` | `120`，单次请求超时秒数 |
| `--max-retries` | `3`，可重试错误的最大重试次数 |

发送前按 EXIF 方向转正图片，只缩小、不放大，转换为 RGB JPEG，质量设为 95，
移除图片元数据。通过 base64 发送图像内容，图片路径、文件名和真值标签不进入提示词。
`--dry-run` 会核对标签并解码所有选中图片，适合在付费请求前检查数据完整性。

默认输出目录为 `outputs/deepseek_v41_flash_emoset_2k/`：

- `config.json`：模型、提示词、请求参数、图像处理参数和数据指纹。
- `samples.json`：本次实际选择的样本清单。
- `predictions.jsonl`：逐图完整 `raw_output`、单独的 `draft` 解释、预测类别、真值与 token 使用量。
- `errors.jsonl`：失败请求记录，便于排查和续跑。
- `metrics.json`：完成状态、样本进度、Accuracy、Macro-F1、逐类指标、混淆矩阵和 token 使用量。

默认拒绝覆盖已有运行文件；重复实验使用新的 `--output-dir`，恢复已有实验使用 `--resume`。
续跑会校验数据及配置，并跳过已保存的成功 API 响应；旧版仅输出标签的实验不能与新提示词续跑混用，
可用 `--output-dir outputs/deepseek_v41_flash_emoset_2k_draft` 开始新实验。
成功返回但缺少非空 `<draft>...</draft>`、末尾唯一 `\boxed{合法类别}` 或含额外文字的输出
计错且不自动重试，完整原始回答仍会保存。请求错误达到重试上限后，程序等待当前并发批次结束、
保存已完成结果并退出；续跑会重新处理失败和未完成样本。
网络失败保持待处理状态，不充当模型分类错误。`metrics.json` 的 `complete` 为 `true`
时才是本次清单的完整评估，否则指标仅反映已完成部分；`--limit` 的小样本结果不代表 2,000 张性能。

## EmoSet 2k 纯 GRPO 训练

入口是 `scripts/run_emoset_grpo.sh`，默认直接调用已有的 `vision-opd` 解释器，
使用本仓库 `verl` 和本地 `/data/xiyan.zhao/work/model/Qwen3.5-4B`。这里的纯 GRPO
使用组内均值、标准差归一化的 advantage 和 clipped policy loss，不启用 OPD、
teacher、自蒸馏、critic、reward model、KL 或 entropy bonus。

训练奖励只有正确性：正确类别得 **1**，错误、空回答、多个类别或无法解析的输出得 **0**。
`emotion_grpo/reward.py` 复用现有评估的答案解析规则，忽略大小写，兼容简单引号、
`<answer>` 包装和 thinking 后的最终答案；不额外加格式奖励。提示词也复用
`emotion_eval/prompts.py`，默认关闭 thinking，最多生成 16 tokens。

数据由 `emotion_grpo/prepare_data.py` 准备：

- 只从官方 `EMOSET/train.json` 无放回抽取 **2,000** 张，8 类各 250 张，seed 固定为 42。
- 从官方 `val.json` 独立抽取 **256** 张，8 类各 32 张，用于训练期间验证。
  该验证结果是均衡子集准确率，不是完整测试集指标。
- 核对官方 train/val/test 清单互斥、选中图片和 annotation 存在、annotation emotion
  与清单标签一致、所选 train/val 图片路径及图片内容不重合；不使用测试样本训练。
- 输出 `train.parquet`、`val.parquet`、两份 `*_manifest.jsonl` 和带 SHA256 的
  `metadata.json`，默认放在 `outputs/emoset_grpo/data/`。同配置、同内容重跑保留文件；
  不同内容需另选目录或显式运行数据准备模块的 `--overwrite`。
- 图片通过多模态字段传入，文件名和真值不进入模型文本；像素面积限制为
  65,536–262,144，patch size 为 16。

```bash
cd /data/xiyan.zhao/work/Draftout

# 数据已经准备好；也可随时确定性重新检查/准备
bash scripts/run_emoset_grpo.sh --prepare-only

# CPU 检查真实 parquet、每类一张 train/val 图片、模板、位置编码和奖励函数
# 不加载模型权重，不创建 Ray/GPU worker
bash scripts/run_emoset_grpo.sh --check

# GPU 单步验证训练链路，输出到独立的 emoset_grpo_2k_smoke 目录
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_emoset_grpo.sh --smoke

# 正式训练；默认 GPU 0,1，1 epoch
CUDA_VISIBLE_DEVICES=0,1 bash scripts/run_emoset_grpo.sh

# 覆盖学习率、epochs；其余 Hydra 参数也可按相同方式追加
bash scripts/run_emoset_grpo.sh \
  actor_rollout_ref.actor.optim.lr=5e-7 trainer.total_epochs=5

# 显式断点续训；相同模型、数据和训练配置
bash scripts/run_emoset_grpo.sh trainer.resume_mode=auto
```

batch 和梯度累积在 `scripts/run_emoset_grpo.sh` 顶部集中设置：

```bash
batch_size="${BATCH_SIZE:-16}"                              # 单卡一次前向/反向的回答数
gradient_accumulation_steps="${GRADIENT_ACCUMULATION_STEPS:-2}"  # 累积次数，1 表示不累积
rollout_n="${ROLLOUT_N:-4}"                                 # 每图回答数
```

可以直接修改默认值，也可以在启动时通过环境变量覆盖：

```bash
# 单卡 batch 8，累积 4 次；默认两卡下每次更新仍处理 16 张图 / 64 条回答
BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/run_emoset_grpo.sh

# 单卡 batch 16，不做梯度累积；默认两卡下每次更新处理 8 张图 / 32 条回答
BATCH_SIZE=16 GRADIENT_ACCUMULATION_STEPS=1 bash scripts/run_emoset_grpo.sh

# 先做 CPU 配置及数据检查
BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=4 bash scripts/run_emoset_grpo.sh --check
```

脚本按 `每步图片数 = batch_size × gradient_accumulation_steps × GPU 数 / rollout_n`
自动换算 `data.train_batch_size`、`ppo_mini_batch_size` 和 actor/log-prob 的单卡 micro batch，
启动时打印最终数值。默认两卡、batch 16、累积 2 次，每步处理 16 张图 / 64 条回答，
每卡各处理 32 条回答后完成一次参数更新。
每步图片数必须为整数且能整除 2,000，否则提前报错，避免丢弃最后一批或改变累积次数。
例如两卡下 `batch=16, accumulation=4` 得到每步 32 张图，不能整除 2,000；
可用 `batch=8, accumulation=4` 保持每步 16 张图。

这些派生参数、节点数及 GPU 数不能再用额外 Hydra 参数覆盖，请使用上述变量和
`CUDA_VISIBLE_DEVICES`；学习率、epochs 等其他参数仍从
`verl/trainer/config/emoset_grpo.yaml` 读取，并可用 Hydra 参数覆盖。
学习率 `1e-6`，模型加载、训练前向和 vLLM rollout 均显式设为 BF16，使用 FSDP2。
vLLM 使用 `temperature=1`，验证使用单回答贪心解码。
默认每步 16 张图，每个 epoch 共 125 steps，训练 1 epoch；调整有效 batch 后 steps 相应变化。
每 25 steps 验证并保存，最多保留两个 actor checkpoint，日志默认输出到终端。
checkpoint 默认写到 `outputs/emoset_grpo/checkpoints/emoset_grpo_2k/`，同时保存
`resolved_config.yaml` 和验证回答。已有 checkpoint 时需要显式续训或换
`trainer.experiment_name`，避免覆盖已有训练结果。

启动脚本从 `CUDA_VISIBLE_DEVICES` 推导 GPU 数；也可覆盖 `PYTHON_BIN`、`MODEL_PATH`、
`EMOSET_ROOT`、`DATA_DIR`。单卡可设置 `CUDA_VISIBLE_DEVICES=0`，实际能否容纳模型与
优化器取决于显存；默认启用了参数和优化器 CPU offload。

当前 `vision-opd` 的 Qwen3.5 线性注意力不能安全地跨样本打包，因此 batch 16 使用
保留 padding 的独立批次维度（`use_remove_padding=false`），关闭动态 batch，SP=1。
启动时会检查每卡样本数能被 micro batch 整除，避免产生不完整的小批次。
更改 GPU 数时也要满足上述图片数整除约束，例如四卡可用 `batch=16, accumulation=1`，
八卡可用 `batch=4, accumulation=2`，均为每步 16 张图。
设置 `BATCH_SIZE=1` 时，脚本自动同时调整 actor/log-prob 两项 micro batch，
并启用单样本去 padding；例如两卡 `BATCH_SIZE=1 GRADIENT_ACCUMULATION_STEPS=32`。
还修复了继承代码在 vanilla policy update 中读取蒸馏参数、
请求额外蒸馏 logits 的问题，使其只在 VOPD 分支执行。vLLM 的显式关闭缓存参数也已
修复为传递对应的 `--no-...`，确保本配置的 `enable_prefix_caching=false` 真正生效。

新增数据、奖励和 actor 回归测试需要使用 `vision-opd`：

```bash
/data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python -m unittest discover -s tests -v
```
