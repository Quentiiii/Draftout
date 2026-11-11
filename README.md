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
`/data/xiyan.zhao/work/model/Qwen3.5-4B`；运行 2B 时改为
`/data/xiyan.zhao/work/model/Qwen3.5-2B`。也可用 `MODEL_PATH` 环境变量或命令行 `--model` 覆盖，例如：

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

Baseline 与 DeepSeek 共用 `emotion_eval/prompts.py` 中的 `DEEPSEEK_EMOTION_PROMPT`，可直接修改。
模型先在 `<draft>...</draft>` 中用 2–4 句话解释图像依据，再以 `\boxed{类别}` 给出最终答案。
三个数据集使用同一模板，各自填入对应的候选类别；EmoSet 的提示词与 DeepSeek 完全一致。
`{labels}` 是唯一会被自动替换的占位符，请保留它；程序按当前数据集填入合法类别列表。
模板中的 `\boxed{{LABEL}}` 使用双大括号转义，填入类别列表后变为 `\boxed{LABEL}`。
推理代码导入该字符串，填入类别后作为 `prompt` 参数传给 `predictor.predict()`。
修改提示词后直接运行会从头覆盖旧结果；如需保留旧结果，请使用新的 `--output-dir`。
`--resume` 要求提示词与原运行一致，不能续跑旧版仅输出标签的实验。
新实验可运行 `bash scripts/run_emotion_baseline.sh --output-dir outputs/qwen35_emotion_draft_sampled`。

启动脚本默认采用非 thinking 图像任务的采样参数：`do_sample=true`、`temperature=0.7`、
`top_p=0.8`、`top_k=20`、`min_p=0`、`presence_penalty=1.5`、`repetition_penalty=1`，基础 seed 为 42。
Transformers 5.5 没有原生 presence penalty；本项目用 logits processor 对已生成 token 一次性扣分，
不惩罚提示词中的 token。参数可通过同名命令行选项覆盖，例如 `--temperature 0.7 --seed 42`。
直接运行 Python 入口仍默认贪心；启动脚本恢复旧贪心行为需要同时传 `--no-do-sample --presence-penalty 0`。

只用物理 GPU 0、3，依次运行 2B 和 4B 的三数据集采样基线：

```bash
bash scripts/run_emotion_sampling_baselines.sh
```

该脚本每个模型分成两份在双卡上并行，batch 为每卡 64，输出分别是
`outputs/qwen35_2b_emotion_draft_sampled_seed42` 和 `outputs/qwen35_4b_emotion_draft_sampled_seed42`。
可添加 `2B` 或 `4B` 仅运行一个模型；默认使用 `--resume`，聚合前验证完整样本覆盖。
采样结果保存版本 3 的完整解码配置；随机种子由基础 seed 和原始 batch 样本 ID 派生。
中断续跑保持原始 batch，部分写入的 batch 会重算后仅补写缺失记录；更改 batch、seed 或采样参数会拒绝续跑。

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

启动脚本默认输出目录是 `outputs/qwen35_emotion_draft_sampled/`（可用 `OUTPUT_DIR` 覆盖）；
直接运行 Python 入口仍默认 `outputs/qwen35_emotion/`。每个数据集分别保存：

- `predictions.jsonl`：逐图路径、真值、预测类别、单独的 `draft` 解释、完整 `raw_output` 和是否正确，每条立即写出。
- `metrics.json`：Accuracy、Macro-F1、每类 precision/recall/F1/support、混淆矩阵。
- `config.json`：模型路径、提示词、推理参数及样本清单摘要，用于核对断点续跑。

所选数据集完成后额外写出 `summary.json`。指标范围为 0–1；评分只读取解释之后的唯一
`\boxed{合法类别}`，解释中提及的情绪不参与评分。缺少非空 `<draft>...</draft>`、答案框不完整、
多个最终答案或格式之外有额外文字的输出记为 `prediction: null` 并计错，不从分母剔除。
Macro-F1 始终覆盖数据集全部类别，
无样本类别记为 0，所以 `--limit` 仅适合验证流程，不能当作完整测试指标。
混淆矩阵行为真值、列为预测，最后一列是无效输出。

直接运行 Python 入口时，`--batch-size` 默认为 1；启动脚本使用其 `batch_size` 变量（当前为 `64`）。
`--max-pixels` 默认为 1,048,576，
`--min-pixels` 默认为 65,536，均指缩放后的图片像素面积。
`--max-new-tokens` 默认为 512，覆盖解释与最终答案。完整参数见 `bash scripts/run_emotion_baseline.sh --help`。
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

## DeepSeek-V4.1-Flash：三个数据集全量评估

入口为 `scripts/run_deepseek_emotion.sh`，Python 模块为 `emotion_eval.deepseek`。
默认评估三个数据集，共 **28,626** 张，与上方本地 baseline 的评估范围一致：

| 数据集 | 默认范围 | 真值来源 |
| --- | --- | --- |
| EmoSet | 官方 `test.json`，17,716 张，split 为 `official_test` | 每图 annotation 的 `emotion`，并核对测试清单类别 |
| Emotion6 | `Emotion6/Emotion6/images/` 全部 1,980 张 | 六个类别目录名：`anger, disgust, fear, joy, sadness, surprise` |
| EMO8 | train 7,141 + val 889 + test 900，共 8,930 张 | 类别目录名，`sad` 映射为 `Sadness` |

Emotion6 为与现有 baseline 对比，继续沿用目录标签，不使用 `ground_truth.txt` 中的人工情绪分布；
本地没有独立测试划分，结果标记为全量评估。如需指定测试划分，可传
`--emotion6-test-list /path/to/test.txt`，每行是相对 `images/` 的路径，例如 `anger/1.jpg`。
EMO8 的 split 为 `all_train_val_test_as_test`，样本 ID 保留 `train/`、`val/`、`test/` 前缀。
这些图片全部只用于推理。

旧入口 `scripts/run_deepseek_emoset.sh` 保留为兼容入口，转调新脚本；
**旧入口无参运行也会评估三个数据集全量，不再默认使用 EmoSet 2k 子集。**
只运行 EmoSet 官方测试集时，显式传入 `--datasets emoset`；旧 2k 实验的运行方式见本节末尾。

模型参数默认使用官方 API 名称 `deepseek-flash`，对应 V4.1-Flash 系列；
这是服务端维护的别名，实际返回的模型名称会随结果保存。
模型名称参考 [DeepSeek 官方发布说明](https://api-docs.deepseek.com/news/news260910/)，
图像请求格式参考 [Vision 文档](https://api-docs.deepseek.com/guides/vision/)。
脚本设置 `temperature=0`，使用 `emotion_eval/prompts.py` 中的 `DEEPSEEK_EMOTION_PROMPT`，
按当前数据集填入候选类别。先用 2–4 句话解释图片中的视觉依据，将解释放在 `<draft>...</draft>` 内，
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

依赖仅为 **Python 3.10+ 和 Pillow**，无需本地模型或 GPU。
启动脚本默认使用已有的 `vision-opd` Python，可通过 `PYTHON_BIN=/path/to/python` 覆盖。
API key 通过环境变量 `DEEPSEEK_API_KEY`，或启动脚本读取的项目根目录 `.deepseek.env` 提供；
该本地文件已加入 Git 忽略规则。首次使用：

```bash
cd /data/xiyan.zhao/work/Draftout
cp .deepseek.env.example .deepseek.env
# 编辑 .deepseek.env，填入自己的 DEEPSEEK_API_KEY。

# 不需要 API key：校验三个数据集的标签与全部图片解码，不请求 API、不写输出文件。
bash scripts/run_deepseek_emotion.sh --dry-run

# 填好 key 后，每个数据集前 2 张检查真实 API，共 6 张；结果放在独立目录。
bash scripts/run_deepseek_emotion.sh \
  --limit 2 --output-dir outputs/deepseek_emotion_smoke

# 三个数据集全量，共 28,626 张。
bash scripts/run_deepseek_emotion.sh

# 只评估 EmoSet 官方测试集，17,716 张。
bash scripts/run_deepseek_emotion.sh --datasets emoset

# 也可指定任意组合。
bash scripts/run_deepseek_emotion.sh --datasets emotion6 emo8

# 中断或 API 错误后，在同一输出目录继续。
bash scripts/run_deepseek_emotion.sh --resume
```

在 `scripts/run_deepseek_emotion.sh` 顶部可直接修改这些变量；同名命令行参数可覆盖脚本设置：

```bash
datasets=(emoset emotion6 emo8)
workers=4
model_name="deepseek-flash"
max_tokens=512
max_image_side=1024
output_dir="outputs/deepseek_v41_flash_emotion"
```

例如 `datasets=(emotion6)` 只运行 Emotion6，`--workers 8` 调整并发数，
`--model` 对应脚本变量 `model_name`。仅有 API key 配置不会自动发起请求；
执行非 `--dry-run` 命令才会调用 API。
若使用其他 Python，可先执行 `/path/to/python -m pip install Pillow`，再运行
`PYTHON_BIN=/path/to/python bash scripts/run_deepseek_emotion.sh`。
也可在项目根目录直接执行 `python -m emotion_eval.deepseek`，默认同样评估三个数据集全量；
此时自行设置环境变量 `DEEPSEEK_API_KEY`，不会通过启动脚本加载 `.deepseek.env`。

常用参数如下，完整选项见 `bash scripts/run_deepseek_emotion.sh --help`：

| 参数 | 默认值与用途 |
| --- | --- |
| `--datasets` | 默认全部；可传 `all`，或 `emoset emotion6 emo8` 的任意组合 |
| `--data-root` | `/data/xiyan.zhao/work/dataset`，包含 `EMOSET/`、`Emotion6/`、`EMO8/` 的父目录 |
| `--emotion6-test-list` | 默认不用独立清单；可指定相对 Emotion6 `images/` 的逐行路径清单 |
| `--limit` | 默认不截断；指定时每个所选数据集仅运行前 N 张，用于检查流程 |
| `--workers` | `4`，并发请求数 |
| `--model` | `deepseek-flash` |
| `--base-url` | `https://api.deepseek.com` |
| `--max-tokens` | `512`，单张图片解释与最终答案的总生成 token 上限 |
| `--max-image-side` | `1024`，发送图片的最长边上限 |
| `--timeout` | `120`，单次请求超时秒数 |
| `--max-retries` | `3`，可重试错误的最大重试次数 |
| `--output-dir` | `outputs/deepseek_v41_flash_emotion`，各数据集输出目录的父目录 |
| `--resume` | 校验数据和配置后，跳过已有成功 API 响应，继续失败和未完成样本 |
| `--dry-run` | 核对标签并解码全部选中图片，不请求 API、不写输出文件 |

发送前按 EXIF 方向转正图片，只缩小、不放大，转换为 RGB JPEG，质量设为 95，
移除图片元数据。通过 base64 发送图像内容，图片路径、文件名和真值标签不进入提示词。

默认输出根目录为 `outputs/deepseek_v41_flash_emotion/`。
每个数据集分别写入 `emoset/`、`emotion6/`、`emo8/` 子目录，包含：

- `config.json`：模型、提示词、请求参数、图像处理参数和数据指纹。
- `samples.json`：本次实际选择的样本清单。
- `predictions.jsonl`：逐图完整 `raw_output`、单独的 `draft` 解释、预测类别、真值与 token 使用量。
- `errors.jsonl`：失败请求记录，便于排查和续跑。
- `metrics.json`：完成状态、样本进度、Accuracy、Macro-F1、逐类指标、混淆矩阵和 token 使用量。

输出根目录的 `summary.json` 按数据集名称映射本次 `--datasets` 所选数据集的 metrics。
默认汇总三个数据集；单独运行某个数据集时，汇总仅包含该数据集，其他历史目录的结果仍保留。
默认拒绝覆盖已有运行文件；重复实验使用新的 `--output-dir`，恢复已有实验使用 `--resume`。
续跑会校验数据及配置，并跳过已保存的成功 API 响应；旧版仅输出标签的实验不能与新提示词续跑混用。
成功返回但缺少非空 `<draft>...</draft>`、末尾唯一 `\boxed{合法类别}` 或含额外文字的输出
计错且不自动重试，完整原始回答仍会保存。请求错误达到重试上限后，程序等待当前并发批次结束、
保存已完成结果并退出；续跑会重新处理失败和未完成样本。
网络失败保持待处理状态，不充当模型分类错误。每个 `metrics.json` 的 `complete` 为 `true`
时才是该数据集本次清单的完整评估，否则指标仅反映已完成部分；`--limit` 的小样本结果不代表全量性能。

保留旧 EmoSet 2k 实验时，显式传 `--test-list`，且不传 `--datasets`，进入旧版单 EmoSet 模式。
此时 `--data-root` 直接指向含 `test.json` 的 `EMOSET/` 目录，输出文件平铺在指定目录下，
不会增加 `emoset/` 子目录。启动脚本同样支持此模式，并按上述方式加载 API key：

```bash
bash scripts/run_deepseek_emoset.sh \
  --test-list splits/emoset_test_2000.json \
  --data-root ../dataset/EMOSET \
  --output-dir outputs/deepseek_v41_flash_emoset_2k \
  --resume
```

去掉 `--resume` 并使用新的输出目录可从头运行；加 `--dry-run` 可只校验图片。
`splits/emoset_test_2000.json` 从此前 `Qwen3-VL/splits/emoset_test_2000.json` 原样复制，
从官方 17,716 张测试图片中按原始类别比例分层抽样，seed 为 `20260612`，
包含 2,000 张互不重复的图片，不包含训练样本。运行时检查清单属于官方测试集，
并核对图片、annotation 和类别。该旧测试子集与下方 GRPO 训练用的 2,000 张样本相互独立。

## EmoSet 2k 纯 GRPO 训练

入口是 `scripts/run_emoset_grpo.sh`，使用 `vision-opd`、本仓库 `verl` 和原版
`/data/xiyan.zhao/work/model/Qwen3.5-2B`。当前实验为
`emoset_grpo_2b_angles_concise_2k_20261110`，总共 **10 epochs**；默认恢复该实验最新 checkpoint。

训练与 `--prompt-style angles` 评估入口共用 `emotion_eval/prompts.py` 的 `EMOTION_ANALYSIS_DRAFT_PROMPT`：
在 `<draft>...</draft>` 中只列出 1–3 个与情绪有关的分析角度，不作分析；
在标签外用一个简洁段落综合分析图片证据，最后输出 `\boxed{类别}`。
训练消息额外包含一个 `<image>` 占位符；CLI 的 `angles` 名称保留兼容。

`emotion_grpo/reward.py` 复用 angles 严格解析器：非空 draft、独立分析、唯一末尾 boxed
均齐全且情绪类别正确得 **1**，其他情况得 **0**。类别忽略大小写，解释中提到的情绪
不参与答案判定；不额外奖励解释质量，也不以列表行数启发式判断角度数。
独立 thinking 模式关闭，完整输出上限为 **1,024 tokens**；
`max_prompt_length=1024`，`rollout.max_model_len=2048`。

仍使用组内均值、标准差归一化的 GRPO advantage 和 clipped policy loss，
不启用 OPD、teacher、自蒸馏、critic、reward model、KL 或 entropy bonus。
训练 rollout 保留 `temperature=1, top_p=1`，每张图采样 4 个回答；验证使用单回答贪心解码。
学习率为 `1e-6`，训练和 rollout 使用 BF16/FSDP2，参数与优化器状态启用 CPU offload。

数据默认保存在 `outputs/emoset_grpo/data_angles_concise_20261110/`：

- 官方 train 中固定抽取 **2,000** 张，每类 250 张；官方 val 中固定抽取 **256** 张，每类 32 张，seed=42。
- 选样与之前的实验一致，当前目录保存新版提示词及独立数据产物。
- 检查 train/val/test 清单互斥、图片和 annotation 存在、标签一致、训练与验证图片内容无重合。
- 保存 parquet、两份样本 manifest、来源哈希及提示词哈希。重复准备相同数据会校验并复用；
  不同内容需使用新目录，或显式调用数据准备模块的 `--overwrite`。
- 图像经多模态字段传入，文件名和真值不进入文本；图像像素面积保持 65,536–262,144。

```bash
cd /data/xiyan.zhao/work/Draftout

# 只准备数据，不加载模型或启动训练
bash scripts/run_emoset_grpo.sh --prepare-only

# 按单卡 GPU 2 的批次配置进行 CPU 预检
# --check 在计算批次后隐藏所有 GPU，仅检查数据、处理器、配置和奖励
CUDA_VISIBLE_DEVICES=2 GRADIENT_ACCUMULATION_STEPS=4 \
  bash scripts/run_emoset_grpo.sh --check

# GPU 2 单卡训练：每步 8 张图 / 32 条回答，10 epochs = 2,500 steps
CUDA_VISIBLE_DEVICES=2 GRADIENT_ACCUMULATION_STEPS=4 \
  bash scripts/run_emoset_grpo.sh

# 或使用两张卡：单卡 micro batch 8，梯度累积 2 次，每步仍为 8 张图
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_emoset_grpo.sh
```

脚本默认卡号是 `2,3`；执行时显式设置 `CUDA_VISIBLE_DEVICES` 即可选择训练卡。
单卡和双卡命令是可选方案，不要同时写入同一个实验目录。
如需后台运行，可执行以下命令并用 `tail -f` 查看日志：

```bash
mkdir -p outputs/emoset_grpo/logs
nohup env CUDA_VISIBLE_DEVICES=2,3 \
  bash scripts/run_emoset_grpo.sh \
  > outputs/emoset_grpo/logs/draft_2b.log 2>&1 &
tail -f outputs/emoset_grpo/logs/draft_2b.log
```

默认 checkpoint 目录是
`outputs/emoset_grpo/checkpoints/emoset_grpo_2b_angles_concise_2k_20261110/`。
每 25 steps 验证并保存，最多保留两份 actor checkpoint；同时保存训练配置、
验证输出以及 `prompt_contract.json`。恢复时校验 prompt 和奖励规则，拒绝旧 prompt
或缺少契约的 checkpoint。中断后以相同模型、数据、GPU 数和 batch 设置重新执行同一命令即可。
`trainer.total_epochs` 是累计轮数，不是追加轮数。

```bash
# 修改训练总轮数或学习率
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_emoset_grpo.sh \
  trainer.total_epochs=5 actor_rollout_ref.actor.optim.lr=5e-7

# 另一份全新实验
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_emoset_grpo.sh \
  trainer.resume_mode=disable trainer.experiment_name=emoset_grpo_2b_draft_run2

# 可选的真实 GPU 单步训练；--check 不等价于这项 GPU 检查
# 输出到独立的 emoset_grpo_2b_angles_concise_2k_smoke 目录
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_emoset_grpo.sh --smoke
```

单卡 micro batch 默认 **8**、梯度累积默认 **2**、每图回答数默认 **4**，
两卡下每步仍处理 8 张图 / 32 条回答，与原训练的有效 batch 相同。
计算公式为 `每步图片数 = BATCH_SIZE × GRADIENT_ACCUMULATION_STEPS × GPU 数 / ROLLOUT_N`。
每步图片数必须为整数并整除 2,000；脚本会在加载数据前检查。
actor/log-prob micro batch、有效 batch 和 GPU 数统一由脚本推导，不能单独用 Hydra 覆盖。
其他 Hydra 参数可按上述形式追加。

Qwen3.5 使用保留 padding 的独立批次维度，关闭动态 batch，SP=1。
`BATCH_SIZE=1` 时才自动启用单样本去 padding；所有批次仍须满足整除约束。
也可通过环境变量覆盖 `PYTHON_BIN`、`MODEL_PATH`、`EMOSET_ROOT`、`DATA_DIR`。

旧普通 draft 与旧 angles GRPO 实验已于 2026-11-11 清理，包括其验证输出、
checkpoint、专属数据、日志、诊断及旧 GRPO 测试集评测结果；旧合并模型此前已删除。
清理清单见 `outputs/cleanup_old_grpo_experiments_20261111.json`。
当前 concise 实验、原版 `work/model/Qwen3.5-2B` 及未训练模型的基线评测结果保留。

## 2×2 Draft GRPO：全局收缩与分段 advantage

入口是 `scripts/run_emoset_draft_grpo.sh`，复用原训练脚本、数据、vLLM rollout、
奖励接口和 PPO clipped loss。配置位于 `verl/trainer/config/emoset_draft_grpo.yaml`，
独立实验目录为 `outputs/emoset_grpo/checkpoints/emoset_draft_grpo_2b_2k/`。
原来的 `run_emoset_grpo.sh` 仍可用于普通 GRPO。

使用当前 `EMOTION_ANALYSIS_DRAFT_PROMPT`。**每次启动，包括断点恢复，先完成 10 次普通
GRPO 更新**，使用原严格格式且分类正确才得 1 分的奖励；这 10 步计入累计 `global_step`。
随后自动切到Draft GRPO 阶段。例如恢复第 375 步时，376–385 为 warmup，386 开始draft 与后缀采样。
验证不计入 warmup，始终使用原单回答贪心流程，并记录分类准确率及格式通过率。

Draft GRPO 阶段在同一策略快照下，每图采样两个 draft，在 `</draft>` 处停止，各自沿完全相同的
assistant token 前缀独立续写两次。中间不增加 user 消息，也不更新参数。4 条完整轨迹
通过原图像处理和 padding 流程，再计算旧策略 log-prob 和原 PPO loss。总输出预算仍为
1,024 tokens，由 draft 和后缀共享，包含 EOS。缺少闭标签、提前 EOS、空 draft 等情况会
明确报错，避免补造 token 或静默丢弃样本；后缀达到长度上限时沿用原来的奖励处理。

Draft GRPO 阶段奖励只取唯一可解析的 `\boxed{Label}` 是否正确，格式不额外加减分。
`image_id/draft_id/branch_id/draft_token_count` 随轨迹传递，batch 重排后仍按 ID 恢复
`R[B,2,2]`。先在每张图内部中心化，再对整个 rollout batch 汇总平方和：

```python
draft_mean = R.mean(dim=2)
image_mean = draft_mean.mean(dim=1)
MS_B = 2 * (draft_mean - image_mean[:, None]).square().sum() / B
MS_W = (R - draft_mean[:, :, None]).square().sum() / (2 * B)
ema_ms_b = beta * ema_ms_b + (1 - beta) * MS_B
ema_ms_w = beta * ema_ms_w + (1 - beta) * MS_W
s = (1 - ema_ms_w / ema_ms_b.clamp_min(eps)).clamp(shrink_min, 1)
value = image_mean[:, None] + s * (draft_mean - image_mean[:, None])
draft_adv = value - image_mean[:, None]
suffix_adv = R - value[:, :, None]
```

统计、EMA 和分段 advantage 使用 BF16、无梯度，EMA 初值为 0。
`num_drafts=2`、`num_continuations_per_draft=2`、`ema_beta=0.9`、`shrink_min=0.1`、
`eps=1e-6`；本版固定 2×2 和每次启动 10 步 warmup。
Ray driver 收齐所有 rollout worker 的轨迹后统一统计；若直接在分片调用估计器，
`global_batch=False` 会 all-reduce 平方和与图像数，各 rank 需要持有每张图片完整的 2×2 分支。

draft 的全部生成 token（含起止标签）使用 `draft_adv`，其后包括 EOS 使用 `suffix_adv`；
prompt 不进入 response loss，padding 的 advantage 和 mask 为 0。同一 draft 的两个副本
均保留，沿用原 `seq-mean-token-mean` 聚合；Draft GRPO 阶段不做 GRPO group normalization、
advantage whitening、前缀去重或阶段加权。KL 等配置继承原 recipe，当前默认关闭。

每个 checkpoint 保存 `draft_advantage_state.json`，恢复时加载 EMA，并重新计 10 步 warmup。
方法契约会拒绝直接混入旧普通 GRPO checkpoint，避免相同 prompt 下混用奖励和估计器状态。

```bash
cd /data/xiyan.zhao/work/Draftout

# CPU 预检：真实数据/图像处理、配置和两阶段奖励检查
bash scripts/run_emoset_draft_grpo.sh --check

# 正式训练；已有同一 Draft GRPO 实验 checkpoint 时自动恢复
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_emoset_draft_grpo.sh

# 可选 GPU 检查：独立 smoke 实验，10 步普通 GRPO + 1 步 Draft GRPO 更新
CUDA_VISIBLE_DEVICES=2,3 bash scripts/run_emoset_draft_grpo.sh --smoke

# 最小 CPU 正确性检查
CUDA_VISIBLE_DEVICES='' /data/xiyan.zhao/anaconda3/envs/vision-opd/bin/python \
  -m unittest discover -s tests -p 'test_draft_*.py' -v
```

Draft GRPO 入口默认单卡 batch 8、梯度累计 2；两卡每步 8 张图 / 32 条完整回答。
2,000 张训练图每个 epoch 为 250 步，10 epochs 共 2,500 步。
可通过 `BATCH_SIZE` 和 `GRADIENT_ACCUMULATION_STEPS` 覆盖。
日志包含 `draft/reward_mean`、`draft/MS_B`、`draft/MS_W`、`draft/s`、
`draft/mean_abs_draft_adv`、`draft/mean_abs_suffix_adv`，以及 `draft/warmup`、
`draft/session_step`。完整输出与分支 ID 写入实验目录的 `rollout/<step>.jsonl`；
验证输出在 `validation/<step>.jsonl`。同一步恢复重评会沿用现有逻辑覆盖对应验证文件。
