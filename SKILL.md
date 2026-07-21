---
name: dl-train
description: Use this skill when designing, implementing, reviewing, or migrating an enterprise deep learning training framework built on native LightningCLI with datasets.load_dataset as the only data ingestion API. It covers strict scratch versus full/LoRA model-source interfaces, class_path/init_args YAML, LightningModule and LightningDataModule boundaries, immutable resolved run configs, checkpointing, export, and tests.
---

# 深度学习训练框架规范

用于搭建、扩展或审查企业级训练框架。核心目标：配置可追踪，数据可验证，训练可恢复，实验可复现，模型可交付，测试可自动化。

## 结构

```text
src/<package>/
  commands/        # 模型注册与 LightningCLI 入口
  configs/         # run 级 resolved config 保存与漂移保护
  tasks/           # LightningModule、DataModule、指标
  models/          # transformers 注册、自定义模型、processor
  criterions/      # loss 与训练目标
  checkpointing/   # checkpoint callback
  exporting/       # 权重与交付格式导出
examples/
tests/
```

## 硬约束

- 训练入口必须使用原生 `LightningCLI`，不得用 Typer、OmegaConf 或自定义工厂手工装配训练对象。
- YAML 用 `class_path` / `init_args` 声明 `LightningModule`、`LightningDataModule`、logger、callbacks 和可注入 criterion；`trainer` 使用 LightningCLI 原生参数命名空间。
- `commands/` 入口只注册自定义模型并创建 `LightningCLI`；不得实现配置合并、对象工厂、训练或导出逻辑。
- `configs/` 只承载 run 配置持久化、规范化和漂移保护等 CLI 配置基础设施。
- 所有训练、验证、测试数据必须且只能通过 `datasets.load_dataset()` 加载；禁止 `load_from_disk`、`Dataset.from_*`、pandas/pyarrow 直读、自定义文件扫描器和自定义 Dataset 数据入口。
- `models/` 必须接入 `transformers` 注册表，支持字符串路径和 Auto 类加载。
- 按模型初始化来源拆分 Scratch、Full Finetune、LoRA Finetune 三个具体 Task：Scratch 只接收 `model_config_path`，Full/LoRA 只接收 `pretrained_model_path`；禁止通用 `model_path` 和互斥可选参数。
- 默认提供 `criterions/`；只有模型稳定返回完整 loss 时才可省略。
- `tasks/` 连接模型、batch、criterion、指标和 optimizer。
- `checkpointing/` 与 `exporting/` 独立于 CLI。

## AI 执行流程

1. 先读 `README.md`、`pyproject.toml`、`src/`、`examples/` 和相关测试。
2. 将当前skill的`.gitignore`挪到项目根目录，如果已经有了则直接覆盖。
3. 先检查 git 分支状态；只能在 `feature/<任务名>` 分支完成代码修改、测试和文档更新。
4. 判断改动属于 config、model、data、criterion、task、checkpoint、export、CLI 还是 test。
5. 新增任务顺序：model -> data -> criterion -> task -> metrics -> command registration -> example -> tests。
6. 保持最小改动；迁移旧入口时以原生 LightningCLI 模式替换配置与装配层，不顺带重构 Task、DataModule、criterion、checkpoint 或 export。
7. 新增行为同步补测试并运行最小验证命令。
8. 合并前确认工作区干净；feature 分支只能通过 squash merge 合并回 `dev`。

## 质量门槛

- YAML 与命令行 override 必须由 `LightningCLI` 解析，禁止维护第二套配置合并语义。
- DataModule 必须把数据源参数显式映射到唯一的 `datasets.load_dataset()` 调用，并保留加载参数的 resolved config。
- 每个 run 保存包含命令行覆盖结果的 `resolved.yaml`；同一 run 的已有配置与本次 resolved config 不一致时，在训练前拒绝执行。
- Task 构造签名、YAML、checkpoint metadata 和 export API 必须使用同一模型来源字段，传入其他任务的路径字段时必须在对象创建前报错。
- 训练前验证 split、字段、样本数、标签和缓存参数。
- checkpoint、resume、export 边界清晰。
- 日志能还原配置、override、trainer 参数、指标和输出路径。
- 默认测试不依赖真实大模型、GPU、网络或远程大数据集。
- 默认 Python 版本约束为 `>=3.10,<3.13`，默认使用 `python3.10` 解释器创建 Poetry 环境。
- `torch` 及依赖 `torch` 的包不得写入 `pyproject.toml`，最终 lock 文件不得出现 `torch` 传递依赖；确需声明时只能作为 optional 依赖。
- `torch`、`torchvision`、`torchaudio`、`datasets`、`modelscope` 等 torch 生态或可能引入 torch 的包只能通过 pip 安装，并在项目 README 中写明安装命令；默认安装 torch 2.8 系列。`transformers`不会引入torch依赖，可以通过`poetry add`安装。
- 遵循 Git Flow：`main` 只作为稳定主线，`dev` 只作为集成分支；不得直接在 `main` 或 `dev` 提交开发改动。
- 每次开发前从 `dev` 创建 `feature/<任务名>` 分支；完成后用 squash merge 合并回 `dev`，禁止普通 merge commit；除非明确要求发布，不从 `dev` 合并到 `main`。

## 架构与目录规范

目标：用稳定边界承载模型、数据、训练、导出变化。

### 标准分层

```text
src/<package>/
  commands/
  configs/
  tasks/
  models/
  criterions/
  checkpointing/
  exporting/
```

项目初始化时在根目录创建 `data-bin/`、`model-bin/`、`outputs/`、`tmp-workspace/` 四个目录：`data-bin/` 用于存放数据，`model-bin/` 用于存放预训练模型和已经导出准备交付的模型，`outputs/` 通常用于存放训练的直接产物，`tmp-workspace/` 用于存放临时脚本和其他临时内容。

- `commands/`：注册自定义模型并创建原生 `LightningCLI`。
- `configs/`：resolved run config 的保存、比较和漂移保护；不定义另一套训练 schema。
- `models/`：configuration/modeling/processing、底座兼容、`transformers` 注册。
- `criterions/`：loss、训练目标、多 loss 组合；模型稳定内置 loss 时可省略。
- `tasks/`：训练步骤、验证步骤、指标、optimizer、scheduler。
- `checkpointing/`：checkpoint 配置和 callback。
- `exporting/`：训练产物到推理或平台格式。

### 边界

- CLI 入口只调用模型注册函数并创建 `LightningCLI`；由 LightningCLI 解析配置、实例化对象并调用 `Trainer` 子命令。
- 训练 YAML 顶层按 LightningCLI 原生结构使用 `model`、`data`、`trainer`、`seed_everything` 和 `ckpt_path`；logger 与 callbacks 放入 `trainer`。
- 不要创建 `_build_logger`、`_build_training_components`、`_task_init_kwargs`、`_data_init_kwargs`、`_load_train_config` 等配置工厂或手写 `trainer.fit(...)`。
- Model 类、配置、processor 必须使用 `transformers` 注册表；入口优先用 `AutoConfig`、`AutoModel`、任务型 AutoModel、`AutoProcessor`；支持字符串路径、HF repo、本地目录。
- `models/` 不依赖 `tasks/`、DataModule 或 CLI；任务相关、蒸馏、对比学习、多 loss 加权放 `criterions/`。
- DataModule 负责调用 `datasets.load_dataset()`、选择 split、校验字段和标签、限制样本、map 预处理、缓存、DataLoader、collator 和训练集增强；不计算 loss、不构造 optimizer、不保存 checkpoint。
- Criterion 默认放 `criterions/`，输入为模型输出和 batch 标签；不读配置、不加载数据、不创建模型，并且可单测。
- Task 通过 Auto 类或注册表加载模型/processor/适配器，调用模型 loss 或 criterion，实现 step、指标聚合、optimizer、scheduler、dtype cast、生成参数和标签 mask；不读配置文件，不创建 CLI，不管理导出目录。
- 三类具体 Task 继承同一公共 Task 复用 step、指标和 optimizer/scheduler；公共基类不接收 `model_config_path`、`pretrained_model_path` 或 LoRA 开关。

### 新增任务流程

1. 在 `models/` 实现或接入模型、configuration、processor，并完成 `transformers` 注册，保证可用字符串路径和 `AutoModel`/任务型 AutoModel 加载。
2. 在 `tasks/<task>/datamodule.py` 实现数据加载和校验。
3. 在 `criterions/` 实现任务 loss；若模型已内置 loss，则在 Task 中显式使用模型返回的 loss。
4. 在 `tasks/<task>/task.py` 或 `finetune.py` 实现 Task。
5. 在 `tasks/<task>/metrics.py` 实现指标。
6. 在模型注册函数中注册新增模型；训练入口保持只创建 `LightningCLI`。
7. 在 `examples/<task>/` 提供原生 LightningCLI YAML 和脚本。
8. 在 `tests/` 覆盖 YAML 解析、class 实例化、override、配置漂移保护、模型 Auto 加载、数据校验、criterion 和训练步骤。

### 命名

- Python 包内用小写下划线。
- 类名包含任务和职责，例如 `TextClassifyDataModule`。
- checkpoint monitor 与 Task log 名一致，例如 `val_loss`。

## 配置系统规范

目标：只使用 LightningCLI/jsonargparse 的配置语义，同时支持本地调试、平台训练和实验复现。

涉及 CLI、YAML、run config 或旧入口迁移时，必须完整读取 [原生 LightningCLI 模式](references/native-lightning-cli.md)。
涉及模型初始化、checkpoint 恢复或导出时，必须完整读取 [模型初始化来源边界](references/model-initialization.md)。
创建训练配置时按任务选择 [Scratch](templates/lightning-cli-scratch.yaml)、[Full Finetune](templates/lightning-cli-full-finetune.yaml) 或 [LoRA Finetune](templates/lightning-cli-lora-finetune.yaml) 模板。

### YAML 与 Override

- `model` 与 `data` 使用 `class_path` / `init_args` 声明具体 Task 和 DataModule。
- Task 构造参数中的 criterion、网络或其他可替换组件也使用 `class_path` / `init_args`，并提供准确的构造类型标注。
- `trainer` 是 LightningCLI 固定 `trainer_class` 的原生参数命名空间；其 `logger` 和 `callbacks` 使用 `class_path` / `init_args`。不要为了给 Trainer 再包一层 `class_path` 而创建自定义工厂。
- 类型、必填项和默认值来自构造函数签名；不要复制到 dataclass schema。
- 命令行覆盖使用 LightningCLI 路径，例如 `--data.init_args.batch_size=4`；临时实验用 override，长期复用进入 YAML。
- 用 `poetry run python -m <package>.commands.app fit --config <yaml>` 启动训练；不得保留旧式位置参数配置和 dotlist 合并器。

### Resolved config 与漂移保护

- 使用独立的 `SaveConfigCallback` 子类保存 `parser.dump(config, skip_none=False)` 的结果，文件名固定为 `resolved.yaml`。
- run 身份由最终 `trainer.log_dir` 或项目显式约定的唯一 run 目录确定；logger、checkpoint 和 resolved config 必须落到可关联的同一 run 身份下。
- 首次启动原子写入 resolved config；再次使用同一 run 时先做规范化后的语义比较，完全一致才允许继续，不一致则在 `fit` 前抛出包含配置路径的错误。
- 比较与写入只在 global zero 执行，并把结果广播到全部进程；禁止用 `overwrite=True` 掩盖配置漂移。
- 保存内容必须包含 YAML、默认值和所有命令行覆盖的最终结果；不得只保存输入 YAML 或手工挑选字段。

### 变更

- 新增或调整构造参数时同步 Task/DataModule/criterion 类型标注、示例 YAML 和测试。
- 删除或重命名参数必须写迁移说明；平台依赖旧字段时保留兼容期。
- 生产配置不写个人路径、token 或密钥。

## 数据管道规范

目标：把外部数据变成可验证、可缓存、可复现的训练 batch。

### DataModule

DataModule 应包含唯一的 `datasets.load_dataset()` 调用、split 选择、字段/标签/样本数校验、样本限制、map 预处理、缓存命名、DataLoader 构造、collator 注入和训练集增强。

DataModule 不应包含 loss、optimizer、checkpoint、CLI 配置解析。

### 唯一加载入口

- 只允许 `from datasets import load_dataset` 后调用 `load_dataset(...)`；返回值必须是 Hugging Face `Dataset`、`DatasetDict`、`IterableDataset` 或 `IterableDatasetDict`。
- DataModule 构造参数至少显式提供 `dataset_path`，并按需提供 `dataset_name`、`data_dir`、`data_files`、`cache_dir`、`revision`、`streaming` 和任务所需的 builder 参数；这些参数直接映射到 `load_dataset` 的 `path`、`name` 及同名参数。
- 禁止调用 `load_from_disk`、`Dataset.from_dict`、`Dataset.from_file`、`Dataset.from_pandas`、`Dataset.from_generator`，禁止用 pandas、pyarrow、json、csv、soundfile 或目录遍历自行构造训练数据集。
- `load_dataset` 返回后允许使用 Hugging Face Dataset 的 `map`、`filter`、`select`、`shuffle`、`cast`/`cast_column` 和缓存能力完成确定性转换；不得在预处理阶段切换到另一套数据容器作为主数据集。
- 测试 fixture 也必须通过 `load_dataset` 加载本地小文件，或 monkeypatch DataModule 模块内导入的 `load_dataset`；不得因测试方便引入生产代码不支持的数据入口。

### 语音任务数据集格式

ASR、VAD、LID 数据集必须包含可解码的 `audio` 字段；`id` 可选但推荐提供。数据来源可以是 Hugging Face Hub、本地/远程 Parquet 或本地 AudioFolder，但都必须通过 `load_dataset`：Hub 使用 `load_dataset("namespace/dataset")`，Parquet 使用 `load_dataset("parquet", data_files=...)`，本地 AudioFolder 使用 `load_dataset("audiofolder", data_dir=...)`。本地语音数据优先使用 AudioFolder：

```text
dataset/
  test/
    audio/sample_001.wav
    metadata.jsonl
```

AudioFolder 的 `metadata.jsonl` 中 `file_name` 必须是相对 split 目录的路径，不写绝对路径。Parquet 不需要 `file_name`，但必须有 `audio` 列或能在加载时转换为 `datasets.Audio` 的音频列。

常用 AudioFolder 元数据模板放在 `templates/audiofolder-*-metadata.jsonl`；复制为目标 split 下的 `metadata.jsonl` 后，再按实际音频文件名和标签修改。

ASR 必需字段为 `audio` 和 `text`：

```jsonl
{"file_name":"audio/00a81de9d20f87d04465.wav","id":"00a81de9d20f87d04465","text":"HOTEL HOTEL BRAVO THANK YOU"}
```

VAD 必需字段为 `audio` 和 `seconds`。`seconds` 推荐使用 `starts` / `durations` 两个等长数组，单位为秒：

```jsonl
{"file_name":"segment-0001.wav","id":"segment-0001","seconds":{"starts":[0.06],"durations":[27.11]}}
```

VAD 数据集在转换为 Hugging Face AudioFolder 时，样本应默认切分为固定 30 秒片段；训练集、验证集和测试集的划分必须先基于原始音频完成，确保不同 split 之间不共享同一条原始录音，再在各 split 内做 30 秒切片。标注转换应优先通过 sample-level mask 完成：先将 Audition 标注转换为 mask，再按音频片段裁剪 mask，最后由 mask 还原为 seconds.starts 和 seconds.durations。转换完成后，应统计并打印每个 split 中有效音和无效音的时长及占比，用于检查数据分布。

LID 必需字段为 `audio` 和 `language_id`。语种标签按原始字符串严格比较，`<others>` 表示未知语种集合：

```jsonl
{"file_name":"CN000001.wav","id":"CN000001","language_id":"en"}
```

### 数据校验

训练前必须校验：

- train/validation/test split 存在。
- 每个 split 包含必需字段。
- 音频、图像、文本字段符合任务预期。
- 标签非空且格式正确。
- 样本限制后仍有数据。

错误信息直接指出缺失项，例如 `Dataset split validation is missing required columns: audio, text`。

### 预处理与缓存

- 原始数据不可变，预处理结果写新字段。
- 预处理函数尽量是纯函数。
- 文本标准化、标签模板、采样率、resize、多进程参数必须可配置。
- 训练和验证规则默认一致；差异必须显式说明。

缓存 key 至少包含原始 fingerprint、split、样本限制、prompt/prefix、标准化参数、确定性预处理参数、预处理版本号。

缓存目录优先级：显式 `preprocess_cache_dir` -> 本地数据集目录下项目缓存 -> 数据集库默认管理。预处理逻辑变化时提升 cache payload `version`。

### Collator

Collator 负责样本到模型输入：

- 加载和标准化原始模态。
- 调用 processor/tokenizer。
- padding、truncation、labels。
- mask 不参与 loss 的 token。
- 保留评估 reference 和 prompt-only 输入。

Collator 应可单测，不依赖 Trainer。

### 增强与 DataLoader

- 增强默认关闭，只在训练 split 启用；验证和测试禁用随机增强。
- 增强参数全部进入配置，高风险增强要有明确范围。
- DataLoader 推荐配置 `batch_size`、`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`、`shuffle`。
- `num_workers=0` 时不传 `persistent_workers=True`；`prefetch_factor` 只在 `num_workers>0` 时传。
- 小样本测试必须能用 `num_workers=0`；训练集 shuffle，验证集不 shuffle。

### 测试要点

- monkeypatch `load_dataset`，断言 DataModule 只调用一次且 `dataset_path`、`dataset_name`、`data_dir`、`data_files`、`cache_dir`、`revision`、`streaming` 映射准确。
- 使用本地小 fixture 经 `load_dataset("json"|"parquet"|"audiofolder", ...)` 完成正常 setup，不访问网络。
- 静态检查生产 DataModule 不调用 `load_from_disk`、`Dataset.from_*` 或 pandas/pyarrow 读取 API。
- split 或字段缺失时报清晰错误。
- sample limit、预处理参数、缓存 key 生效。
- 增强只作用于训练集。
- `num_workers=0` 可构造 DataLoader。

## 训练任务规范

目标：Task 只连接模型、batch、loss、指标和优化器，不混入数据加载、CLI 或导出。

### 基本结构

必须使用 `LightningModule`：

```python
class FinetuneTask(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train_loss", loss)
        return loss
```

### 初始化与 Loss

公共 Task 基类只实现训练步骤、验证步骤、指标、optimizer/scheduler 和共享模型调用逻辑。Scratch、Full、LoRA 三个具体 Task 分别实现自身初始化，并保存各自公开构造参数。

Task 初始化不读 YAML、不加载训练数据、不创建 DataLoader、不写 checkpoint。

- Scratch Task 只暴露 `model_config_path`，从 model-zoo 读取配置和 CMVN 等资源，经 `AutoConfig.from_pretrained(...)` 与任务型 `AutoModel.from_config(...)` 随机初始化；目录中出现任何权重文件都必须拒绝。
- Full Finetune Task 只暴露 `pretrained_model_path`，从同一个完整 Hugging Face 仓库读取配置、CMVN/processor 和预训练权重，并经任务型 `AutoModel.from_pretrained(...)` 加载。
- LoRA Finetune Task 只暴露 `pretrained_model_path` 和 LoRA 自身参数；先按 Full 路径完整加载基础模型，再冻结基础权重并注入 adapter，不额外接收 `model_config_path`。
- 禁止用一个 Task 的 `model_config_path: Optional[...]`、`pretrained_model_path: Optional[...]`、`use_lora` 组合分支表达三种互斥模式，也禁止 `**kwargs` 吞掉非法路径参数。
- 优先使用模型稳定返回的 `loss`。
- 任务相关 loss 使用 `criterions/`。
- `training_step` 只筛选 batch key、前向、记录 loss、返回 loss。
- 不在 `training_step` 中做大量预处理。

### dtype 与 LoRA

- dtype 推荐支持 `auto`、`bf16`、`fp16`、`fp32`。
- 浮点输入可按模型 dtype cast；token id、attention mask、labels 等整数张量不得 cast 成浮点。
- Trainer precision 与模型加载 dtype 必须兼容。
- LoRA/PEFT 参数只存在于 LoRA Task；Full 与 Scratch Task 的公开签名不包含 LoRA 开关或 adapter 参数。
- target modules 支持逗号分隔字符串或列表。
- optimizer 只优化 `requires_grad=True` 参数。

### 验证、指标、优化器

- Task 记录 `train_loss`、`val_loss` 和任务指标，例如 `val_wer`、`val_accuracy`。
- loss 类指标可逐 step 记录；生成或全局指标先收集 prediction/reference，再在 epoch end 聚合。
- 生成式评估使用 prompt-only 输入，不把标签传给 `generate`。
- 生成参数进入 evaluation 配置，decode 后统一标准化再算指标。
- 默认 `AdamW`；scheduler 使用训练总步数，warmup 用 ratio 或 steps，interval 明确为 `step` 或 `epoch`。
- `estimated_stepping_batches <= 0` 时可只返回 optimizer。

### 错误与测试

- 模型不兼容时早失败，错误信息说明缺失结构或不支持原因。
- run config callback 保存完整 resolved config，Lightning logger 记录 trainer 与实验信息，Task 不重复打印配置。
- 测试覆盖三类 Task 的公开签名、仓库完整性、真实模型加载、dtype、LoRA target/冻结状态、batch key 过滤、loss、指标和 optimizer 可训练参数。
- 测试关键工具逻辑时不要加载真实大模型。

## 实验、检查点与导出规范

目标：分清训练过程产物和推理交付产物。

### 实验日志

训练启动时保存包含 override 的 resolved config，并记录 commit、模型路径、数据集路径和 split、checkpoint 目录、trainer summary。

运行中记录 train/validation loss、任务指标、learning rate、epoch、global step 和可用系统指标。

logger 在 `trainer.logger` 中用 `class_path` / `init_args` 声明；启用但依赖缺失时抛清晰错误，不静默降级。

### Checkpoint

在 `trainer.callbacks` 中直接声明标准 `ModelCheckpoint`：

```yaml
trainer:
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: outputs/checkpoints
        filename: epoch={epoch}-step={step}-val_loss={val_loss:.4f}
        monitor: val_loss
        mode: min
        save_top_k: 3
        save_last: false
```

- `monitor` 必须对应 Task 实际 log 的指标。
- loss/error rate 用 `mode: min`；accuracy/F1/BLEU 通常用 `mode: max`。
- `save_top_k` 默认不宜过大；长任务建议启用 `save_last`。
- 文件名包含 epoch、step 和核心指标。

### Resume

恢复训练时明确输入是训练 checkpoint 而不是推理权重，并说明是否恢复 optimizer、scheduler、global step，数据版本和配置是否与原 run 一致，是否继续写入原 output dir。

- Scratch checkpoint metadata 只记录 `model_config_path`，不得出现 `pretrained_model_path`。
- Full/LoRA checkpoint metadata 只记录 `pretrained_model_path`，不得出现 `model_config_path`；LoRA 另记录 adapter 配置。
- 使用对应具体 Task 做真实 checkpoint 恢复测试；来源字段缺失、混入互斥字段或仓库内容不完整时立即失败。

不要默认猜测最新 checkpoint；若 CLI 支持自动选择，必须记录选择结果。

### Export

常见格式：Lightning `.ckpt`、完整 `state_dict`/safetensors、LoRA adapter、Hugging Face repository layout、推理服务自定义格式。

导出命令必须声明支持格式；不支持时直接报错，不猜格式。

导出接口与训练 Task 使用相同来源字段：Scratch 只接受 `model_config_path`，Full/LoRA 只接受 `pretrained_model_path`。不得提供兼容性 `model_path`、同时接收两种路径或静默忽略多余字段。

标准流程：读取权重 -> 校验 key/shape/dtype/metadata -> 读取模板或 base model -> 复制 tokenizer/processor/generation config -> 写目标权重 -> 更新说明 -> 轻量加载测试。

### 目录与指标

```text
outputs/
  runs/<run-id>/
  checkpoints/<run-id>/
  exports/<model-name>/
```

不要把中间 checkpoint 和最终导出物混在同一目录。

checkpoint monitor 选择贴近业务目标的验证指标：ASR 用 `val_wer`/`val_cer`，分类用 `val_accuracy`/`val_f1`/`val_auc`，检测用 `val_map`。指标成本高时可降低验证频率，但不能取消验证。

### 测试要点

- LightningCLI 能从示例 YAML 实例化 checkpoint callback，参数与类型正确。
- 分别检查 Scratch/Full/LoRA 的构造签名、YAML 非法字段报错、真实 checkpoint 恢复和对应导出接口。
- monitor 与 Task log 对齐。
- logger 的 `class_path` 可导入，`init_args` 可实例化；禁用时使用 Trainer 原生支持的 `logger: false`。
- export 拒绝不支持格式。
- export 复制必要非权重文件并可轻量加载。

## 代码规范

目标：降低误改概率。优先边界清晰、类型明确、错误可定位、日志可审计、测试可覆盖。

### 基本规则

- 默认 Python 版本约束为 `>=3.10,<3.13`，默认使用 `python3.10` 解释器创建 Poetry 环境。
- 使用 Python 3.10 兼容的类型标注，新文件默认 `from __future__ import annotations`。
- 模块名、函数名、变量名用小写下划线，类名用 `PascalCase`。
- 文件只承担一个清晰职责。

### 依赖方向

- `commands/` 只依赖模型注册、run config callback 和 LightningCLI。
- `tasks/` 可依赖 models、criterions、公共工具、指标。
- `configs/` 不依赖训练实现。
- `checkpointing/` 不依赖具体任务。
- `exporting/` 不依赖 CLI。
- `models/` 不依赖任务训练代码。

禁止 import 时启动训练、加载大模型、读取大数据；除 run config callback 的明确持久化职责外，禁止配置模块访问 GPU、文件系统或外部服务；禁止跨任务导入私有函数；禁止新增独立 `finetuning/` 杂物模块。

### 函数、类型、配置

- 输入参数显式，不依赖隐式全局状态。
- 返回值结构稳定，纯转换逻辑优先写成纯函数。
- 复杂流程拆成校验、转换、执行、记录。
- 公开构造参数必须有准确类型标注，使 LightningCLI/jsonargparse 能生成并校验配置。
- 不在 CLI 中手工过滤或重映射参数；第三方对象由有明确职责的 Task、DataModule 或 callback 构造。
- `Any` 只用于第三方对象边界。
- 路径参数对外可接收 `str | Path`，内部尽早转 `Path`。

### 异常与日志

- 错误信息必须指出具体配置字段、split、路径或权重 key。
- 不裸 `except Exception` 后静默跳过；可恢复问题 warning，不可恢复问题 raise。
- 可选依赖缺失时说明哪个功能需要哪个包。
- run config callback 保存 resolved config，Lightning logger 记录 trainer summary；入口不实现日志审计逻辑。
- Task 记录指标，DataModule 记录数据规模/split/缓存路径，Export 记录输入/输出/模板/格式校验。
- 日志不得写 token、密钥或敏感路径。

### 依赖、注释、工具

- 核心训练依赖写入项目依赖，开发测试依赖放 dev extras；但 `torch` 及依赖 `torch` 的包除外。
- `torch`、`torchvision`、`torchaudio`、`datasets`、`modelscope` 等 torch 生态或可能引入 torch 的包不得写入 `pyproject.toml` 的主依赖、dev 依赖或 extras，除非该声明明确为 optional 且不会进入默认 lock。
- `lightning` 依赖 `torch`，也只能按 README 使用 pip 安装；原生 CLI 同时安装兼容的 `jsonargparse[signatures]`。
- 交付前检查 lock 文件：除 optional 依赖声明外，不应出现 `torch` 作为直接或传递依赖。
- torch 生态依赖只能由用户按 README 说明使用 pip 安装，不在业务代码、Poetry scripts 或测试夹具中自动安装。
- 可选功能延迟 import，启用但缺失时抛清晰错误。
- 不在业务代码中自动安装依赖，不写个人镜像源或缓存路径。
- 代码能自解释时不写注释；对模型 patch、格式兼容、缓存 key、分布式指标写短说明。
- 推荐统一 `ruff format`/`black`、`ruff`、`pytest`，按项目复杂度选择 `mypy` 或 `pyright`。

### AI 修改规则

1. 先读相关模块和测试，不凭文件名猜边界。
2. 判断字段属于 config、model、data、criterion、task、checkpoint、export 还是 CLI。
3. 小步编辑，避免无关重构。
4. 新增行为同步补测试。
5. 运行最小验证命令。
6. 最终说明改动、验证和未验证项。

## 禁止事项

- 不要把数据读取、训练、日志、导出堆进一个脚本。
- 不要在 `datasets.load_dataset()` 之外增加任何训练数据加载入口。
- 不要只提供 notebook 作为生产入口。
- 不要让 YAML 字段脱离对应构造函数签名，或复制进第二套 schema。
- 不要混用训练 checkpoint 和推理权重。
- 不要写入个人路径、token 或密钥。
