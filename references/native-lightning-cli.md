# 原生 LightningCLI 模式

在实现、迁移或审查训练入口、YAML、run config 和相关测试时读取本文。

## 调用链

保持唯一调用链：

```text
python -m <package>.commands.app fit --config train.yaml <overrides>
  -> 注册 transformers 自定义模型
  -> LightningCLI 解析 YAML、默认值与命令行覆盖
  -> LightningCLI 实例化 LightningModule、LightningDataModule、Trainer、logger、callbacks
  -> run config callback 校验并保存 resolved.yaml
  -> LightningCLI 调用 trainer.fit(...)
```

迁移时删除 Typer/OmegaConf 解析、typed config factory、对象映射函数、logger/callback builder 和手写 `Trainer.fit`。先用 `rg` 找出入口、配置类、builder、示例和测试的完整引用，再删除已经无调用者的旧代码。

## 入口

入口只允许注册模型并创建 `LightningCLI`：

```python
from __future__ import annotations

from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI

from <package>.configs.run_config import DriftSafeSaveConfigCallback
from <package>.models import register_models


def main() -> None:
    register_models()
    LightningCLI(
        model_class=LightningModule,
        datamodule_class=LightningDataModule,
        trainer_class=Trainer,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_callback=DriftSafeSaveConfigCallback,
        save_config_kwargs={"config_filename": "resolved.yaml"},
    )


if __name__ == "__main__":
    main()
```

不要在入口中设置随机种子、矩阵精度、读取环境变量、加载 YAML、转换字典、创建 Task/DataModule/Trainer、调用训练或执行导出。确需全局运行时设置时，放入有明确职责的 callback 或 Task hook，并通过 YAML 声明。

## YAML

原生 LightningCLI 的 Trainer 类型由入口 `trainer_class=Trainer` 固定，因此 `trainer` 下直接书写 Trainer 参数；`class_path/init_args` 用于 model、data、logger、callbacks 和 Task 内可注入对象。以下 model 示例是 Full Finetune；Scratch 和 LoRA 必须改用各自具体 Task 与来源字段：

```yaml
seed_everything: 42

trainer:
  default_root_dir: outputs/runs/asr-demo/run-001
  accelerator: auto
  devices: auto
  precision: bf16-mixed
  max_epochs: 10
  log_every_n_steps: 10
  logger:
    class_path: lightning.pytorch.loggers.CSVLogger
    init_args:
      save_dir: outputs/runs
      name: asr-demo
      version: run-001
  callbacks:
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        dirpath: outputs/checkpoints/asr-demo/run-001
        filename: epoch={epoch}-step={step}-val_wer={val_wer:.4f}
        monitor: val_wer
        mode: min
        save_top_k: 3
        save_last: true

model:
  class_path: <package>.tasks.asr.ASRFullFinetuneTask
  init_args:
    pretrained_model_path: model-bin/pretrained/asr-base
    lr: 0.00002
    criterion:
      class_path: <package>.criterions.asr.ASRCriterion
      init_args:
        label_smoothing: 0.0

data:
  class_path: <package>.tasks.asr.ASRDataModule
  init_args:
    dataset_path: audiofolder
    dataset_name: null
    data_dir: data-bin/asr
    data_files: null
    cache_dir: null
    revision: null
    streaming: false
    train_split: train
    val_split: validation
    batch_size: 8
    num_workers: 4

ckpt_path: null
```

不要写成 `trainer: {class_path: ..., init_args: ...}`；这不是 LightningCLI 为固定 `trainer_class` 生成的配置结构。不要额外保留 `model/lora/optimizer/evaluation/checkpoint/logger` 的手工分组并在 CLI 中重新映射；属于 Task 的参数进入 `model.init_args`，属于 DataModule 的参数进入 `data.init_args`，属于 Trainer/logger/callback 的参数进入 `trainer`。

命令行覆盖示例：

```bash
poetry run python -m <package>.commands.app fit \
  --config examples/asr/train.yaml \
  --data.init_args.batch_size=4 \
  --model.init_args.lr=0.00001 \
  --trainer.max_epochs=1
```

## Resolved run config

在 `<package>/configs/run_config.py` 中实现 `SaveConfigCallback` 子类，保持入口轻量。实现必须满足：

1. 用 `self.parser.dump(self.config, skip_none=False, format="yaml")` 获取包含 CLI 覆盖的最终配置。
2. 以 `trainer.log_dir/resolved.yaml` 为基准配置路径。
3. 首次运行时只由 global zero 创建目录并原子写入；多进程间广播结果。
4. 文件已存在时，将已有 YAML 与候选 YAML 解析为普通结构后比较，避免仅因键顺序或格式产生误报。
5. 完全一致时不重写并允许继续；不一致时所有 rank 在训练开始前抛出 `RuntimeError`，错误包含 resolved config 路径。
6. 不设置 `overwrite=True`，不静默采用旧配置，也不在比较时忽略训练语义字段。

显式固定 `seed_everything`，避免自动随机种子让同一 run 的解析结果自然漂移。若项目允许 resume 时改变 `ckpt_path` 等调用级字段，必须先写清 run identity 规则并单独记录每次 invocation；不要未经约定就在比较函数里排除字段。

## 职责边界

- `LightningModule`：模型/processor、criterion、step、指标、optimizer/scheduler；不读 YAML、不加载 dataset、不管理 run 目录。
- `LightningDataModule`：只通过 `datasets.load_dataset()` 加载数据，再完成 split/字段校验、预处理、collator 和 DataLoader；不调用 `load_from_disk`/`Dataset.from_*`，不计算 loss，不创建 optimizer。
- `criterions/`：loss 纯逻辑；通过 Task 构造参数注入并在 YAML 中声明。
- `trainer.callbacks`：checkpoint、early stopping、学习率监控和 run config 生命周期逻辑。
- `trainer.logger`：实验 logger；依赖缺失时由类导入/初始化直接失败。
- `checkpointing/`：项目自定义 checkpoint callback；标准 `ModelCheckpoint` 直接在 YAML 声明。
- `exporting/`：独立导出 API/命令，只消费明确 checkpoint/权重，不放入训练入口或 Task。

## 测试

迁移时至少更新以下测试：

1. 入口测试：mock `register_models` 和 `LightningCLI`，断言入口只注册一次并以 `subclass_mode_model=True`、`subclass_mode_data=True`、run config callback 创建 CLI。
2. YAML 解析测试：用 LightningCLI/jsonargparse 解析示例 YAML，断言 model、data、logger、callbacks 的 `class_path` 和构造参数可实例化；不复制一套 YAML 解析器做测试。
3. override 测试：传入 `--data.init_args.batch_size=4` 等覆盖，断言保存候选配置包含最终值。
4. resolved config 测试：首次运行写入、相同配置复用、任一 model/data/trainer/logger/callback 字段变化时拒绝，并断言错误包含路径。
5. DataModule 测试：mock 模块内的 `load_dataset`，断言它是唯一加载入口且参数映射准确；本地 smoke fixture 也通过 `load_dataset("json"|"parquet"|"audiofolder", ...)` 加载。
6. 模型来源测试：分别解析 Scratch、Full、LoRA YAML，断言构造签名只接受所属来源字段，互斥字段由 LightningCLI 报错；详细要求见 [模型初始化来源边界](model-initialization.md)。
7. 边界测试：Task、DataModule、criterion、checkpoint callback 和 export 继续各自单测，不通过 CLI 测试间接替代。
8. smoke test：使用 tiny fake Task/DataModule、CPU、`fast_dev_run=true`，不得加载真实大模型、GPU、网络或远程数据集。

统一通过 Poetry 执行项目已有格式化、静态检查和 pytest；至少运行受影响测试，再运行完整测试集。
