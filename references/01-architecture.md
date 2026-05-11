# 架构与目录规范

目标：用稳定边界承载模型、数据、训练、导出变化。

## 标准分层

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

- `commands/`：配置解析、override 合并、对象装配、启动训练或导出。
- `configs/`：schema、默认值、字段分组、兼容字段。
- `models/`：configuration/modeling/processing、底座兼容、`transformers` 注册。
- `criterions/`：loss、训练目标、多 loss 组合；模型稳定内置 loss 时可省略。
- `tasks/`：训练步骤、验证步骤、指标、optimizer、scheduler。
- `checkpointing/`：checkpoint 配置和 callback。
- `exporting/`：训练产物到推理或平台格式。

不要新增独立 `finetuning/` 杂物模块。

## 边界

- CLI 只解析配置、打印 resolved config、构造对象、调用 `trainer.fit(...)` 或导出函数；不读取样本、不拼 batch、不写 loss、不 patch 模型。
- Config 按 `model`、`lora/peft`、`optimizer`、`evaluation`、`data`、`checkpoint`、`logger`、`trainer` 分组。
- Model 类、配置、processor 必须使用 `transformers` 注册表；入口优先用 `AutoConfig`、`AutoModel`、任务型 AutoModel、`AutoProcessor`；支持字符串路径、HF repo、本地目录。
- `models/` 不依赖 `tasks/`、DataModule 或 CLI；任务相关、蒸馏、对比学习、多 loss 加权放 `criterions/`。
- DataModule 负责 dataset 加载、split、字段和标签校验、样本限制、map 预处理、缓存、DataLoader、collator、训练集增强；不计算 loss、不构造 optimizer、不保存 checkpoint。
- Criterion 默认放 `criterions/`，输入为模型输出和 batch 标签；不读配置、不加载数据、不创建模型，并且可单测。
- Task 通过 Auto 类或注册表加载模型/processor/适配器，调用模型 loss 或 criterion，实现 step、指标聚合、optimizer、scheduler、dtype cast、生成参数和标签 mask；不读配置文件，不创建 CLI，不管理导出目录。

## 新增任务流程

1. 在 `configs/<task>/config.py` 定义 schema。
2. 在 `models/` 实现或接入模型、configuration、processor，并完成 `transformers` 注册，保证可用字符串路径和 `AutoModel`/任务型 AutoModel 加载。
3. 在 `tasks/<task>/datamodule.py` 实现数据加载和校验。
4. 在 `criterions/` 实现任务 loss；若模型已内置 loss，则在 Task 中显式使用模型返回的 loss。
5. 在 `tasks/<task>/task.py` 或 `finetune.py` 实现 Task。
6. 在 `tasks/<task>/metrics.py` 实现指标。
7. 在 `commands/` 接入装配或新增子命令。
8. 在 `examples/<task>/` 提供最小 YAML 和脚本。
9. 在 `tests/` 覆盖配置映射、模型 Auto 加载、数据校验、criterion、训练步骤和 CLI 装配。

## 命名

- Python 包内用小写下划线。
- 类名包含任务和职责，例如 `TextClassifyDataModule`。
- 配置类按分组命名，例如 `ModelConfig`、`DataConfig`。
- checkpoint monitor 与 Task log 名一致，例如 `val_loss`。
