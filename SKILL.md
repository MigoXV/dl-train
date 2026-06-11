---
name: deep-learning-training-framework
description: Use this skill when designing, implementing, reviewing, or extending an enterprise deep learning training framework. It covers src package layout, typed YAML configuration, CLI entrypoints, task/datamodule separation, Lightning-style training, experiment logging, checkpointing, export, code standards, and tests.
---

# 深度学习训练框架规范

用于搭建、扩展或审查企业级训练框架。核心目标：配置可追踪，数据可验证，训练可恢复，实验可复现，模型可交付，测试可自动化。

## 结构

```text
src/<package>/
  commands/        # CLI 装配
  configs/         # typed schema 与默认值
  tasks/           # LightningModule、DataModule、指标
  models/          # transformers 注册、自定义模型、processor
  criterions/      # loss 与训练目标
  checkpointing/   # checkpoint callback
  exporting/       # 权重与交付格式导出
examples/
tests/
```

## 硬约束

- `commands/` 只解析配置和装配对象。
- `configs/` 只定义 schema、默认值和分组。
- `models/` 必须接入 `transformers` 注册表，支持字符串路径和 Auto 类加载。
- 默认提供 `criterions/`；只有模型稳定返回完整 loss 时才可省略。
- `tasks/` 连接模型、batch、criterion、指标和 optimizer。
- `checkpointing/` 与 `exporting/` 独立于 CLI。

## AI 执行流程

1. 先读 `README.md`、`pyproject.toml`、`src/`、`examples/` 和相关测试。
2. 将当前skill的`.gitignore`挪到项目根目录，如果已经有了则直接覆盖。
3. 先检查 git 分支状态；只能在 `feature/<任务名>` 分支完成代码修改、测试和文档更新。
4. 判断改动属于 config、model、data、criterion、task、checkpoint、export、CLI 还是 test。
5. 新增任务顺序：schema -> model -> data -> criterion -> task -> metrics -> command -> example -> tests。
6. 保持最小改动，沿用现有模式。
7. 新增行为同步补测试并运行最小验证命令。
8. 合并前确认工作区干净；feature 分支只能通过 squash merge 合并回 `dev`。

## 质量门槛

- schema、YAML、环境变量、override 能合成 resolved config。
- 训练前验证 split、字段、样本数、标签和缓存参数。
- checkpoint、resume、export 边界清晰。
- 日志能还原配置、override、trainer 参数、指标和输出路径。
- 默认测试不依赖真实大模型、GPU、网络或远程大数据集。
- 默认 Python 版本约束为 `>=3.10,<3.13`，默认使用 `python3.10` 解释器创建 Poetry 环境。
- `torch` 及依赖 `torch` 的包不得写入 `pyproject.toml`，最终 lock 文件不得出现 `torch` 传递依赖；确需声明时只能作为 optional 依赖。
- `torch`、`torchvision`、`torchaudio`、`datasets`、`modelscope` 等 torch 生态或可能引入 torch 的包只能通过 pip 安装，并在项目 README 中写明安装命令；默认安装 torch 2.8 系列。`transformers`不会引入torch依赖，可以通过`poetry add`安装。
- 遵循 Git Flow：`main` 只作为稳定主线，`dev` 只作为集成分支；不得直接在 `main` 或 `dev` 提交开发改动。
- 每次开发前从 `dev` 创建 `feature/<任务名>` 分支；完成后用 squash merge 合并回 `dev`，禁止普通 merge commit；除非明确要求发布，不从 `dev` 合并到 `main`。

## 详细规范

- [架构与目录规范](references/01-architecture.md)
- [配置系统规范](references/02-configuration.md)
- [数据管道规范](references/03-data-pipeline.md)
- [训练任务规范](references/04-training-task.md)
- [实验、检查点与导出规范](references/05-experiment-checkpoint-export.md)
- [代码规范](references/07-code-style.md)

## 禁止事项

- 不要把数据读取、训练、日志、导出堆进一个脚本。
- 不要只提供 notebook 作为生产入口。
- 不要让配置字段只存在 YAML 中。
- 不要混用训练 checkpoint 和推理权重。
- 不要写入个人路径、token 或密钥。
