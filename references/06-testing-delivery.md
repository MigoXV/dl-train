# 测试与交付规范

目标：测试工程边界，不证明模型效果。默认测试不得依赖真实大模型、GPU、网络或远程大数据集。

## 测试结构

```text
tests/
  commands/
  configs/
  tasks/
  checkpointing/
  exporting/
  utils/
```

## 测试矩阵

- 纯函数：文本标准化、音频加载、指标、dtype 解析。
- 配置：schema 默认值、YAML 合并、override 映射、`drop_none`。
- DataModule：fake dataset、字段校验、缓存 key、DataLoader 参数。
- Task：fake model、fake processor、loss、metric、decode、optimizer。
- CLI：mock Trainer/Task/DataModule，验证装配参数。
- Export：临时目录、小权重、模板文件复制、错误路径。

Fake 对象优先实现必要接口，例如 `FakeProcessor`、`FakeDatasetDict`、`FakeModel`、`FakeTrainer`。

## 必测行为

- 配置字段进入 schema、YAML、CLI 映射和测试。
- 启动日志包含 config path、overrides、resolved config。
- split 或字段缺失时早失败。
- sample limit、预处理参数、缓存 key 生效。
- 训练步骤返回 loss，验证步骤记录 loss 或聚合指标。
- 生成式指标不把标签作为生成输入。
- checkpoint monitor 与 Task log 名一致。
- export 支持格式成功，不支持格式明确失败。

## 运行命令

```bash
poetry install --extras dev
poetry run pytest -q
poetry run pytest -q -m gpu  # GPU 集成测试单独标记
```

## 示例、审查、交付

- 每个任务提供环境变量驱动 YAML、本地运行脚本、训练命令、smoke override 示例。
- 脚本应短小，复杂逻辑进入 CLI 或 Python 模块。
- 审查时确认分层清晰、新配置同步 schema/YAML/测试、数据校验早失败、指标与 checkpoint monitor 对齐、checkpoint 和 export 未混淆、错误信息能指导排查。
- 交付前确认格式化和测试通过、示例路径不含个人信息、日志能还原最终配置、runs/checkpoints/exports 目录清晰、未验证内容已列出。
