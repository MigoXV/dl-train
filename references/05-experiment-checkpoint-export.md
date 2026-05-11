# 实验、检查点与导出规范

目标：分清训练过程产物和推理交付产物。

## 实验日志

训练启动时记录配置文件、override、resolved config、commit、模型路径、数据集路径和 split、checkpoint 目录、trainer summary。

运行中记录 train/validation loss、任务指标、learning rate、epoch、global step 和可用系统指标。

logger 配置独立成组；启用但依赖缺失时抛清晰错误，不静默降级。

## Checkpoint

推荐字段：

```yaml
checkpoint:
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

## Resume

恢复训练时明确输入是训练 checkpoint 而不是推理权重，并说明是否恢复 optimizer、scheduler、global step，数据版本和配置是否与原 run 一致，是否继续写入原 output dir。

不要默认猜测最新 checkpoint；若 CLI 支持自动选择，必须记录选择结果。

## Export

常见格式：Lightning `.ckpt`、完整 `state_dict`/safetensors、LoRA adapter、Hugging Face repository layout、推理服务自定义格式。

导出命令必须声明支持格式；不支持时直接报错，不猜格式。

标准流程：读取权重 -> 校验 key/shape/dtype/metadata -> 读取模板或 base model -> 复制 tokenizer/processor/generation config -> 写目标权重 -> 更新说明 -> 轻量加载测试。

## 目录与指标

```text
outputs/
  runs/<run-id>/
  checkpoints/<run-id>/
  exports/<model-name>/
```

不要把中间 checkpoint 和最终导出物混在同一目录。

checkpoint monitor 选择贴近业务目标的验证指标：ASR 用 `val_wer`/`val_cer`，分类用 `val_accuracy`/`val_f1`/`val_auc`，检测用 `val_map`。指标成本高时可降低验证频率，但不能取消验证。

## 测试要点

- checkpoint config 参数透传，`None` 字段不传给第三方 callback。
- monitor 与 Task log 对齐。
- logger disabled 返回空 logger。
- export 拒绝不支持格式。
- export 复制必要非权重文件并可轻量加载。
