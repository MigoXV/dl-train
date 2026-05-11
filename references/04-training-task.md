# 训练任务规范

目标：Task 只连接模型、batch、loss、指标和优化器，不混入数据加载、CLI 或导出。

## 基本结构

推荐使用 LightningModule 或等价抽象：

```python
class FinetuneTask(pl.LightningModule):
    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train_loss", loss)
        return loss
```

## 初始化与 Loss

Task 初始化应加载模型和 processor、注册自定义模型类、选择 dtype、应用 LoRA/PEFT、设置 generation config 或 pad token、保存 hparams。

Task 初始化不读 YAML、不加载训练数据、不创建 DataLoader、不写 checkpoint。

- 优先使用模型稳定返回的 `loss`。
- 任务相关 loss 使用 `criterions/`。
- `training_step` 只筛选 batch key、前向、记录 loss、返回 loss。
- 不在 `training_step` 中做大量预处理。

## dtype 与 LoRA

- dtype 推荐支持 `auto`、`bf16`、`fp16`、`fp32`。
- 浮点输入可按模型 dtype cast；token id、attention mask、labels 等整数张量不得 cast 成浮点。
- Trainer precision 与模型加载 dtype 必须兼容。
- LoRA/PEFT 配置独立成组，仅启用时导入 PEFT。
- target modules 支持逗号分隔字符串或列表。
- optimizer 只优化 `requires_grad=True` 参数。

## 验证、指标、优化器

- Task 记录 `train_loss`、`val_loss` 和任务指标，例如 `val_wer`、`val_accuracy`。
- loss 类指标可逐 step 记录；生成或全局指标先收集 prediction/reference，再在 epoch end 聚合。
- 生成式评估使用 prompt-only 输入，不把标签传给 `generate`。
- 生成参数进入 evaluation 配置，decode 后统一标准化再算指标。
- 默认 `AdamW`；scheduler 使用训练总步数，warmup 用 ratio 或 steps，interval 明确为 `step` 或 `epoch`。
- `estimated_stepping_batches <= 0` 时可只返回 optimizer。

## 错误与测试

- 模型不兼容时早失败，错误信息说明缺失结构或不支持原因。
- CLI 记录完整配置和 trainer summary，Task 不重复打印。
- 测试覆盖 dtype、LoRA target、batch key 过滤、loss 返回、生成式指标不泄漏标签、optimizer 可训练参数。
- 测试关键工具逻辑时不要加载真实大模型。
