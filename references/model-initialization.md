# 模型初始化来源边界

实现或审查 Scratch、Full Finetune、LoRA Finetune、checkpoint 恢复和导出时读取本文。

## 目录

- [Task 继承结构](#task-继承结构)
- [Scratch](#scratch)
- [Full Finetune](#full-finetune)
- [LoRA Finetune](#lora-finetune)
- [统一边界](#统一边界)
- [测试](#测试)

## Task 继承结构

定义不含模型来源参数的公共 `BaseTask(LightningModule)`，集中实现 training/validation step、指标聚合、batch key 过滤、optimizer 和 scheduler。定义三个具体子类，各自用显式构造签名加载模型；禁止把两个路径设为 `Optional` 后在一个类内分支，也禁止 `**kwargs` 吞掉未知参数。

```python
class ScratchTask(BaseTask):
    def __init__(self, model_config_path: str, criterion: Criterion, lr: float): ...

class FullFinetuneTask(BaseTask):
    def __init__(self, pretrained_model_path: str, criterion: Criterion, lr: float): ...

class LoraFinetuneTask(BaseTask):
    def __init__(self, pretrained_model_path: str, criterion: Criterion, lora_config: LoraConfig, lr: float): ...
```

Full 与 LoRA 可复用 `models/` 内“从完整仓库加载”函数，但不得通过继承一个暴露多余参数的构造函数实现复用。每个具体 Task 显式调用 `save_hyperparameters()`，只保存自身来源字段。

## Scratch

- 只接受 `model_config_path`；它必须指向项目 model-zoo 的配置仓库。
- 仓库必须包含 `config.json`、任务要求的 CMVN，以及必要 tokenizer/processor/generation 配置资源。
- 仓库不得包含 `model.safetensors`、`pytorch_model.bin`、对应 index/shard、adapter 权重或项目自定义权重文件；发现权重立即报错并列出文件。
- 用 `AutoConfig.from_pretrained(model_config_path)` 读取配置，再用任务型 `AutoModel.from_config(config)` 创建随机初始化模型；不得调用模型的 `from_pretrained`。
- processor 和 CMVN 也只能从 `model_config_path` 读取。

## Full Finetune

- 只接受 `pretrained_model_path`，不得接受或推导第二个 `model_config_path`。
- 路径必须是完整 Hugging Face 模型仓库：包含 `config.json`、任务要求的 CMVN/processor 资源，以及 `save_pretrained` 兼容的完整模型权重或权重 index/shards。
- 配置、CMVN、processor 和权重全部从同一个 `pretrained_model_path` 加载；禁止从 model-zoo 补配置或 CMVN。
- 用任务型 `AutoModel.from_pretrained(pretrained_model_path)` 做真实加载，并确保全部应训练参数保持可训练。

## LoRA Finetune

- 只接受 `pretrained_model_path` 和 LoRA 自身配置，不得接受 `model_config_path` 或通用 `use_lora` 模式开关。
- 先按 Full 规则校验并通过 `AutoModel.from_pretrained(pretrained_model_path)` 加载完整基础模型。
- 再冻结基础模型并调用 PEFT `get_peft_model` 注入 adapter；验证只有约定 adapter/bias 等参数可训练。
- `pretrained_model_path` 指向基础模型完整仓库，不指向仅含 adapter 的仓库；训练恢复所需 adapter 状态来自 checkpoint。

## 统一边界

- YAML 的 `model.class_path` 必须选择具体 Task；`init_args` 只出现该 Task 的来源字段。
- CLI 依赖 LightningCLI 的构造签名拒绝互斥字段；不得自行 pop、忽略或兼容 `model_path`。
- resolved config 和 checkpoint hyperparameters 只记录具体 Task 的来源字段；恢复时校验 Task 类型、字段和仓库内容一致。
- 导出 API 按任务拆分或使用具体类型分派：Scratch 只接受 `model_config_path`，Full/LoRA 只接受 `pretrained_model_path`。Scratch/Full 导出完整 HF 仓库；LoRA 明确导出 adapter 或显式合并后的仓库，并保留基础仓库引用。
- repository validator 放在 `models/` 或独立公共模块；它只校验资源边界，不读取 YAML、不承担训练或导出。

## 测试

1. 用 `inspect.signature` 精确检查三个具体 Task 和导出函数；断言不存在 `model_path`、`**kwargs` 和互斥来源字段。
2. 构造本地 tiny model-zoo：包含真实 `config.json` 和 CMVN；Scratch 通过 `AutoConfig.from_pretrained` + 任务型 `AutoModel.from_config` 实例化，加入任一权重文件后必须失败。
3. 用 tiny Transformers 模型 `save_pretrained` 创建完整本地仓库并加入 CMVN；Full/LoRA 必须真实 `from_pretrained`，缺配置、CMVN、权重或 shard 时分别失败，不用 mock 代替加载。
4. 检查 Scratch 是随机初始化；Full 权重与保存前一致；LoRA 基础权重已加载且冻结、adapter 已注入且仅约定参数可训练。
5. 对三套 YAML 传入互斥字段，断言 LightningCLI 在实例化前报错且错误指出字段。
6. 分别保存并恢复真实 Lightning checkpoint，检查 Task 类型、来源字段、模型/adapter 权重、optimizer、scheduler 和 global step。
7. 分别执行 Scratch、Full、LoRA 导出并轻量重新加载，检查 config、CMVN/processor、权重或 adapter、基础仓库引用完整。

测试只使用本地 tiny 模型和 CPU，不访问网络；“真实加载”指实际调用 Transformers/PEFT/Lightning API，而不是加载生产大模型。
