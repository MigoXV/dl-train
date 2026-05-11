# 数据管道规范

目标：把外部数据变成可验证、可缓存、可复现的训练 batch。

## DataModule

DataModule 应包含 dataset 加载、split 选择、字段/标签/样本数校验、样本限制、map 预处理、缓存命名、DataLoader 构造、collator 注入、训练集增强。

DataModule 不应包含 loss、optimizer、checkpoint、CLI 配置解析。

## 数据校验

训练前必须校验：

- train/validation/test split 存在。
- 每个 split 包含必需字段。
- 音频、图像、文本字段符合任务预期。
- 标签非空且格式正确。
- 样本限制后仍有数据。

错误信息直接指出缺失项，例如 `Dataset split validation is missing required columns: audio, text`。

## 预处理与缓存

- 原始数据不可变，预处理结果写新字段。
- 预处理函数尽量是纯函数。
- 文本标准化、标签模板、采样率、resize、多进程参数必须可配置。
- 训练和验证规则默认一致；差异必须显式说明。

缓存 key 至少包含原始 fingerprint、split、样本限制、prompt/prefix、标准化参数、确定性预处理参数、预处理版本号。

缓存目录优先级：显式 `preprocess_cache_dir` -> 本地数据集目录下项目缓存 -> 数据集库默认管理。预处理逻辑变化时提升 cache payload `version`。

## Collator

Collator 负责样本到模型输入：

- 加载和标准化原始模态。
- 调用 processor/tokenizer。
- padding、truncation、labels。
- mask 不参与 loss 的 token。
- 保留评估 reference 和 prompt-only 输入。

Collator 应可单测，不依赖 Trainer。

## 增强与 DataLoader

- 增强默认关闭，只在训练 split 启用；验证和测试禁用随机增强。
- 增强参数全部进入配置，高风险增强要有明确范围。
- DataLoader 推荐配置 `batch_size`、`num_workers`、`pin_memory`、`persistent_workers`、`prefetch_factor`、`shuffle`。
- `num_workers=0` 时不传 `persistent_workers=True`；`prefetch_factor` 只在 `num_workers>0` 时传。
- 小样本测试必须能用 `num_workers=0`；训练集 shuffle，验证集不 shuffle。

## 测试要点

- 正常 setup。
- split 或字段缺失时报清晰错误。
- sample limit、预处理参数、缓存 key 生效。
- 增强只作用于训练集。
- `num_workers=0` 可构造 DataLoader。
