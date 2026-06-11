# 代码规范

目标：降低误改概率。优先边界清晰、类型明确、错误可定位、日志可审计、测试可覆盖。

## 基本规则

- 默认 Python 版本约束为 `>=3.10,<3.13`，默认使用 `python3.10` 解释器创建 Poetry 环境。
- 使用 Python 3.10 兼容的类型标注，新文件默认 `from __future__ import annotations`。
- 模块名、函数名、变量名用小写下划线，类名用 `PascalCase`。
- 文件只承担一个清晰职责。

## 依赖方向

- `commands/` 可依赖 config、task、checkpoint、exporting。
- `tasks/` 可依赖 models、criterions、公共工具、指标。
- `configs/` 不依赖训练实现。
- `checkpointing/` 不依赖具体任务。
- `exporting/` 不依赖 CLI。
- `models/` 不依赖任务训练代码。

禁止 import 时启动训练、加载大模型、读取大数据；禁止配置模块访问 GPU、文件系统或外部服务；禁止跨任务导入私有函数；禁止新增独立 `finetuning/` 杂物模块。

## 函数、类型、配置

- 输入参数显式，不依赖隐式全局状态。
- 返回值结构稳定，纯转换逻辑优先写成纯函数。
- 复杂流程拆成校验、转换、执行、记录。
- 公开配置必须有 schema，进入实现前 resolve。
- 第三方参数传递前删除 `None`，保留 `0`、`False`、空字符串。
- `Any` 只用于第三方对象边界。
- 路径参数对外可接收 `str | Path`，内部尽早转 `Path`。

## 异常与日志

- 错误信息必须指出具体配置字段、split、路径或权重 key。
- 不裸 `except Exception` 后静默跳过；可恢复问题 warning，不可恢复问题 raise。
- 可选依赖缺失时说明哪个功能需要哪个包。
- CLI 记录 resolved config、overrides、trainer summary。
- Task 记录指标，DataModule 记录数据规模/split/缓存路径，Export 记录输入/输出/模板/格式校验。
- 日志不得写 token、密钥或敏感路径。

## 依赖、注释、工具

- 核心训练依赖写入项目依赖，开发测试依赖放 dev extras；但 `torch` 及依赖 `torch` 的包除外。
- `torch`、`torchvision`、`torchaudio`、`datasets`、`modelscope` 等 torch 生态或可能引入 torch 的包不得写入 `pyproject.toml` 的主依赖、dev 依赖或 extras，除非该声明明确为 optional 且不会进入默认 lock。
- 交付前检查 lock 文件：除 optional 依赖声明外，不应出现 `torch` 作为直接或传递依赖。
- torch 生态依赖只能由用户按 README 说明使用 pip 安装，不在业务代码、Poetry scripts 或测试夹具中自动安装。
- 可选功能延迟 import，启用但缺失时抛清晰错误。
- 不在业务代码中自动安装依赖，不写个人镜像源或缓存路径。
- 代码能自解释时不写注释；对模型 patch、格式兼容、缓存 key、分布式指标写短说明。
- 推荐统一 `ruff format`/`black`、`ruff`、`pytest`，按项目复杂度选择 `mypy` 或 `pyright`。

## AI 修改规则

1. 先读相关模块和测试，不凭文件名猜边界。
2. 判断字段属于 config、model、data、criterion、task、checkpoint、export 还是 CLI。
3. 小步编辑，避免无关重构。
4. 新增行为同步补测试。
5. 运行最小验证命令。
6. 最终说明改动、验证和未验证项。
