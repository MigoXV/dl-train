# 深度学习训练框架技能

该技能用于指导企业级深度学习训练框架的设计、实现和审查，覆盖配置、数据管道、任务、模型、检查点、导出、测试与交付规范。

## Python 版本约束

默认 Python 版本约束为 `>=3.10,<3.13`。创建 Poetry 环境时默认使用 `python3.10` 解释器：

```bash
poetry env use python3.10
```

## Git Flow 工作流

- `main` 分支只作为稳定主线，不允许直接提交。
- `dev` 分支作为集成分支，不允许直接开发提交。
- 每次开发前必须从 `dev` 创建新的 `feature/<任务名>` 分支。
- 所有代码修改、测试和文档更新都必须在 feature 分支完成。
- feature 分支完成后必须通过 squash merge 合并回 `dev`，禁止普通 merge commit。
- 除非明确要求发布，不改动 `main`，也不从 `dev` 合并到 `main`。
- 合并前必须确认工作区干净，并运行项目最小验证命令。
- 如果需要提交，提交信息要简洁说明本次变更目的。
- 如果当前不在 `dev` 或 `feature/<任务名>` 分支，先检查分支状态，不要盲目提交。

## torch 生态依赖约束

`torch` 以及依赖 `torch` 的包不能写入项目 `pyproject.toml`，包括主依赖、开发依赖和默认 extras。最终 lock 文件中不应该出现 `torch` 作为直接或传递依赖；只有明确声明为 optional 且不会进入默认安装路径时才允许例外。

以下包只能通过 pip 安装，并且项目 README 必须写明安装命令：

- `torch`
- `torchvision`
- `torchaudio`
- `datasets`
- `modelscope`
- 其他 torch 生态或可能引入 torch 传递依赖的包

推荐在 Poetry 环境创建后安装：

```bash
poetry env use python3.10
poetry install --extras dev
poetry run pip install "torch==2.8.*" "torchvision==0.23.*" "torchaudio==2.8.*" datasets modelscope
```

默认安装 torch 2.8 系列。实际项目应根据 CPU/CUDA 版本调整 pip 命令和索引地址，但不要把这些依赖交给 Poetry 解析。
