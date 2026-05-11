# 配置系统规范

目标：同时支持本地调试、平台训练和实验复现。

## 合并顺序

1. typed schema 默认值。
2. YAML 配置文件。
3. 命令行 dotlist override。
4. 运行时派生字段。

YAML 环境变量在解析时 resolve；训练启动前记录 resolved config。

## Schema

- 所有公开配置字段必须进入 schema。
- 类型准确，不用字符串承载布尔、数字或列表。
- 默认值保守，能跑本地 smoke test。
- 可选字段用 `None`，不用空字符串。
- 平台路径、实验名、设备数可通过环境变量覆盖。

```python
@dataclass
class OptimizerConfig:
    lr: float = 2e-5
    weight_decay: float = 0.0
```

## YAML 与 Override

YAML 按配置组书写，示例配置可用环境变量但必须给默认值；生产配置不写个人路径或敏感明文。

```yaml
model:
  path: ${oc.env:MODEL_PATH,model-name}
data:
  batch_size: ${oc.env:BATCH_SIZE,32}
checkpoint:
  monitor: ${oc.env:CHECKPOINT_MONITOR,val_loss}
```

命令行 override 使用 dotlist：

```bash
python -m <package>.commands.app train examples/task/train.yaml data.batch_size=4
```

临时实验用 override；长期复用进入 YAML 或模板。

## 审计与映射

训练启动时记录配置文件、override、resolved config、trainer 核心参数、checkpoint 目录和 monitor、logger run name/hash。

CLI 显式映射配置到对象，不把全局 config 无边界传入模块：

```python
task = FinetuneTask(model_path=cfg.model.path, lr=cfg.optimizer.lr)
datamodule = DataModule(**OmegaConf.to_container(cfg.data, resolve=True))
```

新增字段时必须能判断属于 task、data、trainer、checkpoint、logger 还是 export。

## 空值与变更

- 构造第三方对象前删除 `None`，保留 `0`、`False`、空字符串。
- 新增字段同步 schema、示例 YAML、CLI 映射、至少一个测试。
- 删除或重命名字段必须写迁移说明；平台依赖旧字段时保留兼容期。
