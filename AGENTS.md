# xqfactor 仓库开发规范

## 项目定位

`xqfactor` 是数据源无关的因子表达式、算子契约、执行缓存和检验规范框架。

- 核心包不得依赖 RQData、数据库、Parquet、DuckDB 或具体行情字段。
- 具体应用使用 `LeafFactor` 的 resolver 读取数据。
- 核心只负责表达式图、上下文、缓存、后端协议和求值流程。
- Pandas/NumPy 计算后端与 Pandas/SciPy/statsmodels 检验实现均为可选依赖。
- 不在因子定义中实现长期数据存储或跨 universe 数据集管理。

## 项目结构

- `src/xqfactor/factor.py`：因子节点、`LeafFactor`、表达式求值和历史窗口需求。
- `src/xqfactor/runtime.py`：`EvaluationContext`、`FactorValue`、运行时、缓存和算子规范。
- `src/xqfactor/operators.py`：内置算子与自定义算子构造工具。
- `src/xqfactor/backends/pandas.py`：可选 Pandas/NumPy 参考计算后端。
- `src/xqfactor/analysis/spec.py`：后端无关的检验规范。
- `src/xqfactor/analysis/pandas.py`：可选 Pandas 检验器和预处理器。
- `tests/`：不依赖真实在线数据服务的测试。
- `skills/xqfactor/`：供后续 Agent 使用本项目的技能说明和示例。

不得重新引入已删除的全局配置、全局数据 API 注册表或数据源专用叶子因子。

## 公共 API 约束

- 叶子因子统一使用 `LeafFactor(name, resolver, definition_version="1")`。
- resolver 接收 `LeafRequest`，返回 `FactorValue` 或后端可标准化的二维值。
- `EvaluationContext.time_index` 必须包含窗口算子和 `REF` 所需的历史区间。
- `output_start` 和 `output_end` 只控制最终输出切片，不控制叶子数据读取范围。
- 因子值逻辑形状固定为 `(时间, 资产)`；发生形状、index 或 columns 转换时必须注释。
- 缓存只在因子定义、完整上下文、provider 版本和 backend 版本一致时命中。
- 添加内置算子时，同时补充后端实现、公共导出和测试。
- 自定义算子优先使用 `custom_unary`、`custom_binary` 或 `rolling_operator`。

## 开发命令

安装依赖统一使用：

```bash
uv add <依赖名>
uv add --dev <开发依赖名>
```

运行脚本和检查统一使用：

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -m compileall -q src
uv build
```

临时工具使用 `uvx`。不得直接使用系统 Python 安装项目依赖。

## 编码规范

- 兼容 Python 3.12 及以上版本。
- 模块说明、类说明、函数说明、日志、帮助信息和代码注释使用中文。
- 函数必须有输入输出类型标注和中文 docstring，说明输入与返回值含义。
- 复杂逻辑块使用以下形式说明整体意图：

```python
# ************************************************************
# 说明该逻辑块的目标、关键约束和数据形状变化。
# ************************************************************
```

- Pandas、Polars、NumPy、PyTorch 数据形状变化必须说明转换前后形状。
- Pandas index 或 columns 发生变化时必须说明变化内容。
- 只修改当前需求直接涉及的代码，不顺手重构无关模块。
- 不为单次场景增加未要求的抽象、配置项或兼容层。

## 测试要求

- 使用 fake resolver 或内存数据，测试不得调用真实 RQData。
- 至少覆盖叶子取数、表达式组合、窗口需求、轴对齐和缓存隔离。
- 添加后端算子时覆盖 NaN、轴顺序、dtype 和边界输入。
- 修改检验器时覆盖预处理顺序和结果统计。
- 修改公共导出或依赖时必须运行 `uv build`。

## Skill 维护

- 项目使用方法维护在 `skills/xqfactor/SKILL.md`。
- 详细示例维护在 `skills/xqfactor/references/examples.md`，避免在 SKILL.md 重复。
- 公共 API 变化后同步更新技能示例并运行：

```bash
uv run --with pyyaml python /Users/xuanqi/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/xqfactor
```

## 提交规范

- 提交前运行完整测试、Ruff、compileall、技能校验和构建。
- 提交信息使用简短中文祈使式，例如 `重构数据源无关因子运行时`。
- 不提交 `.DS_Store`、缓存、构建产物、真实账号、token 或数据文件。

## 代码阅读路径

从应用创建 `LeafFactor` 开始，resolver 根据 `LeafRequest` 返回原始二维数据；
`operators.py` 将基础因子组合成表达式图；`factor.evaluate()` 进入
`FactorRuntime`，先查询 `MemoryCache`，未命中时递归计算子节点并调用具体后端；
最终后端根据 `EvaluationContext` 截取输出时间区间。检验流程从
`analysis/pandas.py` 的 `AbstractAnalyzer.analyze()` 开始，依次执行处理器后调用
具体检验器的 `_analyze()`。
