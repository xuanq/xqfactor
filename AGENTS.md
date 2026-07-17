# xqfactor 仓库开发规范

## 项目定位

`xqfactor` 是数据源无关、统一使用 Pandas DataFrame 的因子表达式、执行缓存和检验规范框架。

- 核心包不得依赖 RQData、数据库、Parquet、DuckDB 或具体行情字段。
- 具体应用使用 `LeafFactor` 的 resolver 读取数据。
- 核心负责表达式图、Pandas/NumPy 基础算子、上下文、缓存、轴对齐和求值流程。
- 不区分时序因子节点和横截面因子节点；全部算子输入输出均为 `(时间, 资产)` DataFrame。
- Polars、PyTorch 等库只在具体自定义算子内部按需使用，不建立多计算后端体系。
- 不在因子定义中实现长期数据存储或跨 universe 数据集管理。

## 项目结构

- `src/xqfactor/factor.py`：因子节点、`LeafFactor`、表达式求值和历史窗口需求。
- `src/xqfactor/runtime.py`：`EvaluationContext`、`LeafRequest` 和执行缓存协议。
- `src/xqfactor/operators.py`：基于 Pandas/NumPy 的内置算子。
- `src/xqfactor/analysis/base.py`：`AbstractAnalyzer` 和统一输入求值流程。
- `src/xqfactor/analysis/ic.py`：IC 检验器和结果。
- `src/xqfactor/analysis/quantile_return.py`：分组收益检验器和结果。
- `src/xqfactor/analysis/regression.py`：依赖 SciPy/statsmodels 的回归检验器和结果。
- `tests/`：不依赖真实在线数据服务的测试。
- `skills/xqfactor/`：供后续 Agent 使用本项目的技能说明和示例。

不得重新引入全局配置、全局数据 API 注册表、数据源专用叶子因子、`FactorRuntime`、
计算后端注册表或 Processor 预处理体系。

## 公共 API 约束

- 叶子因子统一使用 `LeafFactor(name, resolver, definition_version="1")`。
- resolver 接收 `LeafRequest`，返回 Pandas DataFrame。
- DataFrame 的 index 为时间，columns 为资产；核心按上下文轴执行 `reindex`。
- `EvaluationContext.time_index` 必须包含窗口算子和 `REF` 所需的历史区间。
- `output_start` 和 `output_end` 只控制最终输出切片，不控制叶子数据读取范围。
- 因子值逻辑形状固定为 `(时间, 资产)`；发生形状、index 或 columns 转换时必须注释。
- 缓存只在因子定义、完整上下文和 provider 版本一致时命中。
- 添加内置算子时，同时补充实现、公共导出和测试。
- 自定义算子采用两层定义：纯 DataFrame 计算函数，以及返回
  `CombinedFactor`、`UnaryCombinedFactor`、`BinaryCombinedFactor` 或
  `RollingWindowFactor` 的构造函数。
- 标准化、去极值、中性化和掩码等输入输出仍为因子的操作统一使用算子，不增加 Processor。
- 本项目只内置数据源无关的通用统计检验；行业专用报告、绘图、基准归因和策略业务规则由应用实现。

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
- 添加算子时覆盖 NaN、轴顺序、dtype 和边界输入。
- 修改检验器时覆盖因子求值顺序和结果统计。
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
- 提交信息使用简短中文祈使式，例如 `简化因子执行与检验抽象`。
- 不提交 `.DS_Store`、缓存、构建产物、真实账号、token 或数据文件。

## 代码阅读路径

从应用创建 `LeafFactor` 开始，resolver 根据 `LeafRequest` 返回原始二维 DataFrame；
`operators.py` 将基础因子组合成表达式图；`factor.evaluate()` 使用传入或临时创建的
`MemoryCache`，未命中时递归计算子节点，统一对齐 `EvaluationContext` 的时间轴和资产轴，
最后截取输出时间区间。检验流程从 `analysis/base.py` 的
`AbstractAnalyzer.analyze()` 开始，统一求值主因子和附加输入，再调用领域模块中具体
检验器的统计逻辑。
