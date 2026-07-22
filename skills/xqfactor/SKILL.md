---
name: xqfactor
description: Use xqfactor to define data-source-independent Pandas factors, create LeafFactor resolvers for RQData or other sources, compose built-in and custom DataFrame operators, evaluate expression graphs with explicit contexts and caches, and run IC, quantile-return, regression, or custom analyses. Trigger when an agent needs to integrate, extend, test, or diagnose this xqfactor project.
---

# xqfactor

## 目标

使用本项目定义数据源无关的因子表达式。把外部取数放在应用项目的 `LeafFactor`
resolver 中；把表达式组合、历史需求、轴对齐、递归求值和执行缓存交给 xqfactor。

所有因子节点统一传递 `pandas.DataFrame`：

- index：时间；
- columns：资产；
- 逻辑形状：`(时间数, 资产数)`。

不要创建新的计算后端。Polars、PyTorch 等库只在确有需要的某个自定义算子内部使用，
并在算子入口和出口保持 Pandas DataFrame 契约。

## 开始使用

1. 阅读仓库根目录 `AGENTS.md`。
2. 查看 `src/xqfactor/__init__.py`，确认公共 API。
3. 阅读 `references/examples.md` 中的完整示例。
4. 在应用项目安装依赖：

```bash
uv add xqfactor
uv add "xqfactor[analysis]"
```

`rqdatac`、Polars、PyTorch 等依赖由使用它们的应用项目自行安装。

## 核心对象

- `LeafFactor(name, resolver, definition_version="1")`：定义应用拥有的基础因子取数。
- `EvaluationContext`：声明完整计算时间轴、universe、频率、输出切片、数据语义和
  provider 版本。
- `CombinedFactor`：将一个或多个因子 DataFrame 交给自定义计算函数。
- `RollingWindowFactor`：声明窗口长度并传播历史周期需求。
- `CombinedRollingWindowFactor`：将多个因子 DataFrame 交给带窗口长度的自定义计算函数，
  回调形式为 `func(window, *values)`。
- `FixedFactor`：在指定资产的单标的上下文中计算因子并广播到当前 universe。
- `MemoryCache`：在完全相同的因子定义和执行上下文下复用叶子及中间结果。
- `ExecutionCache`：应用替换缓存实现时遵守的最小协议。
- `AbstractFactor.required_history()` 和 `required_future()`：分别声明输出区间前后
  需要的历史与未来周期数。
- `get_defined_factor_periods()`：返回当前存活因子实例的最大历史和未来周期需求。
- `EvaluationContextBuilder`：数据源实现统一遵守的上下文构造协议。
- `RQDataContextBuilder`：位于 `xqfactor.providers.rqdata`，生成中国股票日、分钟、
  周、月频 `EvaluationContext`。

`time_index` 是完整计算轴，必须包含 `REF`、收益率或窗口算子所需的历史和未来数据。
`output_start` 和 `output_end` 只裁剪最终结果；历史前缀必须位于 `output_start` 之前，
未来尾部必须位于 `output_end` 之后。若任一侧不足，`evaluate()` 会抛出 `ValueError`；
resolver 原本返回的 `NaN` 会保留。

## 定义叶子因子

外部 I/O 全部通过 resolver：

```python
CLOSE = LeafFactor("close", load_close, definition_version="rqdata-post-v1")
```

resolver 接收 `LeafRequest` 并返回 DataFrame。返回前尽量按
`request.context.time_index` 和 `request.context.universe` 对齐；核心仍会再次
`reindex`。数据口径变化时更新 `definition_version` 或
`EvaluationContext.provider_version`，避免复用旧缓存。

不要把 RQData、数据库、Parquet 或 DuckDB 逻辑加入 xqfactor 核心。完整的 RQData
后复权 `CLOSE` 示例见 `references/examples.md`。

需要按当前表达式需求构造 RQData 上下文时，先定义因子，再显式传入汇总结果：

```python
from xqfactor.providers.rqdata import RQDataContextBuilder


periods = get_defined_factor_periods()
context_builder = RQDataContextBuilder()
context = context_builder.build(
    start_date="2025-01-01",
    end_date="2025-06-30",
    universe=("000001.XSHE", "600000.XSHG"),
    market="cn",
    type="stock",
    frequency="D",
    history_period=periods.max_history,
    future_period=periods.max_future,
)
```

周期汇总只观察当前仍存活的因子实例；上下文构造函数不会自动读取汇总结果。
`rqdatac` 仍由应用安装和初始化，`RQDataContextBuilder` 只在实际构造时延迟导入。
公共 `frequency` 必须使用 Pandas 规范 `freqstr`，例如 `D`、`min`、`W-SUN`、
`ME`；数据源自身的 `1d`、`1m`、`1w` 由 provider 或 resolver 映射。

## 使用与定义算子

直接使用 `PCT_CHANGE`、`REF`、`RANK`、`NORM`、`MAD`、`IF`、
`CSNEUTRALIZER`、`FIX` 等内置算子：

```python
RETURNS = PCT_CHANGE(CLOSE, 1)
ALPHA = RANK(RETURNS) * -1
```

`REF(X, n)` 的正数表示过去值，负数表示未来值；`REF(X, -1)` 将 `t+1` 的值对齐
到 `t`。因子表达式分别通过 `required_history()` 和 `required_future()` 声明两侧
依赖，不能把未来周期计入历史需求。

使用 `FIX` 将任意因子固定到一个公共资产，再广播到当前 universe：

```python
CSI500_RETURNS = FIX(RETURNS, "000985.XSHG")
EXCESS_RETURNS = RETURNS - CSI500_RETURNS
```

`FIX` 会在 `universe=(目标资产,)` 的子上下文中求值，被固定表达式中的横截面算子也
因此只作用于这个单标的上下文；它适合指数、基准等不应随研究股票池变化的公共类因子。

自定义算子采用两层定义：

1. 与具体因子无关的 DataFrame 计算函数；
2. 接收任意因子并返回组合节点的构造函数。

```python
def cross_sectional_demean(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sub(frame.mean(axis=1), axis=0)


def DEMEAN(factor: AbstractFactor) -> CombinedFactor:
    return CombinedFactor(cross_sectional_demean, factor)
```

不要创建 `custom_unary(factor, ...)` 这类把算子定义绑定到某个因子实例的 API。
窗口算子用 `RollingWindowFactor(function, window, factor)` 或
`CombinedRollingWindowFactor(function, window, *factors)` 明确声明历史需求；多因子窗口
回调接收完整计算轴上的多个 DataFrame，输入形状均为 `(时间数, 资产数)`。

## 执行与缓存

```python
cache = MemoryCache(maxsize=256)
value = RETURNS.evaluate(context, cache)
```

复用同一个 cache，可以共享叶子和中间节点结果。universe、时间轴、频率、输出切片、
semantics、provider 版本、resolver 版本或因子定义变化都会形成不同缓存键。

xqfactor 的缓存不是市场数据仓库。全市场 Parquet、DuckDB、Redis 或远程缓存由应用
项目负责；如需替换会话执行缓存，只实现 `ExecutionCache` 协议。

## 因子检验

从对应领域模块导入具体检验器；`xqfactor.analysis` 只导出 `AbstractAnalyzer`。
标准化、去极值、中性化等预处理先表达为因子算子，不注册 Processor：

```python
from xqfactor.analysis.ic import ICAnalyzer


NORMALIZED_RETURNS = NORM(RETURNS)
result = ICAnalyzer(forward_returns).analyze(
    {"returns": NORMALIZED_RETURNS},
    context=context,
    cache=cache,
)
```

自定义检验器从 `xqfactor.analysis` 导入 `AbstractAnalyzer` 并实现
`_analyze(factors)`。传入的值已经求值为 Pandas DataFrame；检验器只实现统计逻辑。
IC、分组收益和回归分别位于 `analysis.ic`、`analysis.quantile_return` 和
`analysis.regression`。行业专用报告、绘图、基准归因和策略业务规则留在应用项目。

## 修改项目

- 新增内置算子时，在 `operators.py` 实现 DataFrame 计算函数和构造函数。
- 从 `xqfactor.__init__` 导出公共算子。
- 测试轴顺序、NaN、dtype、历史需求和缓存身份。
- 公共 API 变化后同步更新本 Skill 和详细示例。
- 不重新引入 `FactorRuntime`、计算后端注册表、Processor 或数据源专用叶子因子。

## 验证

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -m compileall -q src
uv build
```

修改 Skill 后还要运行：

```bash
uv run --with pyyaml python /Users/xuanqi/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/xqfactor
```

## 参考

`references/examples.md` 包含：

- RQData 后复权 `CLOSE`；
- `RETURNS = PCT_CHANGE(CLOSE, 1)`；
- Polars 和 PyTorch 自定义算子；
- 标准化后进行 IC 检验；
- 自定义检验器；
- 自定义窗口算子和缓存注意事项。
