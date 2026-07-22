# xqfactor

`xqfactor` 是数据源无关、统一使用 `pandas.DataFrame` 传递因子值的因子表达式、
执行缓存和检验规范框架。

核心包不依赖 `xqdata`、RQData、数据库或本地数据仓库。应用项目通过
`LeafFactor` 的 resolver 实现实际取数；xqfactor 负责表达式组合、历史窗口需求、
轴对齐、递归求值和相同执行上下文下的缓存复用。

## 安装

```bash
uv add xqfactor
```

需要内置统计检验时：

```bash
uv add "xqfactor[analysis]"
```

## 基本用法

```python
import pandas as pd

from xqfactor import EvaluationContext, LeafFactor, LeafRequest, MemoryCache, RANK


def load_close(request: LeafRequest) -> pd.DataFrame:
    """由应用负责从 API、数据库或本地文件读取数据。"""
    return pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=request.context.time_index,
        columns=request.context.universe,
    )


CLOSE = LeafFactor("close", load_close)
factor = RANK(CLOSE)
context = EvaluationContext(
    time_index=("2025-01-01", "2025-01-02"),
    universe=("000001.SZ", "000002.SZ"),
    frequency="D",
)
cache = MemoryCache(maxsize=256)
result = factor.evaluate(context, cache)
```

## 使用未来收益

`REF(X, n)` 中正数 `n` 引用过去值，负数 `n` 引用未来值；因此
`REF(STOCK_RETURN, -1)` 会把 `t+1` 的收益对齐到 `t`。未来因子的依赖需求同时包含
`required_history()` 和 `required_future()`：

```python
from xqfactor import REF


FORWARD_RETURNS = REF(STOCK_RETURN, -1)
assert FORWARD_RETURNS.required_history() == 0
assert FORWARD_RETURNS.required_future() == 1
```

完整 `time_index` 必须在 `output_start` 前提供历史数据、在 `output_end` 后预留未来
数据，最终输出不应包含预留尾部：

```python
context = EvaluationContext(
    time_index=("t0", "t1", "t2", "t3"),
    universe=("000001.SZ",),
    frequency="D",
    output_end=3,
)
```

如果历史轴或未来轴不足，`evaluate()` 会抛出 `ValueError`，避免将边界缺失误判为有效
输出。resolver 原本返回的 `NaN` 会按原样保留。

## 固定公共类因子

使用 `FIX` 可以把任意因子表达式固定到指定资产，再广播到当前 universe。固定过程
在只包含目标资产的子上下文中求值，因此不会因为当前研究股票池变化而改变，适合指数或
基准收益等公共类因子。

```python
from xqfactor import FIX, PCT_CHANGE


RETURNS = PCT_CHANGE(CLOSE, 1)
CSI500_RETURNS = FIX(RETURNS, "000985.XSHG")
EXCESS_RETURNS = RETURNS - CSI500_RETURNS
```

## 自定义算子

自定义算子分为“与因子无关的 DataFrame 计算函数”和“表达式构造函数”两层：

```python
import pandas as pd

from xqfactor import AbstractFactor, CombinedFactor


def cross_sectional_demean(frame: pd.DataFrame) -> pd.DataFrame:
    """将每个时间截面的值减去截面均值。"""
    return frame.sub(frame.mean(axis=1), axis=0)


def DEMEAN(factor: AbstractFactor) -> CombinedFactor:
    """把横截面去均值逻辑应用到任意因子。"""
    return CombinedFactor(cross_sectional_demean, factor)
```

## 职责边界

- 本项目负责因子表达式图、Pandas/NumPy 基础算子、显式执行上下文和内存执行缓存。
- 具体基础因子、在线 API、DolphinDB、Parquet、DuckDB 和全量市场数据由应用项目负责。
- 执行缓存只复用完全相同上下文下的叶子数据和中间因子，不是本地数据仓库。
- Polars、PyTorch 等库可在某个自定义算子内部按需使用，但不形成独立计算后端。
- 标准化、去极值和中性化等预处理使用因子算子表达；检验器只负责统计分析。

## 代码阅读路径

从应用创建 `LeafFactor` 开始，resolver 根据 `LeafRequest` 返回二维 DataFrame；
`operators.py` 和应用自定义构造函数把基础因子组合成表达式图；
`factor.evaluate()` 递归查询 `MemoryCache`、计算子节点并统一对齐时间轴和资产轴，
最后按 `EvaluationContext` 截取输出区间。检验流程从
`analysis/base.py` 的 `AbstractAnalyzer.analyze()` 开始，将因子表达式和检验器附加输入
统一求值后，交给 `ic.py`、`quantile_return.py` 或 `regression.py` 中的具体检验器统计。
