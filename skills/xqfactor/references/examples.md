# xqfactor 使用示例

## 依赖与初始化

在使用 xqfactor 的应用项目中安装依赖：

```bash
uv add "xqfactor[analysis]"
uv add rqdatac
uv add polars
uv add torch
```

`rqdatac`、Polars 和 PyTorch 都是应用依赖，不要加入 xqfactor 核心依赖。

```python
import rqdatac

from xqfactor import MemoryCache


rqdatac.init()
cache = MemoryCache(maxsize=256)
```

## 使用 RQData 定义后复权 CLOSE

RQData 的 `get_price` 使用 `adjust_type="post"` 获取后复权行情。
`expect_df=True` 时，多标的日线结果通常使用
`(order_book_id, date)` MultiIndex。

```python
from typing import Any

import pandas as pd
import rqdatac

from xqfactor import LeafFactor, LeafRequest


def _to_rq_frequency(frequency: str) -> str:
    """将应用频率转换为 rqdatac.get_price 支持的频率。

    输入：EvaluationContext 中的频率字符串。
    输出：RQData 使用的频率字符串。
    """
    return {
        "D": "1d",
        "W-SUN": "1w",
        "min": "1m",
        "ME": "1d",
    }.get(frequency, frequency)


def load_close(request: LeafRequest) -> pd.DataFrame:
    """读取后复权收盘价并转换为时间乘资产的二维表。

    输入：包含完整计算时间轴、universe 和频率的叶子请求。
    输出：index 为完整时间轴、columns 为 universe 的 DataFrame。
    """
    context = request.context
    semantics: dict[str, Any] = dict(context.semantics)
    data = rqdatac.get_price(
        order_book_ids=list(context.universe),
        start_date=context.time_index[0],
        end_date=context.time_index[-1],
        frequency=_to_rq_frequency(context.frequency),
        fields="close",
        adjust_type="post",
        skip_suspended=False,
        expect_df=True,
        market=semantics.get("market", "cn"),
    )

    if data is None or data.empty:
        return pd.DataFrame(
            index=pd.Index(context.time_index, name="datetime"),
            columns=context.universe,
            dtype=float,
        )

    # ************************************************************
    # RQData 结果从 MultiIndex Series：
    # (order_book_id, date/datetime) -> value
    # 转换为 DataFrame (时间数, 资产数)：
    # index=date/datetime，columns=order_book_id。
    # ************************************************************
    close = data["close"].unstack("order_book_id")
    close.index = pd.to_datetime(close.index)
    return close.reindex(
        index=pd.DatetimeIndex(context.time_index),
        columns=context.universe,
    )


CLOSE = LeafFactor(
    name="close",
    resolver=load_close,
    definition_version="rqdata-post-v1",
)
```

如果使用港股，把上下文语义设置为 `(("market", "hk"),)`；resolver 会把它传给
`rqdatac.get_price`。

## 组合 RETURNS 因子并执行

`PCT_CHANGE(X, n)` 的语义是 `X / REF(X, n) - 1`。`REF(X, n)` 中正数表示过去值，
负数表示未来值；例如 `REF(STOCK_RETURN, -1)` 会把下一期收益对齐到当前期。

```python
import pandas as pd

from xqfactor import EvaluationContext, PCT_CHANGE


RETURNS = PCT_CHANGE(CLOSE, 1)

# time_index 包含 2025-01-02 这一期历史值。
# output_start=1 表示最终结果从 2025-01-03 开始返回。
context = EvaluationContext(
    time_index=tuple(
        pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
                "2025-01-07",
            ]
        )
    ),
    universe=("000001.XSHE", "600000.XSHG"),
    frequency="D",
    output_start=1,
    semantics=(("market", "cn"), ("adjust_type", "post")),
    provider_version="rqdata-2025-07",
)

returns_df = RETURNS.evaluate(context, cache)
```

未来收益需要在完整时间轴尾部预留未来数据，并通过 `output_end` 排除预留行；其他
历史窗口则需要在 `output_start` 前预留对应历史周期：

```python
from xqfactor import REF


FORWARD_RETURNS = REF(RETURNS, -1)
assert FORWARD_RETURNS.required_history() == 0
assert FORWARD_RETURNS.required_future() == 1

forward_context = EvaluationContext(
    time_index=tuple(
        pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
                "2025-01-07",
            ]
        )
    ),
    universe=("000001.XSHE", "600000.XSHG"),
    frequency="D",
    output_end=3,
)
forward_returns_df = FORWARD_RETURNS.evaluate(forward_context, cache)
```

如果 `time_index` 在 `output_start` 前或 `output_end` 后没有足够的周期，`evaluate()`
会抛出 `ValueError`，而不是静默返回由边界缺失导致的 `NaN`。resolver 原本返回的
`NaN` 会保留。

复用同一个 `cache` 再计算 `CLOSE`、`RETURNS` 或依赖它们的表达式时，可以复用已经
读取的叶子值和中间结果。

## 根据因子需求构造 RQData 上下文

`get_defined_factor_periods()` 汇总当前仍存活的全部因子表达式节点。上下文构造函数
不会自动读取该结果，应用可以选择是否把它作为历史和未来扩展量：

```python
from xqfactor import get_defined_factor_periods
from xqfactor.providers.rqdata import RQDataContextBuilder


periods = get_defined_factor_periods()
context_builder = RQDataContextBuilder()
rqdata_context = context_builder.build(
    start_date="2025-01-01",
    end_date="2025-06-30",
    universe=("000001.XSHE", "600000.XSHG"),
    market="cn",
    type="stock",
    frequency="D",
    history_period=periods.max_history,
    future_period=periods.max_future,
)
result = FORWARD_RETURNS.evaluate(rqdata_context, cache)
```

`history_period` 和 `future_period` 均按目标频率的 bar 数计算。当前支持中国股票
`D`、`min`、`W-SUN` 和 `ME`；分钟轴包含每个交易日的 09:31—11:30、
13:01—15:00，周频和月频分别使用每周、每月的最后交易日。

## 固定公共类因子

`FIX` 在目标资产的单标的上下文中计算因子，再将结果从 `(时间数, 1)` 广播为当前
universe 的 `(时间数, 资产数)`，适合指数或基准收益等公共类因子：

```python
from xqfactor import FIX


CSI500_RETURNS = FIX(RETURNS, "000985.XSHG")
EXCESS_RETURNS = RETURNS - CSI500_RETURNS
excess_returns = EXCESS_RETURNS.evaluate(context, cache)
```

固定因子的缓存会区分目标资产；同一目标资产在不同当前 universe 下求值时，可以复用
单标的子上下文中的叶子和中间结果。

## 使用 Polars 自定义算子

算子的公共输入输出仍为 Pandas DataFrame，只在这个算子内部转换到 Polars。
算子定义不绑定 `CLOSE`，因此可应用到任意因子。

```python
import pandas as pd
import polars as pl

from xqfactor import AbstractFactor, CombinedFactor


def polars_log1p(frame: pd.DataFrame) -> pd.DataFrame:
    """使用 Polars 计算 log(1 + x)。

    输入：形状为 (时间数, 资产数) 的 Pandas DataFrame。
    输出：形状、index 和 columns 与输入一致的 Pandas DataFrame。
    """
    index_name = frame.index.name or "datetime"

    # ************************************************************
    # DataFrame (T, N) -> Polars DataFrame (T, 1 + N)：
    # 时间 index 临时变为普通列；计算后恢复为 DataFrame (T, N)。
    # ************************************************************
    polars_frame = pl.from_pandas(frame.rename_axis(index_name).reset_index())
    transformed = polars_frame.with_columns(
        (pl.exclude(index_name) + 1).log()
    )
    return transformed.to_pandas().set_index(index_name).reindex_like(frame)


def POLARS_LOG1P(factor: AbstractFactor) -> CombinedFactor:
    """将 Polars log1p 逻辑应用到任意因子。"""
    return CombinedFactor(polars_log1p, factor)


POLARS_LOG_CLOSE = POLARS_LOG1P(CLOSE)
polars_result = POLARS_LOG_CLOSE.evaluate(context, cache)
```

## 使用 PyTorch 自定义算子

```python
import pandas as pd
import torch

from xqfactor import AbstractFactor, CombinedFactor


def torch_sigmoid(frame: pd.DataFrame) -> pd.DataFrame:
    """使用 PyTorch 对因子值执行 sigmoid。

    输入：形状为 (时间数, 资产数) 的 Pandas DataFrame。
    输出：形状、index 和 columns 与输入一致的 Pandas DataFrame。
    """
    # ************************************************************
    # DataFrame (T, N) -> Tensor (T, N) -> DataFrame (T, N)。
    # 时间轴和资产轴在转换前后保持不变。
    # ************************************************************
    tensor = torch.as_tensor(frame.to_numpy(), dtype=torch.float64)
    transformed = torch.sigmoid(tensor).numpy()
    return pd.DataFrame(
        transformed,
        index=frame.index,
        columns=frame.columns,
    )


def TORCH_SIGMOID(factor: AbstractFactor) -> CombinedFactor:
    """将 PyTorch sigmoid 逻辑应用到任意因子。"""
    return CombinedFactor(torch_sigmoid, factor)


TORCH_SIGMOID_CLOSE = TORCH_SIGMOID(CLOSE)
torch_result = TORCH_SIGMOID_CLOSE.evaluate(context, cache)
```

不要为 Polars 或 PyTorch 建立独立后端；新增算子时只维护该算子实际使用的实现。

## 先标准化再进行 IC 检验

预处理仍然输入和输出因子，因此先用 `NORM` 构造标准化因子，再把它交给 IC 检验器。

```python
from xqfactor import NORM
from xqfactor.analysis.ic import ICAnalyzer


raw_returns = RETURNS.evaluate(context, cache)
forward_returns = raw_returns.shift(-1)
NORMALIZED_RETURNS = NORM(RETURNS)

ic_result = ICAnalyzer(forward_returns).analyze(
    {"returns": NORMALIZED_RETURNS},
    context=context,
    cache=cache,
)
ic_series = ic_result.data
ic_summary = ic_result.summary()
normalized_returns = NORMALIZED_RETURNS.evaluate(context, cache)
```

需要先去极值再标准化时直接组合算子：

```python
from xqfactor import MAD, NORM


PROCESSED_RETURNS = NORM(MAD(RETURNS, n=3.0))
```

## 自定义检验器

继承 `AbstractAnalyzer` 并实现 `_analyze`。传入 `_analyze` 的因子已经求值为
`dict[str, pd.DataFrame]`，不再经过 Processor。

```python
from typing import Mapping

import pandas as pd

from xqfactor.analysis import AbstractAnalyzer


class CoverageAnalyzer(AbstractAnalyzer):
    """统计每个因子的非缺失数据覆盖率。"""

    def _analyze(
        self,
        factors: Mapping[str, pd.DataFrame],
    ) -> pd.Series:
        """计算全部时间和资产上的非缺失比例。

        输入：名称到二维因子值的映射。
        输出：index 为因子名、value 为覆盖率的 Series。
        """
        return pd.Series(
            {
                name: factor.notna().to_numpy().mean()
                for name, factor in factors.items()
            },
            name="coverage",
        )


coverage = CoverageAnalyzer().analyze(
    {
        "returns": NORMALIZED_RETURNS,
        "polars_log_close": POLARS_LOG_CLOSE,
    },
    context=context,
    cache=cache,
)
```

自定义检验器可以直接返回 Series、DataFrame、dataclass 或其他业务结果对象。
本项目只内置 IC、分组收益和回归等通用检验；行业专用报告、绘图、基准归因和策略规则
由应用项目实现。

## 自定义窗口算子与缓存

```python
import pandas as pd

from xqfactor import AbstractFactor, RollingWindowFactor


def rolling_mean(
    frame: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """计算时间序列移动平均。

    输入：形状为 (时间数, 资产数) 的因子值和窗口长度。
    输出：形状及轴与输入一致的移动平均 DataFrame。
    """
    return frame.rolling(window).mean()


def MA20(factor: AbstractFactor) -> RollingWindowFactor:
    """将 20 期移动平均应用到任意因子。"""
    return RollingWindowFactor(rolling_mean, 20, factor)


MA20_CLOSE = MA20(CLOSE)
```

多个因子需要共享同一个时间窗口时，使用 `CombinedRollingWindowFactor`。回调的第一个
参数是窗口长度，后续参数按构造函数中的因子顺序接收 DataFrame：

```python
from xqfactor import AbstractFactor, CombinedRollingWindowFactor


def rolling_spread(
    window: int,
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> pd.DataFrame:
    """计算两个因子的滚动均值差。

    输入：窗口长度，以及两个形状为 (时间数, 资产数) 的因子值。
    输出：形状和轴与输入一致的滚动均值差 DataFrame。
    """
    return left.rolling(window).mean() - right.rolling(window).mean()


def ROLLING_SPREAD(
    left: AbstractFactor,
    right: AbstractFactor,
    window: int = 20,
) -> CombinedRollingWindowFactor:
    """将两个因子组合为指定窗口的滚动均值差。"""
    return CombinedRollingWindowFactor(rolling_spread, window, left, right)


ROLLING_SPREAD_CLOSE = ROLLING_SPREAD(CLOSE, RETURNS)
```

构造上下文时至少提供 19 个额外历史周期，并把 `output_start` 设置到目标输出起点。
可以通过 `MA20_CLOSE.required_history()` 查询表达式需要的历史周期数。

缓存键包含：

- 因子定义和 resolver 版本；
- 完整时间轴与输出切片；
- universe 和 frequency；
- semantics；
- provider 版本。

不同 universe、频率或时间区间不会共享缓存。长期全市场数据存储不属于 xqfactor。
