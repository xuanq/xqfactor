# xqfactor 使用示例

## 依赖与主时钟

应用项目自行安装并初始化行情 API：

```bash
uv add "xqfactor[analysis]"
uv add rqdatac
uv add polars
uv add torch
```

使用 RQData 交易日生成 XSHG 主时钟，但 universe 不受主交易所限制：

```python
import rqdatac

from xqfactor import EvaluationContextBuilder, get_defined_factor_periods
from xqfactor.providers.rqdata import RQDataTradingCalendar


rqdatac.init()
periods = get_defined_factor_periods()
context = EvaluationContextBuilder(RQDataTradingCalendar()).build(
    start_date="2026-07-20",
    end_date="2026-07-22",
    universe=(
        "600519.XSHG",
        "0700.HK",
        "000660.KS",
        "SKHY.nasdaq",
        "ETH.binance",
    ),
    primary_exchange="XSHG",
    frequency="min",
    history_period=periods.max_history,
    future_period=periods.max_future,
)
```

若显式指定 `history_period=30`、`future_period=5`，完整轴从
`2026-07-17 14:31+08:00` 到 `2026-07-23 09:35+08:00`，
`previous_time` 为 `2026-07-17 14:30+08:00`。输出切片只包含 7 月 20—22 日
三个 XSHG 交易日的 720 个分钟 bar。

## 定义混合市场 CLOSE

下面示例假定应用已经实现三个来源函数。每个函数返回：

- index：原始 bar 真正完成或数据可得的带时区时刻；
- columns：传入资产；
- values：对应时刻已经可得的收盘/最新价格。

RQData 或 Binance 若使用开始时刻标记 bar，resolver 必须先改成 API 返回的实际
close time。

```python
from collections.abc import Sequence

import pandas as pd

from xqfactor import (
    LeafFactor,
    LeafRequest,
    align_latest_observations,
)


def load_rqdata_close_observations(
    assets: Sequence[str],
    request: LeafRequest,
) -> pd.DataFrame:
    """读取中国、香港资产的已完成价格观测。

    输入：由 rqdatac-cached 负责的资产及叶子请求。
    输出：index 为实际可得时刻、columns 为 assets 的 DataFrame。
    """
    ...


def load_schwab_close_observations(
    assets: Sequence[str],
    request: LeafRequest,
) -> pd.DataFrame:
    """读取美国资产常规交易时段的已完成价格观测。

    输入：由 Schwab API 负责的资产及叶子请求。
    输出：index 为实际可得时刻、columns 为 assets 的 DataFrame。
    """
    ...


def load_korea_close_observations(
    assets: Sequence[str],
    request: LeafRequest,
) -> pd.DataFrame:
    """读取韩国资产已完成的正式收盘观测。

    输入：由应用选定韩国行情源负责的资产及叶子请求。
    输出：index 为实际可得时刻、columns 为 assets 的 DataFrame。
    """
    ...


def load_binance_close_observations(
    assets: Sequence[str],
    request: LeafRequest,
) -> pd.DataFrame:
    """读取加密资产已完成 Kline 的收盘观测。

    输入：由 Binance API 负责的资产及叶子请求。
    输出：index 使用 Kline close time、columns 为 assets 的 DataFrame。
    """
    ...


def load_mixed_market_close(request: LeafRequest) -> pd.DataFrame:
    """按资产路由 API，并对齐到主交易所右闭周期。

    输入：包含混合市场 universe 和完整主时钟的叶子请求。
    输出：形状为 ``(主时钟数, 资产数)``，columns 恢复原始 universe 顺序。
    """
    groups = {
        "rqdata": [
            asset
            for asset in request.context.universe
            if str(asset).endswith((".XSHG", ".XSHE", ".HK"))
        ],
        "korea": [
            asset
            for asset in request.context.universe
            if str(asset).endswith(".KS")
        ],
        "schwab": [
            asset
            for asset in request.context.universe
            if str(asset).endswith(".nasdaq")
        ],
        "binance": [
            asset
            for asset in request.context.universe
            if str(asset).endswith(".binance")
        ],
    }

    # ************************************************************
    # 分组表形状分别为 (T_source, N_group)，按 columns 拼接后变为
    # (所有来源时点并集, universe)；index 仍是各观测实际完成时刻。
    # ************************************************************
    frames = [
        load_rqdata_close_observations(groups["rqdata"], request),
        load_korea_close_observations(groups["korea"], request),
        load_schwab_close_observations(groups["schwab"], request),
        load_binance_close_observations(groups["binance"], request),
    ]
    observations = pd.concat(
        [frame for frame in frames if not frame.empty],
        axis=1,
    ).sort_index()

    # ************************************************************
    # DataFrame 从 (来源观测时点数, 资产数) 转换为
    # (主时钟数, 资产数)；每个 (period_start, time] 周期只取最后非空值。
    # ************************************************************
    aligned = align_latest_observations(observations, request.context)
    return aligned.reindex(columns=request.context.universe)


CLOSE = LeafFactor(
    "close",
    load_mixed_market_close,
    definition_version="mixed-master-close-v1",
)
```

上海 15:00 日频时：

- `600519.XSHG` 使用当日 15:00 收盘；
- `0700.HK` 使用不晚于 15:00 的最后完成分钟 bar，不能读取 16:00 后正式收盘；
- `000660.KS` 可使用 14:30 上海时间已经完成的当天正式收盘；
- `SKHY.nasdaq` 使用该主周期内已经可得的上一常规交易时段收盘；
- `ETH.binance` 使用不晚于 15:00 的最后完成 Kline。

来源休市且主周期内没有新观测时，结果为 NaN。若要无限回看旧价格，应定义口径不同的
LeafFactor 或显式算子，不要修改 `align_latest_observations()` 的默认语义。

## 组合 RETURNS、未来收益和 FIX

```python
from xqfactor import FIX, MemoryCache, PCT_CHANGE, RANK, REF


RETURNS = PCT_CHANGE(CLOSE, 1)
ALPHA = RANK(RETURNS) * -1
FORWARD_RETURNS = REF(RETURNS, -1)
CSI500_RETURNS = FIX(RETURNS, "000985.XSHG")
EXCESS_RETURNS = RETURNS - CSI500_RETURNS

assert RETURNS.required_history() == 1
assert FORWARD_RETURNS.required_future() == 1

cache = MemoryCache(maxsize=256)
alpha = ALPHA.evaluate(context, cache)
forward_returns = FORWARD_RETURNS.evaluate(context, cache)
```

`FIX` 只把子上下文 universe 改为目标资产，主交易所、`previous_time`、完整时间轴、
频率和日历版本保持不变。

## 自定义 Pandas 算子

```python
import pandas as pd

from xqfactor import AbstractFactor, CombinedFactor


def cross_sectional_demean(frame: pd.DataFrame) -> pd.DataFrame:
    """计算每个时间截面的去均值结果。

    输入：形状为 ``(时间数, 资产数)`` 的因子值。
    输出：形状、index 和 columns 均保持不变的 DataFrame。
    """
    return frame.sub(frame.mean(axis=1), axis=0)


def DEMEAN(factor: AbstractFactor) -> CombinedFactor:
    """把横截面去均值逻辑应用到任意因子。"""
    return CombinedFactor(cross_sectional_demean, factor)
```

## 在单个算子内使用 Polars 或 PyTorch

```python
import pandas as pd
import polars as pl
import torch

from xqfactor import AbstractFactor, CombinedFactor


def polars_log1p(frame: pd.DataFrame) -> pd.DataFrame:
    """使用 Polars 计算 log(1 + x) 并恢复 Pandas 轴。"""
    index_name = frame.index.name or "datetime"

    # ************************************************************
    # Pandas DataFrame (T, N) -> Polars DataFrame (T, 1 + N)；
    # 时间 index 临时变成列，计算后恢复为 Pandas DataFrame (T, N)。
    # ************************************************************
    value = pl.from_pandas(frame.rename_axis(index_name).reset_index())
    transformed = value.with_columns((pl.exclude(index_name) + 1).log())
    return transformed.to_pandas().set_index(index_name).reindex_like(frame)


def torch_sigmoid(frame: pd.DataFrame) -> pd.DataFrame:
    """使用 PyTorch 计算 sigmoid 并保持输入轴。"""
    # DataFrame (T, N) -> Tensor (T, N) -> DataFrame (T, N)。
    tensor = torch.as_tensor(frame.to_numpy(), dtype=torch.float64)
    return pd.DataFrame(
        torch.sigmoid(tensor).numpy(),
        index=frame.index,
        columns=frame.columns,
    )


def POLARS_LOG1P(factor: AbstractFactor) -> CombinedFactor:
    """把 Polars log1p 应用到任意因子。"""
    return CombinedFactor(polars_log1p, factor)


def TORCH_SIGMOID(factor: AbstractFactor) -> CombinedFactor:
    """把 PyTorch sigmoid 应用到任意因子。"""
    return CombinedFactor(torch_sigmoid, factor)
```

## 自定义窗口与检验

```python
import pandas as pd

from xqfactor import AbstractFactor, RollingWindowFactor
from xqfactor.analysis.ic import ICAnalyzer


def rolling_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """沿时间轴计算窗口均值并保持输入形状。"""
    return frame.rolling(window).mean()


def MA20(factor: AbstractFactor) -> RollingWindowFactor:
    """把 20 期均值应用到任意因子。"""
    return RollingWindowFactor(rolling_mean, 20, factor)


MA20_CLOSE = MA20(CLOSE)
result = ICAnalyzer(FORWARD_RETURNS).analyze(
    {"alpha": ALPHA},
    context=context,
    cache=cache,
)
```

构造上下文时必须把表达式的 `required_history()` 和 `required_future()` 需求计入两侧
扩展。缓存键包含因子定义以及完整主时钟身份；长期行情缓存不属于 xqfactor。
