# xqfactor

`xqfactor` 是数据源无关、统一使用 `pandas.DataFrame` 传递因子值的因子表达式、
执行缓存和检验规范框架。

核心包不依赖 RQData、数据库或本地数据仓库。应用项目通过 `LeafFactor` resolver
实现实际取数；xqfactor 负责主交易所计算轴、跨市场可得时间对齐、表达式组合、窗口
需求、递归求值和相同执行上下文下的缓存复用。

## 安装

```bash
uv add xqfactor
```

需要内置统计检验时：

```bash
uv add "xqfactor[analysis]"
```

## 主交易所计算时钟

`EvaluationContext` 的时间轴统一使用 `Asia/Shanghai`，每个时点表示主交易所 bar 的
结束时刻。`primary_exchange` 只决定计算轴，不限制 universe；同一上下文可以包含中国、
香港、韩国、美国和加密资产。

推荐通过交易日历构造上下文。应用安装并初始化 `rqdatac` 后，可以生成 XSHG、XSHE 的
`D`、`min`、`W-SUN` 和 `ME` 主时钟：

```bash
uv add rqdatac
```

```python
import rqdatac

from xqfactor import EvaluationContextBuilder, get_defined_factor_periods
from xqfactor.providers.rqdata import RQDataTradingCalendar


rqdatac.init()
periods = get_defined_factor_periods()
context_builder = EvaluationContextBuilder(RQDataTradingCalendar())
context = context_builder.build(
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

`history_period` 和 `future_period` 按主轴 bar 计数。构造器还会单独保存
`previous_time`，使完整轴第一行也有明确的左开右闭周期
`(previous_time, time_index[0]]`。

## 定义跨市场 LeafFactor

资产识别、API 路由、复权方式和交易时段口径都由 resolver 决定。原始数据 index 必须
使用观测真正完成或可得的时刻，而不是无时区自然日或 Kline 开始时刻。

`align_latest_observations()` 对每个主轴周期只保留最后一个非空观测；周期内没有新数据
时返回 `NaN`，不会把任意久以前的旧价格前向填充：

```python
import pandas as pd

from xqfactor import (
    LeafFactor,
    LeafRequest,
    align_latest_observations,
)


def load_close(request: LeafRequest) -> pd.DataFrame:
    """按资产路由数据源并返回主时钟周期内最后可得价格。

    输入：包含混合市场 universe 和主交易所时钟的叶子请求。
    输出：index 为完整主时钟、columns 为原始 universe 的收盘价 DataFrame。
    """
    context = request.context

    # ************************************************************
    # 应用在此按资产后缀分组调用 rqdatac-cached、Schwab 和 Binance。
    # 每个原始表形状为 (观测数, 分组资产数)，index 必须改为 bar 完成时刻；
    # 多个表按 columns 拼接后变为 (全部观测时点并集, universe 子集)。
    # ************************************************************
    raw_observations = load_mixed_market_observations(context)
    aligned = align_latest_observations(raw_observations, context)
    return aligned.reindex(columns=context.universe)


CLOSE = LeafFactor(
    "close",
    load_close,
    definition_version="mixed-master-close-v1",
)
```

若上海日频主轴时点为 15:00：

- 中国股票使用当天 15:00 收盘价；
- 港股使用不晚于 15:00 的最后完成分钟价，不能使用 16:00 后才可得的正式收盘价；
- 已收盘的韩国市场使用当天正式收盘价；
- 美国股票使用该时点前最近一个主轴周期内已经完成的常规交易时段收盘；
- Binance 必须按 Kline `close time` 判断是否已经完成。

需要各资产所在交易所正式日收盘时，应定义独立的 `SESSION_CLOSE` LeafFactor，不要改变
主时钟 `CLOSE` 的无前视语义。路由、复权或交易时段口径变化时更新
`LeafFactor.definition_version`。

## 组合与执行

```python
from xqfactor import MemoryCache, PCT_CHANGE, RANK, REF


RETURNS = PCT_CHANGE(CLOSE, 1)
ALPHA = RANK(RETURNS) * -1
FORWARD_RETURNS = REF(RETURNS, -1)

cache = MemoryCache(maxsize=256)
alpha = ALPHA.evaluate(context, cache)
```

`REF(X, n)` 中正数引用过去，负数引用未来。完整 `time_index` 必须在
`output_start` 前提供足够历史 bar，并在 `output_end` 后提供足够未来 bar；不足时
`evaluate()` 会明确报错。

`EvaluationContext` 指纹包含完整主时钟、`previous_time`、universe、主交易所、频率、
输出切片和日历版本。数据源路由及价格定义由 LeafFactor 指纹隔离。

## 固定公共类因子

`FIX` 会在只包含目标资产的子上下文中计算表达式，再广播到当前 universe。子上下文保留
主交易所、首周期边界和日历版本：

```python
from xqfactor import FIX


CSI500_RETURNS = FIX(RETURNS, "000985.XSHG")
EXCESS_RETURNS = RETURNS - CSI500_RETURNS
```

## 自定义算子

自定义算子分为纯 DataFrame 计算函数和表达式构造函数两层：

```python
import pandas as pd

from xqfactor import AbstractFactor, CombinedFactor


def cross_sectional_demean(frame: pd.DataFrame) -> pd.DataFrame:
    """将每个时间截面的值减去该截面均值。"""
    return frame.sub(frame.mean(axis=1), axis=0)


def DEMEAN(factor: AbstractFactor) -> CombinedFactor:
    """把横截面去均值逻辑应用到任意因子。"""
    return CombinedFactor(cross_sectional_demean, factor)
```

## 职责边界

- 核心不解析资产命名，也不维护资产、交易所或数据源注册表。
- `TradingCalendar` 只提供主交易所 bar 结束时刻；其他交易所可通过协议接入。
- resolver 返回的时间索引必须带时区；核心会转换到上海时区后再对齐。
- 执行缓存不是行情数据库，不负责长期数据存储或跨 universe 数据集管理。
- Polars、PyTorch 可在单个自定义算子内部使用，但不形成独立计算后端。

## 代码阅读路径

从 `context.py` 的 `EvaluationContextBuilder.build()` 开始，交易日历生成候选 bar，
构造器截取历史、输出和未来轴并单独保存 `previous_time`；
`align_latest_observations()` 将 resolver 的可得时间观测分配到主轴周期。
`providers/rqdata.py` 的 `RQDataTradingCalendar.get_bar_index()` 提供 XSHG、XSHE
日历适配。因子执行从 `factor.py` 的 `AbstractFactor.evaluate()` 开始，递归计算并按
时区化主轴和 universe 对齐，最后截取输出区间。检验流程从
`analysis/base.py` 的 `AbstractAnalyzer.analyze()` 开始，共享同一个执行缓存后进入
IC、分组收益或回归统计。
