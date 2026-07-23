---
name: xqfactor
description: Use xqfactor to build timezone-aware primary-exchange evaluation clocks, align mixed-market observations without lookahead, define data-source-independent Pandas factors, evaluate cached expression graphs, and run factor analyses.
---

# xqfactor

## 目标

使用本项目定义数据源无关的因子表达式。外部 I/O、资产识别、API 路由、复权和交易时段
口径放在应用拥有的 `LeafFactor` resolver 中；主交易所时钟、跨市场可得时间对齐、
表达式图、历史需求、轴对齐和执行缓存交给 xqfactor。

所有因子节点统一传递 Pandas DataFrame：

- index：`Asia/Shanghai` 时区的主时钟 bar 结束时刻；
- columns：资产；
- 逻辑形状：`(时间数, 资产数)`。

不要创建数据源注册表或多计算后端。

## 开始使用

1. 阅读仓库根目录 `AGENTS.md`。
2. 查看 `src/xqfactor/__init__.py`，确认公共 API。
3. 阅读 `references/examples.md` 中的完整示例。
4. 在应用项目安装依赖：

```bash
uv add xqfactor
uv add "xqfactor[analysis]"
```

RQData、rqdatac-cached、Schwab、Binance、Polars 和 PyTorch 等依赖由应用自行安装。

## 主时钟与上下文

- `EvaluationContext`：完整主交易所时间轴、首周期左边界、universe、频率、输出切片和
  日历版本。
- `TradingCalendar`：返回指定交易所和频率 bar 结束时刻的协议。
- `EvaluationContextBuilder(calendar)`：根据自然日范围及历史、未来 bar 数构造上下文。
- `RQDataTradingCalendar`：位于 `xqfactor.providers.rqdata`，支持 XSHG、XSHE 的
  `D`、`min`、`W-SUN`、`ME`。

```python
from xqfactor import EvaluationContextBuilder, get_defined_factor_periods
from xqfactor.providers.rqdata import RQDataTradingCalendar


periods = get_defined_factor_periods()
context = EvaluationContextBuilder(RQDataTradingCalendar()).build(
    start_date="2026-07-20",
    end_date="2026-07-22",
    universe=("600519.XSHG", "0700.HK", "SKHY.nasdaq", "ETH.binance"),
    primary_exchange="XSHG",
    frequency="min",
    history_period=periods.max_history,
    future_period=periods.max_future,
)
```

`time_index` 是完整计算轴；`previous_time` 是其首行周期的左边界。每个时点对应
`(period_start, time]`。`output_start/output_end` 只裁剪根节点结果。

## 定义跨市场叶子因子

`align_latest_observations(raw, context)` 要求 raw index 是数据真正完成或可得的带时区
时刻。它按列选择各主轴周期内最后一个非空观测，无新观测时返回 NaN，不跨周期前填。

resolver 应：

1. 按资产分组调用 rqdatac-cached、Schwab、Binance 等 API；
2. 将来源时间改为实际 bar 完成时刻；
3. 把分组结果按 columns 拼接；
4. 调用 `align_latest_observations()`；
5. 按 `context.universe` 恢复列顺序。

```python
CLOSE = LeafFactor(
    "close",
    load_mixed_market_close,
    definition_version="mixed-master-close-v1",
)
```

上海 15:00 日频时，港股使用不晚于 15:00 的最后完成分钟价，不使用未来正式收盘；
韩国已收盘资产可以使用当天正式收盘；美股使用主周期内已经可得的上一常规时段收盘；
Binance 必须按 Kline close time 判断完成。需要各市场官方日收盘时定义独立
`SESSION_CLOSE` LeafFactor。

路由、复权或交易时段定义变化时更新 `definition_version`。不要把数据版本重新放入
EvaluationContext。

## 表达式、窗口与缓存

直接组合 `PCT_CHANGE`、`REF`、`RANK`、`NORM`、`MAD`、`IF`、
`CSNEUTRALIZER`、`FIX` 等算子：

```python
RETURNS = PCT_CHANGE(CLOSE, 1)
ALPHA = RANK(RETURNS) * -1
FORWARD_RETURNS = REF(RETURNS, -1)
```

`REF(X, n)` 正数引用过去，负数引用未来。表达式通过 `required_history()` 和
`required_future()` 声明两侧需求；上下文不足时求值会报错。

`FIX` 在目标资产的单标的子上下文中求值并广播，保留主交易所、首周期边界、频率和
日历版本。

复用 `MemoryCache` 时，因子定义、完整时间轴、`previous_time`、universe、
`primary_exchange`、频率、输出切片或日历版本变化都会形成不同缓存键。

## 自定义算子与检验

自定义算子采用两层定义：

1. 输入输出均为 DataFrame 的纯计算函数；
2. 接收因子并返回 `CombinedFactor`、`UnaryCombinedFactor`、
   `BinaryCombinedFactor` 或窗口节点的构造函数。

Polars、PyTorch 只在某个自定义算子内部转换，入口和出口保持 Pandas DataFrame。

检验器从具体领域模块导入。标准化、去极值、中性化等仍然使用因子算子，检验器只处理
已经求值的 DataFrame。

## 修改与验证

- 公共 API 变化时同步更新本 Skill 和 `references/examples.md`。
- 不引入全局配置、数据 API 注册表、Processor、FactorRuntime 或多后端体系。
- 测试时区、周期左右边界、NaN、轴顺序、历史/未来需求和缓存身份。

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -m compileall -q src
uv build
uv run --with pyyaml python /Users/xuanqi/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/xqfactor
```
