"""使用 RQData 交易日历实现求值上下文构造协议。"""

from __future__ import annotations

from datetime import date
from importlib import import_module
from math import ceil
from typing import Protocol, Sequence, cast

import pandas as pd

from xqfactor.context import (
    AssetId,
    DateInput,
    EvaluationContext,
    Frequency,
    Market,
    validate_frequency,
)

_VALID_MARKETS = {"cn", "hk", "us"}
_VALID_TYPES = {
    "stock",
    "futures",
    "fund",
    "index",
    "option",
    "convertible",
    "spot",
}
_SUPPORTED_FREQUENCIES = {"D", "min", "W-SUN", "ME"}
_MINUTES_PER_CN_STOCK_DAY = 240


class TradingCalendarProvider(Protocol):
    """RQData 交易日查询所需的最小接口。"""

    def get_trading_dates(
        self,
        start_date: DateInput,
        end_date: DateInput,
        market: str = "cn",
    ) -> Sequence[date]:
        """返回起止日期内的交易日。

        输入：包含边界的开始日期、结束日期和市场标识。
        输出：按时间升序排列的交易日期序列。
        """


def _normalize_date(value: DateInput, name: str) -> pd.Timestamp:
    """将日期参数标准化为无时区的自然日。

    输入：日期值及用于错误信息的参数名。
    输出：时间归零且不含时区的 pandas Timestamp。
    """
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 不是有效日期") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} 不是有效日期")
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _validate_period(value: int, name: str) -> int:
    """校验历史或未来扩展周期。

    输入：待校验周期数和参数名。
    输出：合法的非负整数周期数。
    """
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} 必须是非负整数")
    return value


def _normalize_universe(universe: Sequence[AssetId]) -> tuple[AssetId, ...]:
    """校验并冻结资产池。

    输入：资产标识序列。
    输出：保持调用方顺序且不含重复值的资产标识元组。
    """
    if isinstance(universe, (str, bytes)):
        raise ValueError("universe 必须是资产标识序列，不能是单个字符串")
    normalized = tuple(universe)
    if not normalized:
        raise ValueError("universe 不能为空")
    if any(not isinstance(asset, (str, int)) for asset in normalized):
        raise ValueError("universe 中的资产标识必须是 str 或 int")
    if len(set(normalized)) != len(normalized):
        raise ValueError("universe 不能包含重复资产")
    return normalized


def _load_default_api() -> TradingCalendarProvider:
    """延迟加载 rqdatac 模块。

    输入：无。
    输出：已经由应用完成初始化的 rqdatac 模块。
    """
    try:
        return cast(TradingCalendarProvider, import_module("rqdatac"))
    except ImportError as error:
        raise ImportError(
            "使用 RQDataContextBuilder 前请在应用项目执行 `uv add rqdatac`，"
            "并完成 rqdatac.init()"
        ) from error


def _calendar_span(periods: int, frequency: str) -> int:
    """估算首轮交易日历查询需要覆盖的自然日数。

    输入：历史和未来需求中的较大 bar 数，以及标准化后的频率。
    输出：日历查询在起止日期两侧扩展的自然日数。
    """
    if frequency == "min":
        trading_days = ceil(periods / _MINUTES_PER_CN_STOCK_DAY)
        return max(14, trading_days * 3)
    if frequency == "W-SUN":
        return max(28, periods * 14)
    if frequency == "ME":
        return max(62, periods * 62)
    return max(14, periods * 3)


def _align_calendar_bounds(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """将周频和月频日历查询边界扩展到完整周期。

    输入：待查询自然日起止边界和目标频率。
    输出：不会产生残缺周或残缺月的查询边界。
    """
    if frequency == "W-SUN":
        return (
            start_date.to_period("W-SUN").start_time,
            end_date.to_period("W-SUN").end_time.normalize(),
        )
    if frequency == "ME":
        return (
            start_date.to_period("M").start_time,
            end_date.to_period("M").end_time.normalize(),
        )
    return start_date, end_date


def _normalize_trading_dates(values: Sequence[date]) -> pd.DatetimeIndex:
    """标准化 RQData 返回的交易日。

    输入：RQData 返回的日期序列。
    输出：升序、去重、时间归零的 DatetimeIndex。
    """
    if len(values) == 0:
        return pd.DatetimeIndex([], name="datetime")
    index = pd.DatetimeIndex(pd.to_datetime(tuple(values))).normalize()
    return pd.DatetimeIndex(index.unique()).sort_values().rename("datetime")


def _stock_minute_index(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """生成中国股票的一分钟交易时间轴。

    输入：形状为 ``(交易日数,)`` 的交易日 DatetimeIndex。
    输出：形状为 ``(交易日数 * 240,)`` 的分钟 DatetimeIndex；每个交易日包含
    09:31—11:30 和 13:01—15:00，index 名称保持为 datetime。
    """
    sessions: list[pd.DatetimeIndex] = []
    for trading_date in trading_dates:
        sessions.extend(
            [
                pd.date_range(
                    trading_date + pd.Timedelta(hours=9, minutes=30),
                    trading_date + pd.Timedelta(hours=11, minutes=30),
                    freq="min",
                    inclusive="right",
                    name="datetime",
                ),
                pd.date_range(
                    trading_date + pd.Timedelta(hours=13),
                    trading_date + pd.Timedelta(hours=15),
                    freq="min",
                    inclusive="right",
                    name="datetime",
                ),
            ]
        )
    if not sessions:
        return pd.DatetimeIndex([], name="datetime")
    return sessions[0].append(sessions[1:]).rename("datetime")


def _period_end_index(
    trading_dates: pd.DatetimeIndex,
    frequency: str,
) -> pd.DatetimeIndex:
    """把日频交易轴转换为周末或月末交易轴。

    输入：形状为 ``(交易日数,)`` 的日频轴和目标频率。
    输出：形状为 ``(周期数,)`` 的轴；每个元素为相应周或月的最后交易日。
    """
    # Period API 的自然月周期仍使用 M；输出上下文的 offset freqstr 保持 ME。
    period_frequency = "W-SUN" if frequency == "W-SUN" else "M"
    periods = trading_dates.to_period(period_frequency)
    ends = pd.Series(trading_dates, index=periods).groupby(level=0).last()
    return pd.DatetimeIndex(ends.to_numpy(), name="datetime")


def _build_bar_index(
    trading_dates: pd.DatetimeIndex,
    frequency: str,
) -> pd.DatetimeIndex:
    """将交易日转换为目标频率的完整 bar 轴。

    输入：日频交易轴和标准化频率。
    输出：目标频率对应的 DatetimeIndex。
    """
    if frequency == "min":
        return _stock_minute_index(trading_dates)
    if frequency in {"W-SUN", "ME"}:
        return _period_end_index(trading_dates, frequency)
    return trading_dates


def _context_from_bar_index(
    bar_index: pd.DatetimeIndex,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    universe: tuple[AssetId, ...],
    market: str,
    instrument_type: str,
    frequency: str,
    history_period: int,
    future_period: int,
) -> EvaluationContext | None:
    """尝试从候选 bar 轴构造上下文。

    输入：候选完整轴、输出日期范围、资产与数据语义、目标频率及两侧扩展量。
    输出：候选轴充足时返回 EvaluationContext；两侧 bar 不足时返回 None。
    """
    # ************************************************************
    # bar_index 从完整候选轴筛出日期落在 [start_date, end_date] 的输出轴；
    # 分钟 index 先归一化为日期再比较，周月 index 已是实际末期交易日。
    # ************************************************************
    bar_dates = bar_index.normalize()
    output_positions = (bar_dates >= start_date) & (bar_dates <= end_date)
    positions = output_positions.nonzero()[0]
    if len(positions) == 0:
        raise ValueError("start_date/end_date 范围内没有目标频率的交易 bar")

    first_output = int(positions[0])
    last_output = int(positions[-1])
    if first_output < history_period:
        return None
    if len(bar_index) - last_output - 1 < future_period:
        return None

    # ************************************************************
    # 候选轴 (T_candidate,) 截取为完整计算轴 (T_full,)；输出轴在其中的位置为
    # [history_period, history_period + T_output)，两侧恰好保留请求的 bar 数。
    # ************************************************************
    full_start = first_output - history_period
    full_end = last_output + future_period + 1
    time_index = tuple(bar_index[full_start:full_end])
    output_count = last_output - first_output + 1
    return EvaluationContext(
        time_index=time_index,
        universe=universe,
        frequency=frequency,
        output_start=history_period,
        output_end=history_period + output_count,
        semantics=(("market", market), ("type", instrument_type)),
        provider_version="rqdata",
    )


class RQDataContextBuilder:
    """使用 RQData 交易日历构造求值上下文。"""

    def __init__(self, api: TradingCalendarProvider | None = None) -> None:
        """创建 RQData 上下文构造器。

        输入：可选的 RQData 兼容交易日历接口；省略时在首次构造时加载 rqdatac。
        输出：可复用的上下文构造器实例。
        """
        self._api = api

    def build(
        self,
        start_date: DateInput,
        end_date: DateInput,
        universe: Sequence[AssetId],
        market: Market = "cn",
        type: str = "stock",
        frequency: Frequency = "D",
        history_period: int = 0,
        future_period: int = 0,
    ) -> EvaluationContext:
        """使用 RQData 交易日历构造 EvaluationContext。

        输入：包含边界的自然日起止日期、资产池、市场、资产类型、Pandas 规范
        freqstr，以及按目标频率 bar 计数的历史和未来扩展量。
        输出：完整时间轴包含两侧扩展，输出切片仅覆盖起止日期范围的求值上下文。
        """
        normalized_start = _normalize_date(start_date, "start_date")
        normalized_end = _normalize_date(end_date, "end_date")
        if normalized_start > normalized_end:
            raise ValueError("start_date 不能晚于 end_date")
        if market not in _VALID_MARKETS:
            raise ValueError(f"不支持的 market 参数：{market}")
        if type not in _VALID_TYPES:
            raise ValueError(f"不支持的 type 参数：{type}")

        validated_frequency = validate_frequency(frequency)
        if market != "cn" or type != "stock":
            raise NotImplementedError("当前仅实现中国市场股票 EvaluationContext")
        if validated_frequency not in _SUPPORTED_FREQUENCIES:
            raise NotImplementedError(
                "RQData 中国市场股票上下文当前仅支持 D、min、W-SUN 和 ME"
            )

        normalized_universe = _normalize_universe(universe)
        normalized_history = _validate_period(history_period, "history_period")
        normalized_future = _validate_period(future_period, "future_period")
        active_api = self._api if self._api is not None else _load_default_api()

        # ************************************************************
        # 首轮按频率估算自然日范围；若节假日或停市导致 bar 不足，则每轮把范围
        # 扩大一倍。周/月边界对齐到完整周期，避免把残缺周期误当作末期 bar。
        # ************************************************************
        span = _calendar_span(
            max(normalized_history, normalized_future),
            validated_frequency,
        )
        for _ in range(8):
            query_start, query_end = _align_calendar_bounds(
                normalized_start - pd.Timedelta(days=span),
                normalized_end + pd.Timedelta(days=span),
                validated_frequency,
            )
            trading_dates = _normalize_trading_dates(
                active_api.get_trading_dates(
                    start_date=query_start.date(),
                    end_date=query_end.date(),
                    market=market,
                )
            )
            bar_index = _build_bar_index(trading_dates, validated_frequency)
            context = _context_from_bar_index(
                bar_index,
                normalized_start,
                normalized_end,
                normalized_universe,
                market,
                type,
                validated_frequency,
                normalized_history,
                normalized_future,
            )
            if context is not None:
                return context
            span *= 2

        raise ValueError("RQData 交易日历无法提供足够的历史或未来周期")
