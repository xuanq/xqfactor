"""使用 RQData 交易日实现上海、深圳证券交易所主时钟。"""

from __future__ import annotations

from datetime import date
from importlib import import_module
from typing import Protocol, Sequence, cast

import pandas as pd

from xqfactor.context import (
    DateInput,
    EVALUATION_TIMEZONE,
    ExchangeId,
    Frequency,
    validate_frequency,
)


_SUPPORTED_EXCHANGES = {"XSHG", "XSHE"}
_SUPPORTED_FREQUENCIES = {"D", "min", "W-SUN", "ME"}


class TradingDateProvider(Protocol):
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
        ...


def _load_default_api() -> TradingDateProvider:
    """延迟加载 rqdatac 模块。

    输入：无。
    输出：已经由应用完成初始化的 rqdatac 模块。
    """
    try:
        return cast(TradingDateProvider, import_module("rqdatac"))
    except ImportError as error:
        raise ImportError(
            "使用 RQDataTradingCalendar 前请在应用项目执行 `uv add rqdatac`，"
            "并完成 rqdatac.init()"
        ) from error


def _normalize_date(value: DateInput, name: str) -> pd.Timestamp:
    """把日期参数转换为上海时区自然日。

    输入：日期值及用于错误信息的参数名。
    输出：时区为 Asia/Shanghai 且时间归零的 Timestamp。
    """
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 不是有效日期") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} 不是有效日期")
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(EVALUATION_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(EVALUATION_TIMEZONE)
    return timestamp.normalize()


def _align_calendar_bounds(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    frequency: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """把周频和月频查询边界扩展到完整自然周期。

    输入：上海时区自然日起止边界和目标频率。
    输出：覆盖完整周或月的上海时区查询边界。
    """
    start_naive = start_date.tz_localize(None)
    end_naive = end_date.tz_localize(None)
    if frequency == "W-SUN":
        aligned_start = start_naive.to_period("W-SUN").start_time
        aligned_end = end_naive.to_period("W-SUN").end_time.normalize()
    elif frequency == "ME":
        aligned_start = start_naive.to_period("M").start_time
        aligned_end = end_naive.to_period("M").end_time.normalize()
    else:
        return start_date, end_date
    return (
        aligned_start.tz_localize(EVALUATION_TIMEZONE),
        aligned_end.tz_localize(EVALUATION_TIMEZONE),
    )


def _normalize_trading_dates(values: Sequence[date]) -> pd.DatetimeIndex:
    """标准化 RQData 返回的交易日。

    输入：RQData 返回的日期序列。
    输出：升序、去重、时间归零且无时区的 DatetimeIndex。
    """
    if len(values) == 0:
        return pd.DatetimeIndex([], name="datetime")
    index = pd.DatetimeIndex(pd.to_datetime(tuple(values))).normalize()
    return pd.DatetimeIndex(index.unique()).sort_values().rename("datetime")


def _stock_minute_index(trading_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """生成中国股票一分钟 bar 结束时刻。

    输入：形状为 ``(交易日数,)`` 的无时区交易日 DatetimeIndex。
    输出：形状为 ``(交易日数 * 240,)`` 的 Asia/Shanghai DatetimeIndex；
    每个交易日包含 09:31—11:30 和 13:01—15:00。
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
                    tz=EVALUATION_TIMEZONE,
                    name="datetime",
                ),
                pd.date_range(
                    trading_date + pd.Timedelta(hours=13),
                    trading_date + pd.Timedelta(hours=15),
                    freq="min",
                    inclusive="right",
                    tz=EVALUATION_TIMEZONE,
                    name="datetime",
                ),
            ]
        )
    if not sessions:
        return pd.DatetimeIndex([], tz=EVALUATION_TIMEZONE, name="datetime")
    return sessions[0].append(sessions[1:]).rename("datetime")


def _stock_daily_close_index(
    trading_dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    """把交易日转换为中国股票实际日线结束时刻。

    输入：形状为 ``(交易日数,)`` 的无时区交易日 DatetimeIndex。
    输出：相同形状、每天 15:00 且时区为 Asia/Shanghai 的 DatetimeIndex。
    """
    closes = trading_dates + pd.Timedelta(hours=15)
    return closes.tz_localize(EVALUATION_TIMEZONE).rename("datetime")


def _period_end_index(
    daily_closes: pd.DatetimeIndex,
    frequency: str,
) -> pd.DatetimeIndex:
    """把日线结束时刻转换为周末或月末交易轴。

    输入：形状为 ``(交易日数,)`` 的日线收盘轴和目标频率。
    输出：形状为 ``(周期数,)`` 的轴；每个元素为相应周期最后交易日的实际收盘时刻。
    """
    local_dates = daily_closes.tz_localize(None).normalize()
    period_frequency = "W-SUN" if frequency == "W-SUN" else "M"
    periods = local_dates.to_period(period_frequency)
    ends = pd.Series(daily_closes, index=periods).groupby(level=0).last()
    return pd.DatetimeIndex(ends.to_list(), name="datetime")


class RQDataTradingCalendar:
    """使用 RQData 交易日生成 XSHG、XSHE 主时钟 bar。"""

    def __init__(
        self,
        api: TradingDateProvider | None = None,
        version: str = "rqdata-cn-stock-calendar-v2",
    ) -> None:
        """创建 RQData 交易日历。

        输入：可选的 RQData 兼容交易日接口和稳定日历版本。
        输出：可注入 EvaluationContextBuilder 的交易日历实例。
        """
        if not isinstance(version, str) or not version:
            raise ValueError("version 必须是非空字符串")
        self._api = api
        self._version = version

    @property
    def version(self) -> str:
        """返回稳定的交易日历实现版本。"""
        return self._version

    def get_bar_index(
        self,
        primary_exchange: ExchangeId,
        start_date: DateInput,
        end_date: DateInput,
        frequency: Frequency,
    ) -> pd.DatetimeIndex:
        """返回中国股票主交易所 bar 结束时刻。

        输入：XSHG 或 XSHE、包含边界的上海自然日范围和 Pandas 规范频率。
        输出：严格递增、无重复且时区为 Asia/Shanghai 的 DatetimeIndex。
        """
        if primary_exchange not in _SUPPORTED_EXCHANGES:
            raise NotImplementedError("RQDataTradingCalendar 当前仅支持 XSHG 和 XSHE")
        validated_frequency = validate_frequency(frequency)
        if validated_frequency not in _SUPPORTED_FREQUENCIES:
            raise NotImplementedError(
                "RQDataTradingCalendar 当前仅支持 D、min、W-SUN 和 ME"
            )

        normalized_start = _normalize_date(start_date, "start_date")
        normalized_end = _normalize_date(end_date, "end_date")
        if normalized_start > normalized_end:
            raise ValueError("start_date 不能晚于 end_date")
        query_start, query_end = _align_calendar_bounds(
            normalized_start,
            normalized_end,
            validated_frequency,
        )
        active_api = self._api if self._api is not None else _load_default_api()
        trading_dates = _normalize_trading_dates(
            active_api.get_trading_dates(
                start_date=query_start.date(),
                end_date=query_end.date(),
                market="cn",
            )
        )

        if validated_frequency == "min":
            return _stock_minute_index(trading_dates)
        daily_closes = _stock_daily_close_index(trading_dates)
        if validated_frequency in {"W-SUN", "ME"}:
            return _period_end_index(daily_closes, validated_frequency)
        return daily_closes
