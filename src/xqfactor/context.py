"""因子求值主时钟、交易日历构造器和跨市场观测对齐。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from math import ceil
from typing import Any, Protocol, Sequence, runtime_checkable
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


AssetId = str | int
DateInput = str | date | datetime | pd.Timestamp
ExchangeId = str
Frequency = str
EVALUATION_TIMEZONE = "Asia/Shanghai"
_NON_CANONICAL_FREQUENCY_ALIASES = {
    "d": "D",
    "1d": "D",
    "1m": "ME",
    "W": "W-SUN",
    "M": "ME",
    "T": "min",
}


def validate_frequency(frequency: Frequency) -> Frequency:
    """校验频率是否为 Pandas 规范 freqstr。

    输入：待校验的 Pandas 频率字符串。
    输出：与输入相同的规范 freqstr；别名、弃用值或非法值会抛出 ValueError。
    """
    if not isinstance(frequency, str) or not frequency:
        raise ValueError("frequency 必须是非空 Pandas freqstr")
    alias_target = _NON_CANONICAL_FREQUENCY_ALIASES.get(frequency)
    if alias_target is not None:
        raise ValueError(
            "frequency 必须使用 Pandas 规范 freqstr："
            f"{frequency!r} 应改为 {alias_target!r}"
        )
    try:
        # ************************************************************
        # Pandas 2.x 会对 M、T 等旧别名发出 FutureWarning 后返回新 freqstr；
        # 这里屏蔽警告并统一通过字符串比较给出项目自身的稳定错误信息。
        # ************************************************************
        with catch_warnings():
            simplefilter("ignore", FutureWarning)
            canonical = to_offset(frequency).freqstr
    except ValueError as error:
        raise ValueError(
            f"frequency 不是有效的 Pandas freqstr：{frequency!r}"
        ) from error
    if frequency != canonical:
        raise ValueError(
            "frequency 必须使用 Pandas 规范 freqstr："
            f"{frequency!r} 应改为 {canonical!r}"
        )
    return frequency


def _to_shanghai_timestamp(value: Any, name: str) -> pd.Timestamp:
    """把时间值转换为上海时区 Timestamp。

    输入：可由 Pandas 解析的时间值和用于错误信息的字段名。
    输出：时区为 Asia/Shanghai 的 Timestamp；无时区输入按上海本地时间解释。
    """
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} 不是有效时间") from error
    if pd.isna(timestamp):
        raise ValueError(f"{name} 不是有效时间")
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(EVALUATION_TIMEZONE)
    return timestamp.tz_convert(EVALUATION_TIMEZONE)


def _normalize_date(value: DateInput, name: str) -> pd.Timestamp:
    """把日期参数转换为上海时区自然日。

    输入：日期或时间值和用于错误信息的参数名。
    输出：转换到 Asia/Shanghai 后归一化到午夜的 Timestamp。
    """
    return _to_shanghai_timestamp(value, name).normalize()


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


@dataclass(frozen=True)
class EvaluationContext:
    """一次因子求值使用的主交易所时钟和资产池。"""

    time_index: tuple[pd.Timestamp, ...]
    previous_time: pd.Timestamp
    universe: tuple[AssetId, ...]
    primary_exchange: ExchangeId
    frequency: Frequency
    output_start: int = 0
    output_end: int | None = None
    calendar_version: str = "default"

    def __post_init__(self) -> None:
        """标准化时间与不可变字段，并校验主时钟和输出区间。"""
        time_index = tuple(
            _to_shanghai_timestamp(value, f"time_index[{position}]")
            for position, value in enumerate(self.time_index)
        )
        universe = _normalize_universe(self.universe)
        if not time_index:
            raise ValueError("time_index 不能为空")

        datetime_index = pd.DatetimeIndex(time_index)
        if datetime_index.has_duplicates:
            raise ValueError("time_index 不能包含重复值")
        if not datetime_index.is_monotonic_increasing:
            raise ValueError("time_index 必须严格递增")

        previous_time = _to_shanghai_timestamp(self.previous_time, "previous_time")
        if previous_time >= time_index[0]:
            raise ValueError("previous_time 必须早于 time_index 首个时点")
        if not isinstance(self.primary_exchange, str) or not self.primary_exchange:
            raise ValueError("primary_exchange 必须是非空字符串")
        validate_frequency(self.frequency)
        if not isinstance(self.calendar_version, str) or not self.calendar_version:
            raise ValueError("calendar_version 必须是非空字符串")

        output_end = len(time_index) if self.output_end is None else self.output_end
        if not 0 <= self.output_start < output_end <= len(time_index):
            raise ValueError("output_start/output_end 超出 time_index 范围")

        object.__setattr__(self, "time_index", time_index)
        object.__setattr__(self, "previous_time", previous_time)
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "output_end", output_end)

    @property
    def timezone(self) -> str:
        """返回求值主时钟统一使用的时区名称。"""
        return EVALUATION_TIMEZONE

    @property
    def period_start_index(self) -> tuple[pd.Timestamp, ...]:
        """返回各主轴右闭周期对应的左边界。"""
        return (self.previous_time, *self.time_index[:-1])

    @property
    def output_time_index(self) -> tuple[pd.Timestamp, ...]:
        """返回最终结果对应的时间轴。"""
        return self.time_index[self.output_start : self.output_end]

    @property
    def start_time(self) -> pd.Timestamp:
        """返回最终结果起始时间。"""
        return self.output_time_index[0]

    @property
    def end_time(self) -> pd.Timestamp:
        """返回最终结果结束时间。"""
        return self.output_time_index[-1]

    def fingerprint(self) -> str:
        """生成包含主时钟、求值范围和日历版本的上下文指纹。"""
        from xqfactor.runtime import stable_fingerprint

        return stable_fingerprint(
            {
                "time_index": self.time_index,
                "previous_time": self.previous_time,
                "universe": self.universe,
                "primary_exchange": self.primary_exchange,
                "frequency": self.frequency,
                "timezone": self.timezone,
                "output_start": self.output_start,
                "output_end": self.output_end,
                "calendar_version": self.calendar_version,
            }
        )


@dataclass(frozen=True)
class LeafRequest:
    """叶子因子的取数请求。"""

    factor_name: str
    context: EvaluationContext
    definition_version: str


@runtime_checkable
class TradingCalendar(Protocol):
    """向通用上下文构造器提供主交易所 bar 结束时刻的协议。"""

    @property
    def version(self) -> str:
        """返回稳定的交易日历实现版本。"""
        ...

    def get_bar_index(
        self,
        primary_exchange: ExchangeId,
        start_date: DateInput,
        end_date: DateInput,
        frequency: Frequency,
    ) -> pd.DatetimeIndex:
        """返回指定交易所和频率的 bar 结束时刻。

        输入：主交易所、包含边界的自然日起止日期和 Pandas 规范频率。
        输出：严格递增、无重复且带时区的 bar 结束时刻 DatetimeIndex。
        """
        ...


def _calendar_span(periods: int, frequency: str) -> int:
    """估算首轮交易日历查询需要覆盖的自然日数。

    输入：单侧所需最大 bar 数和标准化后的频率。
    输出：日历查询在目标日期两侧扩展的自然日数。
    """
    if frequency == "min":
        trading_days = ceil(periods / 240)
        return max(14, trading_days * 3)
    if frequency == "W-SUN":
        return max(28, periods * 14)
    if frequency == "ME":
        return max(62, periods * 62)
    return max(14, periods * 3)


def _normalize_calendar_index(values: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """校验交易日历结果并统一到上海时区。

    输入：交易日历返回的 bar 结束时刻。
    输出：严格递增、无重复且转换到 Asia/Shanghai 的 DatetimeIndex。
    """
    if not isinstance(values, pd.DatetimeIndex):
        raise TypeError("TradingCalendar.get_bar_index 必须返回 DatetimeIndex")
    if values.tz is None:
        raise ValueError("TradingCalendar 返回的 bar 时间必须包含时区")
    index = values.tz_convert(EVALUATION_TIMEZONE)
    if index.has_duplicates:
        raise ValueError("TradingCalendar 返回的 bar 时间不能重复")
    if not index.is_monotonic_increasing:
        raise ValueError("TradingCalendar 返回的 bar 时间必须严格递增")
    return index


class EvaluationContextBuilder:
    """使用可插拔交易日历构造主交易所求值上下文。"""

    def __init__(self, calendar: TradingCalendar) -> None:
        """创建通用上下文构造器。

        输入：提供 bar 结束时刻的交易日历实现。
        输出：可复用的 EvaluationContext 构造器。
        """
        self._calendar = calendar

    def build(
        self,
        start_date: DateInput,
        end_date: DateInput,
        universe: Sequence[AssetId],
        primary_exchange: ExchangeId,
        frequency: Frequency = "D",
        history_period: int = 0,
        future_period: int = 0,
    ) -> EvaluationContext:
        """构造包含完整计算轴和最终输出切片的求值上下文。

        输入：上海自然日起止范围、资产池、主交易所、Pandas 规范频率及两侧
        按主轴 bar 计数的扩展量。
        输出：可直接传给因子表达式求值的 EvaluationContext。
        """
        normalized_start = _normalize_date(start_date, "start_date")
        normalized_end = _normalize_date(end_date, "end_date")
        if normalized_start > normalized_end:
            raise ValueError("start_date 不能晚于 end_date")
        if not isinstance(primary_exchange, str) or not primary_exchange:
            raise ValueError("primary_exchange 必须是非空字符串")

        normalized_universe = _normalize_universe(universe)
        validated_frequency = validate_frequency(frequency)
        normalized_history = _validate_period(history_period, "history_period")
        normalized_future = _validate_period(future_period, "future_period")
        calendar_version = self._calendar.version
        if not isinstance(calendar_version, str) or not calendar_version:
            raise ValueError("TradingCalendar.version 必须是非空字符串")

        # ************************************************************
        # 完整轴左侧除 history_period 外再多查询一个 bar，用作首周期左边界；
        # 若节假日导致候选 bar 不足，则逐轮扩大自然日查询范围。
        # ************************************************************
        span = _calendar_span(
            max(normalized_history + 1, normalized_future),
            validated_frequency,
        )
        for _ in range(8):
            query_start = normalized_start - pd.Timedelta(days=span)
            query_end = normalized_end + pd.Timedelta(days=span)
            bar_index = _normalize_calendar_index(
                self._calendar.get_bar_index(
                    primary_exchange=primary_exchange,
                    start_date=query_start,
                    end_date=query_end,
                    frequency=validated_frequency,
                )
            )

            bar_dates = bar_index.normalize()
            output_mask = (bar_dates >= normalized_start) & (
                bar_dates <= normalized_end
            )
            positions = np.flatnonzero(output_mask)
            if len(positions) == 0:
                raise ValueError("start_date/end_date 范围内没有目标频率的交易 bar")

            first_output = int(positions[0])
            last_output = int(positions[-1])
            full_start = first_output - normalized_history
            full_end = last_output + normalized_future + 1
            if full_start < 1 or full_end > len(bar_index):
                span *= 2
                continue

            # ************************************************************
            # 候选轴 (T_candidate,) 截取为完整计算轴 (T_full,)；
            # previous_time 单独保存候选轴中紧邻完整轴的前一个时点。
            # ************************************************************
            time_index = tuple(bar_index[full_start:full_end])
            output_count = last_output - first_output + 1
            return EvaluationContext(
                time_index=time_index,
                previous_time=bar_index[full_start - 1],
                universe=normalized_universe,
                primary_exchange=primary_exchange,
                frequency=validated_frequency,
                output_start=normalized_history,
                output_end=normalized_history + output_count,
                calendar_version=calendar_version,
            )

        raise ValueError("交易日历无法提供足够的历史、未来周期或首周期左边界")


def align_latest_observations(
    observations: pd.DataFrame,
    context: EvaluationContext,
) -> pd.DataFrame:
    """把跨市场观测按主时钟周期对齐为最后可得值。

    输入：index 为观测完成/可得时刻、columns 为资产的二维数据，以及主时钟上下文。
    输出：形状从 ``(观测数, 资产数)`` 转为
    ``(len(context.time_index), 资产数)``；每个 ``(period_start, time]``
    周期仅保留各列最后一个非空值，无新观测时为 NaN。
    """
    if not isinstance(observations, pd.DataFrame):
        raise TypeError("observations 必须是 pandas.DataFrame")
    if observations.columns.has_duplicates:
        raise ValueError("observations columns 不能包含重复值")

    try:
        observation_index = pd.DatetimeIndex(observations.index)
    except (TypeError, ValueError) as error:
        raise ValueError("observations index 必须是可解析的时间索引") from error
    if observation_index.tz is None:
        raise ValueError("observations index 必须包含观测可得时区")
    observation_index = observation_index.tz_convert(EVALUATION_TIMEZONE)
    if observation_index.has_duplicates:
        raise ValueError("observations index 不能包含重复值")

    # ************************************************************
    # 原始 DataFrame (T_observation, N) 按完成时刻排序并统一时区；
    # 输出 DataFrame (T_context, N) 的 index 切换为主时钟，columns 保持不变。
    # ************************************************************
    normalized = observations.copy(deep=True)
    normalized.index = observation_index
    normalized = normalized.sort_index()
    period_starts = pd.DatetimeIndex(context.period_start_index)
    period_ends = pd.DatetimeIndex(context.time_index)
    result: dict[Any, list[Any]] = {}

    for column in normalized.columns:
        available = normalized[column].dropna()
        available_times = pd.DatetimeIndex(available.index)
        values = np.full(len(period_ends), np.nan, dtype=object)
        if not available.empty:
            # ************************************************************
            # 每列可用观测 (T_available,) 通过 searchsorted 一次映射到全部
            # 主周期 (T_context,)；候选时刻必须严格晚于对应左边界。
            # ************************************************************
            positions = available_times.searchsorted(period_ends, side="right") - 1
            candidate_positions = np.maximum(positions, 0)
            has_new_observation = (positions >= 0) & (
                available_times.asi8[candidate_positions] > period_starts.asi8
            )
            values[has_new_observation] = available.to_numpy()[
                positions[has_new_observation]
            ]
        result[column] = values.tolist()

    aligned = pd.DataFrame(
        result,
        index=pd.DatetimeIndex(context.time_index),
        columns=observations.columns,
    )
    return aligned.infer_objects(copy=False)
