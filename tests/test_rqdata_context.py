from datetime import date

import pandas as pd
import pytest

from xqfactor import (
    EVALUATION_TIMEZONE,
    EvaluationContext,
    EvaluationContextBuilder,
    TradingCalendar,
)
from xqfactor.providers.rqdata import RQDataTradingCalendar


class FakeTradingDateProvider:
    """使用工作日模拟 RQData 中国市场交易日接口。"""

    def __init__(self) -> None:
        """创建覆盖测试日期范围的工作日日历。"""
        self.trading_dates = pd.bdate_range("2022-01-01", "2027-12-31")

    def get_trading_dates(
        self,
        start_date: date,
        end_date: date,
        market: str = "cn",
    ) -> list[date]:
        """返回指定自然日期范围内的工作日。

        输入：包含边界的开始日期、结束日期和市场标识。
        输出：按时间升序排列的工作日日期列表。
        """
        assert market == "cn"
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        selected = self.trading_dates[
            (self.trading_dates >= start) & (self.trading_dates <= end)
        ]
        return [timestamp.date() for timestamp in selected]


@pytest.fixture
def calendar() -> RQDataTradingCalendar:
    """返回注入内存交易日接口的 RQData 交易日历。"""
    return RQDataTradingCalendar(
        api=FakeTradingDateProvider(),
        version="fake-rqdata-v1",
    )


@pytest.fixture
def builder(calendar: RQDataTradingCalendar) -> EvaluationContextBuilder:
    """返回使用 RQData 交易日历的通用上下文构造器。"""
    return EvaluationContextBuilder(calendar)


def _timestamp(value: str) -> pd.Timestamp:
    """构造上海时区测试时间。"""
    return pd.Timestamp(value, tz=EVALUATION_TIMEZONE)


def test_rqdata_calendar_implements_trading_calendar_protocol(
    calendar: RQDataTradingCalendar,
) -> None:
    """RQData 适配器应实现可插拔 TradingCalendar 协议。"""
    assert isinstance(calendar, TradingCalendar)


def test_build_daily_context_with_exact_period_extensions(
    builder: EvaluationContextBuilder,
) -> None:
    """日频上下文应使用实际收盘时刻并精确保留两侧 bar。"""
    context = builder.build(
        start_date="2024-01-08",
        end_date="2024-01-10",
        universe=("000001.XSHE", "0700.HK", "ETH.binance"),
        primary_exchange="XSHG",
        history_period=2,
        future_period=1,
    )

    assert isinstance(context, EvaluationContext)
    assert context.frequency == "D"
    assert context.primary_exchange == "XSHG"
    assert context.previous_time == _timestamp("2024-01-03 15:00")
    assert context.time_index == tuple(
        pd.to_datetime(
            [
                "2024-01-04 15:00",
                "2024-01-05 15:00",
                "2024-01-08 15:00",
                "2024-01-09 15:00",
                "2024-01-10 15:00",
                "2024-01-11 15:00",
            ]
        ).tz_localize(EVALUATION_TIMEZONE)
    )
    assert context.output_start == 2
    assert context.output_end == 5
    assert context.output_time_index == tuple(
        pd.to_datetime(
            [
                "2024-01-08 15:00",
                "2024-01-09 15:00",
                "2024-01-10 15:00",
            ]
        ).tz_localize(EVALUATION_TIMEZONE)
    )
    assert context.calendar_version == "fake-rqdata-v1"


def test_build_requested_cross_session_minute_example(
    builder: EvaluationContextBuilder,
) -> None:
    """分钟主时钟应精确生成 2026-07-20 至 07-22 的 30/5 扩展示例。"""
    context = builder.build(
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
        history_period=30,
        future_period=5,
    )

    assert context.previous_time == _timestamp("2026-07-17 14:30")
    assert context.time_index[0] == _timestamp("2026-07-17 14:31")
    assert context.output_time_index[0] == _timestamp("2026-07-20 09:31")
    assert context.output_time_index[-1] == _timestamp("2026-07-22 15:00")
    assert context.time_index[-1] == _timestamp("2026-07-23 09:35")
    assert len(context.output_time_index) == 720
    assert len(context.time_index) == 755
    assert context.output_start == 30
    assert context.output_end == 750


def test_build_minute_context_uses_cn_stock_sessions(
    builder: EvaluationContextBuilder,
) -> None:
    """分钟上下文应生成 240 个日内 bar 并跨午休与交易日扩展。"""
    context = builder.build(
        start_date="2024-01-08",
        end_date="2024-01-08",
        universe=("000001.XSHE",),
        primary_exchange="XSHE",
        frequency="min",
        history_period=1,
        future_period=1,
    )

    assert len(context.time_index) == 242
    assert context.previous_time == _timestamp("2024-01-05 14:59")
    assert context.time_index[0] == _timestamp("2024-01-05 15:00")
    assert context.output_time_index[0] == _timestamp("2024-01-08 09:31")
    assert context.output_time_index[119] == _timestamp("2024-01-08 11:30")
    assert context.output_time_index[120] == _timestamp("2024-01-08 13:01")
    assert context.output_time_index[-1] == _timestamp("2024-01-08 15:00")
    assert context.time_index[-1] == _timestamp("2024-01-09 09:31")


def test_build_weekly_context_uses_last_trading_close(
    builder: EvaluationContextBuilder,
) -> None:
    """周频上下文应使用每个 W-SUN 周期最后交易日的收盘时刻。"""
    context = builder.build(
        start_date="2024-01-08",
        end_date="2024-01-21",
        universe=("000001.XSHE",),
        primary_exchange="XSHG",
        frequency="W-SUN",
        history_period=1,
        future_period=1,
    )

    assert context.time_index == tuple(
        pd.to_datetime(
            [
                "2024-01-05 15:00",
                "2024-01-12 15:00",
                "2024-01-19 15:00",
                "2024-01-26 15:00",
            ]
        ).tz_localize(EVALUATION_TIMEZONE)
    )
    assert context.output_time_index == tuple(
        pd.to_datetime(["2024-01-12 15:00", "2024-01-19 15:00"]).tz_localize(
            EVALUATION_TIMEZONE
        )
    )


def test_build_monthly_context_uses_last_trading_close(
    builder: EvaluationContextBuilder,
) -> None:
    """月频上下文应使用每个自然月最后交易日的收盘时刻。"""
    context = builder.build(
        start_date="2024-02-01",
        end_date="2024-03-31",
        universe=("000001.XSHE",),
        primary_exchange="XSHG",
        frequency="ME",
        history_period=1,
        future_period=1,
    )

    assert context.time_index == tuple(
        pd.to_datetime(
            [
                "2024-01-31 15:00",
                "2024-02-29 15:00",
                "2024-03-29 15:00",
                "2024-04-30 15:00",
            ]
        ).tz_localize(EVALUATION_TIMEZONE)
    )
    assert context.output_time_index == tuple(
        pd.to_datetime(["2024-02-29 15:00", "2024-03-29 15:00"]).tz_localize(
            EVALUATION_TIMEZONE
        )
    )


@pytest.mark.parametrize(
    ("overrides", "error_type", "message"),
    [
        ({"universe": ()}, ValueError, "universe 不能为空"),
        (
            {"universe": ("000001.XSHE", "000001.XSHE")},
            ValueError,
            "重复资产",
        ),
        (
            {"start_date": "2024-01-10", "end_date": "2024-01-08"},
            ValueError,
            "不能晚于",
        ),
        ({"history_period": -1}, ValueError, "history_period"),
        ({"future_period": True}, ValueError, "future_period"),
        ({"primary_exchange": ""}, ValueError, "primary_exchange"),
        ({"primary_exchange": "XHKG"}, NotImplementedError, "XSHG 和 XSHE"),
        ({"frequency": "d"}, ValueError, "D"),
        ({"frequency": "1d"}, ValueError, "D"),
        ({"frequency": "1m"}, ValueError, "ME"),
        ({"frequency": "W"}, ValueError, "W-SUN"),
        ({"frequency": "M"}, ValueError, "ME"),
        ({"frequency": "T"}, ValueError, "min"),
        ({"frequency": "tick"}, ValueError, "不是有效"),
        ({"frequency": "5min"}, NotImplementedError, "仅支持"),
    ],
)
def test_build_context_validates_inputs(
    builder: EvaluationContextBuilder,
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """通用构造器和日历适配器应分别校验公共参数及支持范围。"""
    arguments: dict[str, object] = {
        "start_date": "2024-01-08",
        "end_date": "2024-01-10",
        "primary_exchange": "XSHG",
        "frequency": "D",
        "universe": ("000001.XSHE",),
    }
    arguments.update(overrides)

    with pytest.raises(error_type, match=message):
        builder.build(**arguments)  # type: ignore[arg-type]


def test_build_context_rejects_range_without_target_bar(
    builder: EvaluationContextBuilder,
) -> None:
    """仅包含非交易日的日频范围应明确报错。"""
    with pytest.raises(ValueError, match="没有目标频率"):
        builder.build(
            start_date="2024-01-06",
            end_date="2024-01-07",
            universe=("000001.XSHE",),
            primary_exchange="XSHG",
        )
