from datetime import date

import pandas as pd
import pytest

from xqfactor import EvaluationContext, EvaluationContextBuilder
from xqfactor.providers.rqdata import RQDataContextBuilder


class FakeTradingCalendar:
    """使用工作日模拟 RQData 中国市场交易日历。"""

    def __init__(self) -> None:
        """创建覆盖测试日期范围的工作日日历。"""
        self.trading_dates = pd.bdate_range("2023-01-01", "2025-12-31")

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
def calendar() -> FakeTradingCalendar:
    """返回测试使用的内存交易日历。"""
    return FakeTradingCalendar()


@pytest.fixture
def builder(calendar: FakeTradingCalendar) -> RQDataContextBuilder:
    """返回注入内存交易日历的 RQData 上下文构造器。"""
    return RQDataContextBuilder(api=calendar)


def test_build_daily_context_with_exact_period_extensions(
    builder: RQDataContextBuilder,
) -> None:
    """日频上下文应按交易日精确保留历史和未来 bar。"""
    context_builder: EvaluationContextBuilder = builder
    assert isinstance(builder, EvaluationContextBuilder)
    context = context_builder.build(
        start_date="2024-01-08",
        end_date="2024-01-10",
        universe=("000001.XSHE", "600000.XSHG"),
        history_period=2,
        future_period=1,
    )

    assert isinstance(context, EvaluationContext)
    assert context.frequency == "D"
    assert context.time_index == tuple(
        pd.to_datetime(
            [
                "2024-01-04",
                "2024-01-05",
                "2024-01-08",
                "2024-01-09",
                "2024-01-10",
                "2024-01-11",
            ]
        )
    )
    assert context.output_start == 2
    assert context.output_end == 5
    assert context.output_time_index == tuple(
        pd.to_datetime(["2024-01-08", "2024-01-09", "2024-01-10"])
    )
    assert dict(context.semantics) == {"market": "cn", "type": "stock"}
    assert context.provider_version == "rqdata"


def test_build_minute_context_uses_cn_stock_sessions(
    builder: RQDataContextBuilder,
) -> None:
    """分钟上下文应生成 240 个日内 bar 并跨交易日扩展。"""
    context = builder.build(
        start_date="2024-01-08",
        end_date="2024-01-08",
        universe=("000001.XSHE",),
        frequency="min",
        history_period=1,
        future_period=1,
    )

    assert len(context.time_index) == 242
    assert context.time_index[0] == pd.Timestamp("2024-01-05 15:00:00")
    assert context.output_time_index[0] == pd.Timestamp("2024-01-08 09:31:00")
    assert context.output_time_index[119] == pd.Timestamp("2024-01-08 11:30:00")
    assert context.output_time_index[120] == pd.Timestamp("2024-01-08 13:01:00")
    assert context.output_time_index[-1] == pd.Timestamp("2024-01-08 15:00:00")
    assert context.time_index[-1] == pd.Timestamp("2024-01-09 09:31:00")


def test_build_weekly_context_uses_last_trading_day(
    builder: RQDataContextBuilder,
) -> None:
    """周频上下文应使用每个 W-SUN 周期的最后交易日。"""
    context = builder.build(
        start_date="2024-01-08",
        end_date="2024-01-21",
        universe=("000001.XSHE",),
        frequency="W-SUN",
        history_period=1,
        future_period=1,
    )

    assert context.time_index == tuple(
        pd.to_datetime(["2024-01-05", "2024-01-12", "2024-01-19", "2024-01-26"])
    )
    assert context.output_time_index == tuple(
        pd.to_datetime(["2024-01-12", "2024-01-19"])
    )


def test_build_monthly_context_uses_last_trading_day(
    builder: RQDataContextBuilder,
) -> None:
    """月频上下文应使用每个自然月的最后交易日。"""
    context = builder.build(
        start_date="2024-02-01",
        end_date="2024-03-31",
        universe=("000001.XSHE",),
        frequency="ME",
        history_period=1,
        future_period=1,
    )

    assert context.time_index == tuple(
        pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-29", "2024-04-30"])
    )
    assert context.output_time_index == tuple(
        pd.to_datetime(["2024-02-29", "2024-03-29"])
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
        ({"market": "invalid"}, ValueError, "market"),
        ({"type": "invalid"}, ValueError, "type"),
        ({"frequency": "d"}, ValueError, "D"),
        ({"frequency": "1d"}, ValueError, "D"),
        ({"frequency": "1m"}, ValueError, "ME"),
        ({"frequency": "W"}, ValueError, "W-SUN"),
        ({"frequency": "M"}, ValueError, "ME"),
        ({"frequency": "T"}, ValueError, "min"),
        ({"frequency": "tick"}, ValueError, "不是有效"),
        ({"frequency": "5min"}, NotImplementedError, "仅支持"),
        ({"market": "hk"}, NotImplementedError, "中国市场股票"),
        ({"type": "futures"}, NotImplementedError, "中国市场股票"),
    ],
)
def test_build_context_validates_inputs(
    builder: RQDataContextBuilder,
    overrides: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    """上下文构造函数应区分非法参数和尚未实现的合法组合。"""
    arguments: dict[str, object] = {
        "start_date": "2024-01-08",
        "end_date": "2024-01-10",
        "market": "cn",
        "type": "stock",
        "frequency": "D",
        "universe": ("000001.XSHE",),
    }
    arguments.update(overrides)

    with pytest.raises(error_type, match=message):
        builder.build(**arguments)  # type: ignore[arg-type]


def test_build_context_rejects_range_without_target_bar(
    builder: RQDataContextBuilder,
) -> None:
    """仅包含非交易日的日频范围应明确报错。"""
    with pytest.raises(ValueError, match="没有目标频率"):
        builder.build(
            start_date="2024-01-06",
            end_date="2024-01-07",
            universe=("000001.XSHE",),
        )
