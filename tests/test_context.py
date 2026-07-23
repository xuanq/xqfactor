import pandas as pd
import pytest

from xqfactor import (
    EVALUATION_TIMEZONE,
    EvaluationContext,
    align_latest_observations,
    validate_frequency,
)


def _context() -> EvaluationContext:
    """创建带首周期边界的分钟主时钟测试上下文。"""
    return EvaluationContext(
        time_index=tuple(
            pd.to_datetime(
                [
                    "2026-07-20 09:31",
                    "2026-07-20 09:32",
                    "2026-07-20 09:33",
                ]
            )
        ),
        previous_time="2026-07-17 15:00",
        universe=("A", "B"),
        primary_exchange="XSHG",
        frequency="min",
        output_start=1,
        calendar_version="fake-v1",
    )


def test_evaluation_context_normalizes_main_clock_and_output_slice() -> None:
    """执行上下文应统一上海时区并显式保存主时钟周期边界。"""
    context = _context()

    expected = tuple(
        pd.to_datetime(
            [
                "2026-07-20 09:31",
                "2026-07-20 09:32",
                "2026-07-20 09:33",
            ]
        ).tz_localize(EVALUATION_TIMEZONE)
    )
    assert context.time_index == expected
    assert context.previous_time == pd.Timestamp(
        "2026-07-17 15:00",
        tz=EVALUATION_TIMEZONE,
    )
    assert context.period_start_index == (
        context.previous_time,
        expected[0],
        expected[1],
    )
    assert context.output_time_index == expected[1:]
    assert context.start_time == expected[1]
    assert context.end_time == expected[2]
    assert context.timezone == EVALUATION_TIMEZONE
    assert context.fingerprint() == context.fingerprint()


def test_evaluation_context_converts_other_timezone_to_shanghai() -> None:
    """带时区输入应转换到上海时区而不是丢弃绝对时间。"""
    context = EvaluationContext(
        time_index=(pd.Timestamp("2026-07-20 01:31", tz="UTC"),),
        previous_time=pd.Timestamp("2026-07-20 01:30", tz="UTC"),
        universe=("A",),
        primary_exchange="XSHG",
        frequency="min",
    )

    assert context.time_index == (
        pd.Timestamp("2026-07-20 09:31", tz=EVALUATION_TIMEZONE),
    )


@pytest.mark.parametrize("frequency", ["D", "min", "5min", "W-SUN", "ME"])
def test_validate_frequency_accepts_canonical_pandas_freqstr(frequency: str) -> None:
    """频率校验应接受 Pandas 规范 freqstr并保持原值。"""
    assert validate_frequency(frequency) == frequency


@pytest.mark.parametrize(
    ("frequency", "canonical"),
    [
        ("d", "D"),
        ("1d", "D"),
        ("1m", "ME"),
        ("W", "W-SUN"),
        ("M", "ME"),
        ("T", "min"),
    ],
)
def test_validate_frequency_rejects_noncanonical_aliases(
    frequency: str,
    canonical: str,
) -> None:
    """频率校验应拒绝别名并提示对应规范值。"""
    with pytest.raises(ValueError, match=canonical):
        validate_frequency(frequency)


def test_validate_frequency_rejects_tick() -> None:
    """tick 不是 Pandas offset，不应作为 EvaluationContext.frequency。"""
    with pytest.raises(ValueError, match="不是有效"):
        validate_frequency("tick")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {
                "time_index": (
                    "2026-07-20 09:32",
                    "2026-07-20 09:31",
                )
            },
            "严格递增",
        ),
        (
            {
                "time_index": (
                    "2026-07-20 09:31",
                    "2026-07-20 09:31",
                )
            },
            "重复",
        ),
        ({"previous_time": "2026-07-20 09:31"}, "必须早于"),
        ({"frequency": "W"}, "W-SUN"),
        ({"primary_exchange": ""}, "primary_exchange"),
    ],
)
def test_evaluation_context_validates_main_clock(
    overrides: dict[str, object],
    message: str,
) -> None:
    """手工构造上下文时应拒绝歧义或非法主时钟。"""
    arguments: dict[str, object] = {
        "time_index": ("2026-07-20 09:31",),
        "previous_time": "2026-07-20 09:30",
        "universe": ("A",),
        "primary_exchange": "XSHG",
        "frequency": "min",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        EvaluationContext(**arguments)  # type: ignore[arg-type]


def test_align_latest_observations_uses_right_closed_main_periods() -> None:
    """对齐应包含右边界、排除左边界，并按列选择周期内最后非空观测。"""
    context = _context()
    observations = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, 4.0],
            "B": [10.0, float("nan"), 30.0, float("nan")],
        },
        index=pd.to_datetime(
            [
                "2026-07-17 15:00",
                "2026-07-20 09:30",
                "2026-07-20 09:31",
                "2026-07-20 09:33:01",
            ],
            format="mixed",
        ).tz_localize(EVALUATION_TIMEZONE),
    )

    result = align_latest_observations(observations, context)

    assert result.iloc[0]["A"] == 3.0
    assert pd.isna(result.iloc[1]["A"])
    assert pd.isna(result.iloc[2]["A"])
    assert result.iloc[0]["B"] == 30.0
    assert pd.isna(result.iloc[1]["B"])
    assert pd.isna(result.iloc[2]["B"])
    assert list(result.columns) == ["A", "B"]
    assert tuple(result.index) == context.time_index


def test_align_latest_observations_does_not_fill_empty_source_period() -> None:
    """来源市场一个主轴周期内无新观测时应返回 NaN。"""
    context = _context()
    observations = pd.DataFrame(
        {"A": [1.0, 2.0]},
        index=pd.to_datetime(
            ["2026-07-20 09:31", "2026-07-20 09:33"],
        ).tz_localize(EVALUATION_TIMEZONE),
    )

    result = align_latest_observations(observations, context)

    assert result.iloc[0, 0] == 1.0
    assert pd.isna(result.iloc[1, 0])
    assert result.iloc[2, 0] == 2.0


def test_align_latest_observations_rejects_naive_availability_time() -> None:
    """观测可得时刻缺少时区时不得猜测来源市场。"""
    with pytest.raises(ValueError, match="必须包含观测可得时区"):
        align_latest_observations(
            pd.DataFrame({"A": [1.0]}, index=["2026-07-20 09:31"]),
            _context(),
        )
