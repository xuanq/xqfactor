import pytest

from xqfactor import EvaluationContext, validate_frequency


def test_evaluation_context_is_explicit_and_hashable() -> None:
    """执行上下文应显式保存计算轴、规范频率、输出区间和数据语义。"""
    context = EvaluationContext(
        time_index=("t0", "t1", "t2"),
        universe=("A", "B"),
        frequency="min",
        output_start=1,
        semantics=(("adjust_type", "post"),),
    )

    assert context.output_time_index == ("t1", "t2")
    assert context.start_time == "t1"
    assert context.end_time == "t2"
    assert context.frequency == "min"
    assert context.fingerprint() == context.fingerprint()


@pytest.mark.parametrize("frequency", ["D", "min", "5min", "W-SUN", "ME"])
def test_validate_frequency_accepts_canonical_pandas_freqstr(frequency: str) -> None:
    """频率校验应接受 Pandas 规范 freqstr 并保持原值。"""
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


def test_evaluation_context_enforces_canonical_frequency() -> None:
    """手工构造上下文时也应执行统一频率校验。"""
    with pytest.raises(ValueError, match="W-SUN"):
        EvaluationContext(
            time_index=("t0",),
            universe=("A",),
            frequency="W",
        )
