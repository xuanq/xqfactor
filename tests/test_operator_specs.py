import pandas as pd

from xqfactor import (
    AbstractFactor,
    CombinedFactor,
    EvaluationContext,
    LeafFactor,
    LeafRequest,
    RollingWindowFactor,
)


def _leaf(name: str, offset: float = 0.0) -> LeafFactor:
    """创建带指定偏移量的测试叶子因子。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回形状为 (3 个时间点, 2 个资产) 的测试数据。"""
        return pd.DataFrame(
            [
                [1.0 + offset, 2.0 + offset],
                [2.0 + offset, 4.0 + offset],
                [3.0 + offset, 6.0 + offset],
            ],
            index=request.context.time_index,
            columns=request.context.universe,
        )

    return LeafFactor(name, resolver)


def _context(output_start: int = 0) -> EvaluationContext:
    """创建自定义算子测试上下文。"""
    return EvaluationContext(
        time_index=("t0", "t1", "t2"),
        universe=("A", "B"),
        frequency="D",
        output_start=output_start,
    )


def cross_sectional_demean(frame: pd.DataFrame) -> pd.DataFrame:
    """将每个时间截面的值减去该截面均值。"""
    return frame.sub(frame.mean(axis=1), axis=0)


def DEMEAN(factor: AbstractFactor) -> CombinedFactor:
    """将可复用的横截面去均值逻辑包装为因子算子。"""
    return CombinedFactor(cross_sectional_demean, factor)


def rolling_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """沿时间轴计算指定窗口均值。"""
    return frame.rolling(window).mean()


def MA2(factor: AbstractFactor) -> RollingWindowFactor:
    """将两期均值逻辑包装为窗口因子算子。"""
    return RollingWindowFactor(rolling_mean, 2, factor)


def test_custom_operator_can_be_reused_with_different_factors() -> None:
    """自定义算子定义不应绑定某一个具体因子实例。"""
    first = DEMEAN(_leaf("first"))
    second = DEMEAN(_leaf("second", offset=10.0))

    first_value = first.evaluate(_context())
    second_value = second.evaluate(_context())

    assert list(first_value.loc["t0"]) == [-0.5, 0.5]
    assert list(second_value.loc["t0"]) == [-0.5, 0.5]


def test_custom_rolling_operator_preserves_history_contract() -> None:
    """自定义窗口算子应声明并使用额外历史周期。"""
    rolling = MA2(_leaf("close"))
    result = rolling.evaluate(_context(output_start=1))

    assert rolling.required_history() == 1
    assert list(result["A"]) == [1.5, 2.5]
