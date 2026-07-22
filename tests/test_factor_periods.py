import gc
import weakref

import pandas as pd

from xqfactor import (
    REF,
    FactorPeriodRequirements,
    LeafFactor,
    LeafRequest,
    RollingWindowFactor,
    get_defined_factor_periods,
)


def _resolver(request: LeafRequest) -> pd.DataFrame:
    """返回与请求轴一致的空测试数据。

    输入：叶子因子的求值请求。
    输出：index 和 columns 分别与上下文时间轴、资产轴一致的 DataFrame。
    """
    return pd.DataFrame(
        index=request.context.time_index,
        columns=request.context.universe,
    )


def _rolling_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """计算测试用滚动均值。

    输入：形状为 ``(时间数, 资产数)`` 的因子值和窗口长度。
    输出：形状与两条轴均保持不变的滚动均值 DataFrame。
    """
    return frame.rolling(window).mean()


def test_defined_factor_periods_aggregate_all_live_nodes() -> None:
    """周期汇总应返回全部存活因子节点的最大历史和未来需求。"""
    baseline = get_defined_factor_periods()
    leaf = LeafFactor("close", _resolver)
    history_window = baseline.max_history + 2
    historical = RollingWindowFactor(_rolling_mean, history_window, leaf)
    future = REF(leaf, -(baseline.max_future + 1))

    requirements = get_defined_factor_periods()

    assert requirements == FactorPeriodRequirements(
        max_history=historical.required_history(),
        max_future=future.required_future(),
    )


def test_defined_factor_registry_does_not_prevent_collection() -> None:
    """弱引用登记不应延长因子实例生命周期。"""

    def create_factor() -> weakref.ReferenceType[LeafFactor]:
        """创建只由弱引用观察的临时因子。"""
        factor = LeafFactor("temporary", _resolver)
        return weakref.ref(factor)

    factor_reference = create_factor()
    gc.collect()

    assert factor_reference() is None
