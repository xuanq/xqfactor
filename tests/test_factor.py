import pandas as pd

from xqfactor import (
    EvaluationContext,
    FactorRuntime,
    LeafFactor,
    RANK,
    REF,
)
from xqfactor.backends import PandasBackend


def _context(output_start: int = 0) -> EvaluationContext:
    """构造包含历史数据的测试上下文。"""
    return EvaluationContext(
        time_index=("t0", "t1", "t2"),
        universe=("A", "B"),
        frequency="D",
        output_start=output_start,
    )


def test_leaf_factor_uses_resolver() -> None:
    """叶子因子只调用应用提供的 resolver。"""
    calls = []

    def resolver(request):
        calls.append(request)
        return pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    value = factor.evaluate(_context(), FactorRuntime(PandasBackend())).data

    assert value.shape == (3, 2)
    assert calls[0].factor_name == "close"
    assert calls[0].context.frequency == "D"


def test_binary_expression_and_rank_are_backend_computed() -> None:
    """二元表达式和 RANK 由参考后端执行。"""
    factor = LeafFactor(
        "close",
        lambda request: pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        ),
    )
    runtime = FactorRuntime(PandasBackend())
    value = (RANK(factor) + 1).evaluate(_context(), runtime).data

    assert value.loc["t0", "A"] == 1.5
    assert value.loc["t0", "B"] == 2.0


def test_ref_uses_history_and_returns_requested_slice() -> None:
    """REF 在完整时间轴上移动，根节点再截取输出范围。"""
    factor = LeafFactor(
        "close",
        lambda request: pd.DataFrame(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        ),
    )
    value = (
        REF(factor, 1)
        .evaluate(_context(output_start=1), FactorRuntime(PandasBackend()))
        .data
    )

    assert list(value.index) == ["t1", "t2"]
    assert list(value["A"]) == [1.0, 2.0]


def test_shared_leaf_is_computed_once_in_one_runtime() -> None:
    """同一因子图共享叶子节点时只执行一次 resolver。"""
    calls = 0

    def resolver(request):
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    (factor + factor).evaluate(_context(), FactorRuntime(PandasBackend()))

    assert calls == 1
