import pandas as pd

from xqfactor import EvaluationContext, FactorRuntime, LeafFactor
from xqfactor.backends import PandasBackend


def _context(provider_version: str = "v1") -> EvaluationContext:
    """构造指定数据版本的执行上下文。"""
    return EvaluationContext(
        time_index=("t0", "t1"),
        universe=("A", "B"),
        frequency="D",
        provider_version=provider_version,
    )


def test_same_context_reuses_leaf_value() -> None:
    """相同上下文重复执行时应命中内存缓存。"""
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
    runtime = FactorRuntime(PandasBackend())
    factor.evaluate(_context(), runtime)
    factor.evaluate(_context(), runtime)

    assert calls == 1


def test_provider_version_and_universe_isolate_cache() -> None:
    """数据版本或资产池不同都不能错误复用缓存。"""
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
    runtime = FactorRuntime(PandasBackend())
    factor.evaluate(_context(), runtime)
    factor.evaluate(_context("v2"), runtime)
    factor.evaluate(
        EvaluationContext(time_index=("t0", "t1"), universe=("A",), frequency="D"),
        runtime,
    )

    assert calls == 3
