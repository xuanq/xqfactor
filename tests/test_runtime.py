import pytest

from xqfactor import EvaluationContext, FactorRuntime, MemoryCache
from xqfactor.backends import PandasBackend


def test_evaluation_context_is_explicit_and_hashable() -> None:
    """执行上下文不依赖交易日历或数据 API。"""
    context = EvaluationContext(
        time_index=("t0", "t1", "t2"),
        universe=("A", "B"),
        frequency="1m",
        output_start=1,
    )

    assert context.output_time_index == ("t1", "t2")
    assert context.start_time == "t1"
    assert context.end_time == "t2"


def test_memory_cache_rejects_invalid_size() -> None:
    """缓存容量必须为正整数。"""
    with pytest.raises(ValueError):
        MemoryCache(maxsize=0)


def test_runtime_accepts_replaceable_backend() -> None:
    """运行时只要求后端协议，不再要求 xqdata。"""
    runtime = FactorRuntime(PandasBackend(), MemoryCache())
    assert runtime.backend.name == "pandas"
