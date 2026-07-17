import pandas as pd
from functools import partial

from xqfactor import (
    CombinedFactor,
    ConstantFactor,
    EvaluationContext,
    LeafFactor,
    LeafRequest,
    MemoryCache,
)


def _context(provider_version: str = "v1") -> EvaluationContext:
    """构造指定数据版本的执行上下文。"""
    return EvaluationContext(
        time_index=("t0", "t1"),
        universe=("A", "B"),
        frequency="D",
        provider_version=provider_version,
    )


def test_same_context_reuses_leaf_value() -> None:
    """相同上下文和共享缓存重复执行时应命中内存缓存。"""
    calls = 0

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回形状为 (2 个时间点, 2 个资产) 的测试数据。"""
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    cache = MemoryCache()
    factor.evaluate(_context(), cache)
    factor.evaluate(_context(), cache)

    assert calls == 1


def test_provider_version_and_universe_isolate_cache() -> None:
    """数据版本或资产池不同都不能错误复用缓存。"""
    calls = 0

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求轴一致的常数 DataFrame。"""
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    cache = MemoryCache()
    factor.evaluate(_context(), cache)
    factor.evaluate(_context("v2"), cache)
    factor.evaluate(
        EvaluationContext(
            time_index=("t0", "t1"),
            universe=("A",),
            frequency="D",
        ),
        cache,
    )

    assert calls == 3


def test_shared_leaf_is_computed_once_in_expression_graph() -> None:
    """同一因子图共享叶子节点时应只执行一次 resolver。"""
    calls = 0

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求轴一致的常数 DataFrame。"""
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    (factor + factor).evaluate(_context())

    assert calls == 1


def test_parameterized_closures_have_independent_cache_fingerprints() -> None:
    """同一工厂函数创建的不同闭包算子不应错误共享缓存。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求轴一致的常数 DataFrame。"""
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    def scaled_factor(scale: float) -> CombinedFactor:
        """创建捕获指定缩放倍数的自定义因子。"""

        def scale_frame(frame: pd.DataFrame) -> pd.DataFrame:
            """将二维因子值乘以闭包中的缩放倍数。"""
            return frame * scale

        return CombinedFactor(scale_frame, LeafFactor("close", resolver))

    doubled = scaled_factor(2.0)
    tripled = scaled_factor(3.0)
    cache = MemoryCache()

    doubled_value = doubled.evaluate(_context(), cache)
    tripled_value = tripled.evaluate(_context(), cache)

    assert doubled.fingerprint() != tripled.fingerprint()
    assert doubled_value.iloc[0, 0] == 2.0
    assert tripled_value.iloc[0, 0] == 3.0


def test_definition_version_change_invalidates_leaf_cache() -> None:
    """显式修改叶子定义版本后不应继续命中旧缓存。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回由当前定义版本决定的二维测试数据。"""
        return pd.DataFrame(
            float(request.definition_version),
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver, definition_version="1")
    cache = MemoryCache()
    first = factor.evaluate(_context(), cache)
    factor.definition_version = "2"
    second = factor.evaluate(_context(), cache)

    assert first.iloc[0, 0] == 1.0
    assert second.iloc[0, 0] == 2.0


def test_callable_runtime_state_does_not_change_fingerprint() -> None:
    """resolver 的调用计数变化不应让同一节点的缓存键漂移。"""
    calls = 0

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """累计调用次数并返回二维测试数据。"""
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    fingerprint = factor.fingerprint()
    cache = MemoryCache()
    factor.evaluate(_context(), cache)
    factor.evaluate(_context(), cache)

    assert factor.fingerprint() == fingerprint
    assert calls == 1


def test_partial_callable_has_stable_parameterized_fingerprint() -> None:
    """偏函数应可用作算子，且不同绑定参数应生成不同指纹。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求轴一致的常数 DataFrame。"""
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    def scale(frame: pd.DataFrame, multiplier: float) -> pd.DataFrame:
        """将二维因子值乘以给定倍数。"""
        return frame * multiplier

    leaf = LeafFactor("close", resolver)
    doubled = CombinedFactor(partial(scale, multiplier=2.0), leaf)
    tripled = CombinedFactor(partial(scale, multiplier=3.0), leaf)

    assert doubled.fingerprint() != tripled.fingerprint()
    assert doubled.evaluate(_context()).iloc[0, 0] == 2.0
    assert tripled.evaluate(_context()).iloc[0, 0] == 3.0


def test_nested_callable_closure_values_affect_fingerprint() -> None:
    """闭包捕获的 callable 定义不同应生成不同算子指纹。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求轴一致的常数 DataFrame。"""
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    def operation(multiplier: float):
        """创建捕获缩放倍数的 DataFrame 操作函数。"""
        return lambda frame: frame * multiplier

    def wrapper(inner):
        """创建捕获另一个 callable 的 DataFrame 操作函数。"""
        return lambda frame: inner(frame)

    leaf = LeafFactor("close", resolver)
    doubled = CombinedFactor(wrapper(operation(2.0)), leaf)
    tripled = CombinedFactor(wrapper(operation(3.0)), leaf)

    assert doubled.fingerprint() != tripled.fingerprint()


def test_dataframe_constants_do_not_collide_when_hidden_values_differ() -> None:
    """大表未展示区域的差异也必须参与常量因子缓存指纹。"""
    first_value = pd.DataFrame(0.0, index=range(100), columns=range(100))
    second_value = first_value.copy()
    second_value.iloc[50, 50] = 1.0
    first = ConstantFactor(first_value)
    second = ConstantFactor(second_value)

    assert repr(first_value) == repr(second_value)
    assert first.fingerprint() != second.fingerprint()


def test_mapping_key_types_do_not_collide_in_operator_arguments() -> None:
    """整数键与同文本字符串键并存时必须生成独立的算子缓存指纹。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求轴一致的常数 DataFrame。"""
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    def mapping_size(
        frame: pd.DataFrame,
        *,
        config: dict[object, str],
    ) -> pd.DataFrame:
        """将二维因子值乘以映射键数量。"""
        return frame * len(config)

    leaf = LeafFactor("close", resolver)
    two_keys = CombinedFactor(mapping_size, leaf, config={1: "int", "1": "str"})
    one_key = CombinedFactor(mapping_size, leaf, config={"1": "str"})
    cache = MemoryCache()

    assert two_keys.fingerprint() != one_key.fingerprint()
    assert two_keys.evaluate(_context(), cache).iloc[0, 0] == 2.0
    assert one_key.evaluate(_context(), cache).iloc[0, 0] == 1.0


def test_replacing_callable_invalidates_cached_factor_value() -> None:
    """替换节点计算函数后必须使用新指纹重新计算结果。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求轴一致的常数 DataFrame。"""
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    def double(frame: pd.DataFrame) -> pd.DataFrame:
        """将二维因子值乘以二。"""
        return frame * 2.0

    def triple(frame: pd.DataFrame) -> pd.DataFrame:
        """将二维因子值乘以三。"""
        return frame * 3.0

    factor = CombinedFactor(double, LeafFactor("close", resolver))
    cache = MemoryCache()
    first = factor.evaluate(_context(), cache)
    factor.func = triple
    second = factor.evaluate(_context(), cache)

    assert first.iloc[0, 0] == 2.0
    assert second.iloc[0, 0] == 3.0
