import pandas as pd
import pytest

from xqfactor import (
    FIX,
    EvaluationContext,
    LeafFactor,
    LeafRequest,
    MemoryCache,
    PCT_CHANGE,
    RANK,
    REF,
)


def _context(output_start: int = 0) -> EvaluationContext:
    """构造包含历史数据的测试上下文。"""
    return EvaluationContext(
        time_index=("t0", "t1", "t2"),
        universe=("A", "B"),
        frequency="D",
        output_start=output_start,
    )


def test_leaf_factor_uses_resolver() -> None:
    """叶子因子应只调用应用提供的 resolver。"""
    calls: list[LeafRequest] = []

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回形状为 (3 个时间点, 2 个资产) 的测试收盘价。"""
        calls.append(request)
        return pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    value = factor.evaluate(_context())

    assert value.shape == (3, 2)
    assert calls[0].factor_name == "close"
    assert calls[0].context.frequency == "D"


def test_binary_expression_and_rank_use_dataframe_semantics() -> None:
    """基础表达式和 RANK 应直接按 DataFrame 语义执行。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回形状为 (3 个时间点, 2 个资产) 的测试收盘价。"""
        return pd.DataFrame(
            [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    value = (RANK(factor) + 1).evaluate(_context())

    assert value.loc["t0", "A"] == 1.5
    assert value.loc["t0", "B"] == 2.0


def test_ref_uses_history_and_returns_requested_slice() -> None:
    """REF 应在完整时间轴上移动，根节点再截取输出范围。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回形状为 (3 个时间点, 2 个资产) 的测试收盘价。"""
        return pd.DataFrame(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("close", resolver)
    value = REF(factor, 1).evaluate(_context(output_start=1))

    assert list(value.index) == ["t1", "t2"]
    assert list(value["A"]) == [1.0, 2.0]


def test_leaf_result_is_aligned_to_context_axes() -> None:
    """叶子结果应按上下文时间轴和资产轴重排并补充缺失值。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回顺序不同且缺少一个资产的二维数据。"""
        return pd.DataFrame(
            [[3.0], [1.0]],
            index=["t2", "t0"],
            columns=["B"],
        )

    value = LeafFactor("close", resolver).evaluate(_context())

    assert list(value.index) == ["t0", "t1", "t2"]
    assert list(value.columns) == ["A", "B"]
    assert pd.isna(value.loc["t1", "B"])
    assert value.loc["t2", "B"] == 3.0


def test_fix_uses_single_asset_context_and_broadcasts() -> None:
    """FIX 应在目标资产上下文取数，并将结果广播到当前资产池。"""
    requested_universes: list[tuple[str | int, ...]] = []

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回目标资产的三期测试值。"""
        requested_universes.append(request.context.universe)
        return pd.DataFrame(
            [[10.0], [11.0], [12.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        )

    value = FIX(LeafFactor("close", resolver), "INDEX").evaluate(_context())

    assert requested_universes == [("INDEX",)]
    assert list(value.columns) == ["A", "B"]
    assert value.loc["t1", "A"] == value.loc["t1", "B"] == 11.0


def test_fix_preserves_expression_history_and_cross_sectional_semantics() -> None:
    """FIX 应保留子表达式历史需求，并在单标的轴上执行横截面算子。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求资产轴一致的三期测试值。"""
        return pd.DataFrame(
            [[10.0], [12.0], [15.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        )

    leaf = LeafFactor("close", resolver)
    fixed_return = FIX(PCT_CHANGE(leaf, 1), "INDEX")
    fixed_rank = FIX(RANK(leaf), "INDEX")

    return_value = fixed_return.evaluate(_context(output_start=1))
    rank_value = fixed_rank.evaluate(_context())

    assert fixed_return.required_history() == 1
    assert list(return_value["A"]) == pytest.approx([0.2, 0.25])
    assert rank_value.loc["t0", "A"] == rank_value.loc["t0", "B"] == 1.0


def test_fix_reuses_single_asset_cache_across_current_universes() -> None:
    """不同当前 universe 求值时应复用相同目标资产的子因子缓存。"""
    calls = 0

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求资产轴一致的常数测试数据。"""
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            5.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    def context_with_universe(*universe: str) -> EvaluationContext:
        """创建指定当前资产池的测试上下文。"""
        return EvaluationContext(
            time_index=("t0", "t1", "t2"),
            universe=universe,
            frequency="D",
        )

    leaf = LeafFactor("close", resolver)
    fixed = FIX(leaf, "INDEX")
    cache = MemoryCache()

    first = fixed.evaluate(context_with_universe("A", "B"), cache)
    second = fixed.evaluate(context_with_universe("A", "C"), cache)

    assert calls == 1
    assert list(first.columns) == ["A", "B"]
    assert list(second.columns) == ["A", "C"]
    assert fixed.fingerprint() != FIX(leaf, "OTHER").fingerprint()


def test_fix_can_build_excess_return_expression() -> None:
    """FIX 应支持构造资产收益相对公共资产收益的超额收益。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """根据请求资产返回三期价格序列。"""
        prices = {
            "A": [100.0, 110.0, 121.0],
            "B": [200.0, 210.0, 231.0],
            "INDEX": [1000.0, 1020.0, 1040.0],
        }
        return pd.DataFrame(
            {asset: prices[asset] for asset in request.context.universe},
            index=request.context.time_index,
        )

    returns = PCT_CHANGE(LeafFactor("close", resolver), 1)
    excess_returns = returns - FIX(returns, "INDEX")
    value = excess_returns.evaluate(_context(output_start=1))

    assert list(value["A"]) == pytest.approx([0.08, 0.0803921568627451])
    assert list(value["B"]) == pytest.approx([0.03, 0.0803921568627451])
