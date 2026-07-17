import pandas as pd

from xqfactor import EvaluationContext, LeafFactor, LeafRequest, RANK, REF


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
