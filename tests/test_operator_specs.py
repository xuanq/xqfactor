import pandas as pd

from xqfactor import EvaluationContext, FactorRuntime, LeafFactor, rolling_operator
from xqfactor.backends import PandasBackend


def test_custom_rolling_operator_preserves_history_contract() -> None:
    """自定义窗口算子可由应用提供计算函数。"""
    factor = LeafFactor(
        "close",
        lambda request: pd.DataFrame(
            [[1.0], [2.0], [3.0]],
            index=request.context.time_index,
            columns=request.context.universe,
        ),
    )
    rolling = rolling_operator(
        factor,
        2,
        lambda frame, window: frame.rolling(window).mean(),
    )
    context = EvaluationContext(
        time_index=("t0", "t1", "t2"),
        universe=("A",),
        frequency="D",
        output_start=1,
    )

    result = rolling.evaluate(context, FactorRuntime(PandasBackend())).data

    assert rolling.required_history() == 1
    assert list(result["A"]) == [1.5, 2.5]
