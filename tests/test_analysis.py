from typing import Mapping

import pandas as pd

from xqfactor import EvaluationContext, FactorRuntime, LeafFactor
from xqfactor.analysis.pandas import AbstractAnalyzer, ICAnalyzer, Normalizer
from xqfactor.backends import PandasBackend


def _factor() -> LeafFactor:
    """创建用于检验流程测试的内存叶子因子。"""

    def resolver(request):
        """返回形状为 (3 个时间点, 3 个资产) 的测试因子。"""
        return pd.DataFrame(
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 2.0],
                [2.0, 3.0, 1.0],
            ],
            index=request.context.time_index,
            columns=request.context.universe,
        )

    return LeafFactor("factor", resolver)


def _context() -> EvaluationContext:
    """创建检验流程测试上下文。"""
    return EvaluationContext(
        time_index=("t0", "t1", "t2"),
        universe=("A", "B", "C"),
        frequency="D",
    )


def test_normalization_runs_before_ic_analysis() -> None:
    """处理器应先标准化因子，再由 IC 检验器计算相关系数。"""
    context = _context()
    runtime = FactorRuntime(PandasBackend())
    returns = pd.DataFrame(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.1, 0.2],
            [0.2, 0.3, 0.1],
        ],
        index=context.time_index,
        columns=context.universe,
    )
    analyzer = ICAnalyzer(
        returns=returns,
        context=context,
        runtime=runtime,
        keep_processed_results=True,
    )
    analyzer.register_processor("normalization", Normalizer())

    result = analyzer.analyze({"factor": _factor()})
    normalized = analyzer.processed_results[("factor", "normalization")]

    assert normalized.mean(axis=1).abs().max() < 1e-12
    assert result.data["factor"].tolist() == [1.0, 1.0, 1.0]


def test_custom_analyzer_receives_processed_dataframes() -> None:
    """自定义检验器应接收已经求值和预处理的 DataFrame。"""

    class CoverageAnalyzer(AbstractAnalyzer):
        """统计非缺失值覆盖率。"""

        def _analyze(self, factors: Mapping[str, pd.DataFrame]) -> pd.Series:
            """返回各因子的非缺失比例。"""
            return pd.Series(
                {
                    name: factor.notna().to_numpy().mean()
                    for name, factor in factors.items()
                }
            )

    analyzer = CoverageAnalyzer(
        context=_context(),
        runtime=FactorRuntime(PandasBackend()),
    )
    result = analyzer.analyze({"factor": _factor()})

    assert result["factor"] == 1.0
