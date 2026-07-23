from typing import Mapping

import pandas as pd
import pytest

from xqfactor import EvaluationContext, LeafFactor, LeafRequest
from xqfactor.analysis import AbstractAnalyzer


class EchoAnalyzer(AbstractAnalyzer):
    """返回已求值主因子和附加输入的测试检验器。"""

    def _analyze(
        self,
        factors: Mapping[str, pd.DataFrame],
        **inputs: pd.DataFrame,
    ) -> tuple[Mapping[str, pd.DataFrame], Mapping[str, pd.DataFrame]]:
        """返回基类传入的两类 DataFrame 映射。"""
        return factors, inputs


def _context() -> EvaluationContext:
    """创建检验器基类测试上下文。"""
    return EvaluationContext(
        time_index=tuple(
            pd.date_range(
                "2024-01-02 15:00",
                periods=2,
                freq="D",
                tz="Asia/Shanghai",
            )
        ),
        previous_time="2024-01-01 15:00",
        universe=("A", "B"),
        primary_exchange="XSHG",
        frequency="D",
    )


def test_analyzer_copies_direct_dataframe_inputs() -> None:
    """直接传入的 DataFrame 应复制后再交给统计逻辑。"""
    frame = pd.DataFrame([[1.0]], index=["t0"], columns=["A"])
    factors, _ = EchoAnalyzer().analyze({"factor": frame})

    factors["factor"].loc["t0", "A"] = 2.0

    assert frame.loc["t0", "A"] == 1.0


def test_factor_input_requires_context() -> None:
    """检验因子表达式但未提供 context 时应明确报错。"""

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """返回与请求轴一致的常数 DataFrame。"""
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    with pytest.raises(ValueError, match="必须提供 context"):
        EchoAnalyzer().analyze({"factor": LeafFactor("factor", resolver)})


def test_main_and_additional_inputs_share_execution_cache() -> None:
    """主因子和附加输入引用同一节点时应只调用一次 resolver。"""
    calls = 0

    def resolver(request: LeafRequest) -> pd.DataFrame:
        """记录调用次数并返回与请求轴一致的 DataFrame。"""
        nonlocal calls
        calls += 1
        return pd.DataFrame(
            1.0,
            index=request.context.time_index,
            columns=request.context.universe,
        )

    factor = LeafFactor("factor", resolver)
    factors, inputs = EchoAnalyzer(reference=factor).analyze(
        {"factor": factor},
        context=_context(),
    )

    assert calls == 1
    pd.testing.assert_frame_equal(factors["factor"], inputs["reference"])


def test_abstract_analyzer_cannot_be_instantiated() -> None:
    """AbstractAnalyzer 应是不能直接实例化的真实抽象类。"""
    with pytest.raises(TypeError):
        AbstractAnalyzer()
