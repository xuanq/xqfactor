import numpy as np
import pandas as pd
import pytest

from xqfactor.analysis.ic import ICAnalyzer


def test_ic_analyzer_supports_multiple_factors_and_nan() -> None:
    """IC 检验应计算多个因子并按配对有效值忽略 NaN。"""
    index = ["t0", "t1"]
    columns = ["A", "B", "C"]
    returns = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 2.0, np.nan]],
        index=index,
        columns=columns,
    )
    positive = returns.copy()
    negative = -returns

    result = ICAnalyzer(returns).analyze({"positive": positive, "negative": negative})

    assert result.data["positive"].tolist() == pytest.approx([1.0, 1.0])
    assert result.data["negative"].tolist() == pytest.approx([-1.0, -1.0])


def test_ic_summary_can_select_one_factor() -> None:
    """IC 汇总应支持只选择指定因子。"""
    returns = pd.DataFrame(
        [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]],
        index=["t0", "t1"],
        columns=["A", "B", "C"],
    )
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
        index=returns.index,
        columns=returns.columns,
    )

    summary = ICAnalyzer(returns).analyze({"factor": factor}).summary("factor")

    assert list(summary.index) == ["factor"]
    assert summary.loc["factor", "ic_mean"] == 0.0
    assert summary.loc["factor", "gt_zero_ratio"] == 0.5
