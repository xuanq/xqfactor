import pandas as pd
import pytest

from xqfactor.analysis.quantile_return import QuantileReturnAnalyzer


def test_quantile_return_handles_ties_and_missing_values() -> None:
    """分组收益应稳定处理并列因子值和缺失资产。"""
    factor = pd.DataFrame(
        [[1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 3.0, float("nan")]],
        index=["t0", "t1"],
        columns=["A", "B", "C", "D"],
    )
    returns = pd.DataFrame(
        [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]],
        index=factor.index,
        columns=factor.columns,
    )

    result = QuantileReturnAnalyzer(returns, n_groups=2).analyze({"factor": factor})

    assert result.data.loc["t0", ("factor", 1)] == pytest.approx(0.15)
    assert result.data.loc["t0", ("factor", 2)] == pytest.approx(0.35)
    assert result.data.loc["t1", ("factor", 1)] == pytest.approx(0.15)
    assert result.data.loc["t1", ("factor", 2)] == pytest.approx(0.3)


def test_quantile_long_short_uses_highest_minus_lowest_group() -> None:
    """多空收益应等于最高组收益减最低组收益。"""
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0]],
        index=["t0"],
        columns=["A", "B", "C", "D"],
    )
    returns = pd.DataFrame(
        [[0.1, 0.2, 0.4, 0.6]],
        index=factor.index,
        columns=factor.columns,
    )

    result = QuantileReturnAnalyzer(returns, n_groups=2).analyze({"factor": factor})

    assert result.long_short().loc["t0", "factor"] == pytest.approx(0.35)


def test_quantile_return_rejects_one_group() -> None:
    """分组数量小于二时应明确报错。"""
    returns = pd.DataFrame([[0.1]], index=["t0"], columns=["A"])

    with pytest.raises(ValueError, match="至少为 2"):
        QuantileReturnAnalyzer(returns, n_groups=1)


def test_long_short_uses_actual_highest_group_when_assets_are_insufficient() -> None:
    """有效资产少于目标组数时应使用该期实际最高组计算多空收益。"""
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=["t0"],
        columns=["A", "B", "C"],
    )
    returns = pd.DataFrame(
        [[0.1, 0.2, 0.3]],
        index=factor.index,
        columns=factor.columns,
    )

    result = QuantileReturnAnalyzer(returns, n_groups=5).analyze({"factor": factor})

    assert result.long_short().loc["t0", "factor"] == pytest.approx(0.2)


def test_long_short_preserves_missing_actual_highest_group_return() -> None:
    """实际最高组收益缺失时多空收益应为 NaN，不能退回次高组。"""
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0]],
        index=["t0"],
        columns=["A", "B", "C"],
    )
    returns = pd.DataFrame(
        [[0.1, 0.2, float("nan")]],
        index=factor.index,
        columns=factor.columns,
    )

    result = QuantileReturnAnalyzer(returns, n_groups=5).analyze({"factor": factor})

    assert pd.isna(result.long_short().loc["t0", "factor"])
