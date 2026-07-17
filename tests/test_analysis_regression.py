import numpy as np
import pandas as pd
import pytest

from xqfactor.analysis.regression import RegressionAnalyzer


def _regression_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """创建系数恒为二的因子、收益率和权重数据。"""
    factor = pd.DataFrame(
        [[1.0, 2.0, 3.0], [1.0, 2.0, np.nan]],
        index=["t0", "t1"],
        columns=["A", "B", "C"],
    )
    returns = factor * 2.0
    weights = pd.DataFrame(
        [[1.0, 2.0, 3.0], [2.0, 1.0, 1.0]],
        index=factor.index,
        columns=factor.columns,
    )
    return factor, returns, weights


@pytest.mark.parametrize("model", ["OLS", "WLS"])
def test_regression_analyzer_returns_expected_coefficients(model: str) -> None:
    """OLS 和 WLS 都应返回逐期预期因子系数。"""
    factor, returns, weights = _regression_inputs()
    analyzer = RegressionAnalyzer(
        returns,
        weights=weights if model == "WLS" else None,
        model=model,
    )

    coefficients = analyzer.analyze({"factor": factor}).coefficients()

    assert coefficients["factor"].tolist() == pytest.approx([2.0, 2.0])


def test_wls_uses_unit_weights_when_weights_are_omitted() -> None:
    """WLS 未显式提供权重时应使用单位权重。"""
    factor, returns, _ = _regression_inputs()

    coefficients = (
        RegressionAnalyzer(returns, model="WLS")
        .analyze({"factor": factor})
        .coefficients()
    )

    assert coefficients["factor"].tolist() == pytest.approx([2.0, 2.0])


def test_regression_rejects_unknown_model() -> None:
    """不支持的回归模型应在构造检验器时立即报错。"""
    _, returns, _ = _regression_inputs()

    with pytest.raises(ValueError, match="不支持的回归模型"):
        RegressionAnalyzer(returns, model="unknown")


def test_coefficients_preserve_nan_for_misaligned_factor_dates() -> None:
    """多个因子有效日期不一致时系数结果应保留 NaN 而不是报错。"""
    returns = pd.DataFrame(
        [[2.0, 4.0], [2.0, 4.0]],
        index=["t0", "t1"],
        columns=["A", "B"],
    )
    first = pd.DataFrame(
        [[1.0, 2.0], [np.nan, np.nan]],
        index=returns.index,
        columns=returns.columns,
    )
    second = pd.DataFrame(
        [[np.nan, np.nan], [1.0, 2.0]],
        index=returns.index,
        columns=returns.columns,
    )

    coefficients = (
        RegressionAnalyzer(returns, model="OLS")
        .analyze({"first": first, "second": second})
        .coefficients()
    )

    assert coefficients.loc["t0", "first"] == pytest.approx(2.0)
    assert np.isnan(coefficients.loc["t0", "second"])
    assert np.isnan(coefficients.loc["t1", "first"])
    assert coefficients.loc["t1", "second"] == pytest.approx(2.0)
