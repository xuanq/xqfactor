"""逐期横截面因子收益回归及结果统计。"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.api import OLS, WLS

from .base import AbstractAnalyzer, FactorInput


class RegressionAnalysisResult:
    """逐期横截面回归结果。"""

    def __init__(self, results: pd.DataFrame) -> None:
        """保存各期拟合结果对象。

        输入：index 为时间、columns 为因子名称的拟合结果 DataFrame。
        输出：回归分析结果对象。
        """
        self._results = results

    @property
    def data(self) -> pd.DataFrame:
        """返回原始回归结果对象。"""
        return self._results

    def coefficients(self) -> pd.DataFrame:
        """返回各因子的逐期回归系数。

        输入：无。
        输出：index 为时间、columns 为因子名称的系数 DataFrame。
        """
        return self._results.map(
            lambda result: result.params.loc["factor"],
            na_action="ignore",
        )

    def summary(self) -> pd.DataFrame:
        """返回平均因子收益和跨期 t 检验结果。

        输入：无。
        输出：index 为因子名称、columns 为统计指标的 DataFrame。
        """
        coefficients = self.coefficients()
        test = ttest_1samp(coefficients, 0, nan_policy="omit")
        return pd.DataFrame(
            {
                "return_mean": coefficients.mean(),
                "t_stat": test.statistic,
                "p_value": test.pvalue,
            },
            index=coefficients.columns,
        )


class RegressionAnalyzer(AbstractAnalyzer):
    """执行逐期横截面 OLS 或 WLS 因子收益回归。"""

    def __init__(
        self,
        returns: FactorInput,
        weights: FactorInput | None = None,
        model: str = "WLS",
    ) -> None:
        """创建横截面回归检验器。

        输入：收益率、可选权重以及 OLS/WLS 模型名称。
        输出：横截面回归检验器实例。
        """
        normalized_model = model.upper()
        if normalized_model not in {"OLS", "WLS"}:
            raise ValueError(f"不支持的回归模型: {model}")
        super().__init__(returns=returns, weights=weights)
        self.model = normalized_model

    def _analyze(
        self,
        factors: Mapping[str, pd.DataFrame],
        **inputs: pd.DataFrame,
    ) -> RegressionAnalysisResult:
        """分别回归每个输入因子并返回逐期拟合对象。

        输入：待检验因子映射、returns 和可选 weights DataFrame。
        输出：每个因子、每个时间点的拟合结果。
        """
        returns = inputs["returns"]
        weights = inputs.get("weights")
        outputs: dict[str, pd.Series] = {}
        for name, factor in factors.items():
            fitted: dict[Any, Any] = {}
            for timestamp in factor.index:
                frame = pd.DataFrame(
                    {
                        "factor": factor.loc[timestamp],
                        "returns": returns.loc[timestamp],
                    }
                )
                if weights is not None:
                    frame["weights"] = weights.loc[timestamp]
                frame = frame.dropna()
                if frame.empty:
                    continue

                if self.model == "OLS":
                    result = OLS(frame["returns"], frame[["factor"]]).fit()
                else:
                    weight_values = (
                        frame["weights"] if "weights" in frame else np.ones(len(frame))
                    )
                    result = WLS(
                        frame["returns"],
                        frame[["factor"]],
                        weights=weight_values,
                    ).fit()
                fitted[timestamp] = result
            outputs[name] = pd.Series(fitted)
        return RegressionAnalysisResult(pd.DataFrame(outputs))


__all__ = ["RegressionAnalysisResult", "RegressionAnalyzer"]
