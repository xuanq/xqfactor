"""因子 IC 检验及结果统计。"""

from __future__ import annotations

from typing import Mapping

import pandas as pd

from .base import AbstractAnalyzer, FactorInput


class ICAnalysisResult:
    """IC 序列及常用统计结果。"""

    def __init__(self, ic: pd.DataFrame) -> None:
        """保存各因子的逐期 IC。

        输入：index 为时间、columns 为因子名称的 IC DataFrame。
        输出：IC 结果对象。
        """
        self._ic = ic

    @property
    def data(self) -> pd.DataFrame:
        """返回逐期 IC DataFrame。"""
        return self._ic

    def ic(self, factor: str | list[str] | None = None) -> pd.DataFrame:
        """返回全部或指定因子的 IC。

        输入：单个因子名称、因子名称列表或 None。
        输出：只包含指定因子的逐期 IC DataFrame。
        """
        if factor is None:
            return self._ic
        columns = [factor] if isinstance(factor, str) else factor
        return self._ic[columns]

    def summary(self, factor: str | list[str] | None = None) -> pd.DataFrame:
        """返回 IC 均值、标准差、IR 和方向稳定性。

        输入：单个因子名称、因子名称列表或 None。
        输出：index 为因子名称、columns 为统计指标的 DataFrame。
        """
        ic = self.ic(factor)
        return pd.DataFrame(
            {
                "ic_mean": ic.mean(),
                "ic_std": ic.std(ddof=1),
                "ir": ic.mean().abs() / ic.std(ddof=1),
                "gt_zero_ratio": (ic > 0).mean(),
                "abs_gt0.02_ratio": (ic.abs() > 0.02).mean(),
            }
        )


class ICAnalyzer(AbstractAnalyzer):
    """计算因子与给定收益率之间的逐期横截面相关系数。"""

    def __init__(self, returns: FactorInput) -> None:
        """创建 IC 检验器。

        输入：收益率因子表达式或二维收益率 DataFrame。
        输出：IC 检验器实例。
        """
        super().__init__(returns=returns)

    def _analyze(
        self,
        factors: Mapping[str, pd.DataFrame],
        **inputs: pd.DataFrame,
    ) -> ICAnalysisResult:
        """计算每个因子的逐期 IC。

        输入：待检验因子映射，以及名为 returns 的收益率 DataFrame。
        输出：逐期 IC 及汇总统计结果。
        """
        returns = inputs["returns"]
        values = {
            name: factor.corrwith(returns, axis=1) for name, factor in factors.items()
        }
        return ICAnalysisResult(pd.DataFrame(values))


__all__ = ["ICAnalysisResult", "ICAnalyzer"]
