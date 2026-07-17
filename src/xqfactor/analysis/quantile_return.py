"""因子分组收益检验及结果统计。"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .base import AbstractAnalyzer, FactorInput


def _assign_quantiles(row: pd.Series, n_groups: int) -> pd.Series:
    """将一个时间截面的有效资产划分为近似等量分组。

    输入：index 为资产的单期因子值，以及目标分组数量。
    输出：index 保持不变、值为组号或 NaN 的 Series。
    """
    result = pd.Series(np.nan, index=row.index, dtype=float)
    valid = row.dropna()
    if valid.empty:
        return result

    # ************************************************************
    # Series (资产数,) 先按原 columns 顺序打破并列值，再按有效资产数限制
    # 实际分组数；输出恢复原资产 index，缺失资产仍为 NaN。
    # ************************************************************
    actual_groups = min(n_groups, len(valid))
    ranked = valid.rank(method="first")
    labels = range(1, actual_groups + 1)
    result.loc[valid.index] = pd.qcut(
        ranked,
        actual_groups,
        labels=labels,
    ).astype(float)
    return result


def _long_short_return(
    row: pd.Series,
    lowest_group: float,
    highest_group: float,
) -> float:
    """计算单期实际最高组减最低组的收益。

    输入：index 为组号的单期收益，以及因子值实际形成的最低、最高组号。
    输出：多空收益；不足两个组或任一端收益缺失时返回 NaN。
    """
    if pd.isna(lowest_group) or pd.isna(highest_group):
        return float("nan")
    if lowest_group == highest_group:
        return float("nan")
    return float(row.loc[int(highest_group)] - row.loc[int(lowest_group)])


class QuantileReturnResult:
    """分组收益分析结果。"""

    def __init__(
        self,
        quantile_returns: pd.DataFrame,
        group_bounds: Mapping[str, pd.DataFrame],
    ) -> None:
        """保存因子分组收益。

        输入：逐期分组收益，以及各因子每期实际形成的最低、最高组号。
        输出：分组收益结果对象。
        """
        self._quantile_returns = quantile_returns
        self._group_bounds = dict(group_bounds)

    @property
    def data(self) -> pd.DataFrame:
        """返回分组收益 DataFrame。"""
        return self._quantile_returns

    def long_short(self) -> pd.DataFrame:
        """返回各因子最高组减最低组的收益。

        输入：无。
        输出：index 为时间、columns 为因子名称的多空收益 DataFrame。
        """
        results: list[pd.Series] = []
        for factor in self._quantile_returns.columns.get_level_values(0).unique():
            frame = self._quantile_returns[factor]
            bounds = self._group_bounds[factor]
            # ************************************************************
            # 分组收益形状为 (时间, 固定目标组数)，bounds 形状为 (时间, 2)；
            # 按因子值实际形成的边界组取值，不能用收益非空性推断组是否存在。
            # ************************************************************
            series = pd.Series(
                (
                    _long_short_return(
                        frame.loc[index],
                        bounds.loc[index, "lowest"],
                        bounds.loc[index, "highest"],
                    )
                    for index in frame.index
                ),
                index=frame.index,
                dtype=float,
            )
            series.name = factor
            results.append(series)
        return pd.concat(results, axis=1)


class QuantileReturnAnalyzer(AbstractAnalyzer):
    """计算因子的横截面分组收益。"""

    def __init__(self, returns: FactorInput, n_groups: int = 5) -> None:
        """创建分组收益检验器。

        输入：收益率因子或 DataFrame，以及至少为 2 的分组数量。
        输出：分组收益检验器实例。
        """
        if n_groups < 2:
            raise ValueError("n_groups 必须至少为 2")
        super().__init__(returns=returns)
        self.n_groups = n_groups

    def _analyze(
        self,
        factors: Mapping[str, pd.DataFrame],
        **inputs: pd.DataFrame,
    ) -> QuantileReturnResult:
        """计算每个因子的分组平均收益。

        输入：待检验因子映射，以及名为 returns 的收益率 DataFrame。
        输出：各因子的逐期分组收益。
        """
        returns = inputs["returns"]
        outputs: list[pd.DataFrame] = []
        group_bounds: dict[str, pd.DataFrame] = {}
        for name, factor in factors.items():
            groups = factor.apply(
                _assign_quantiles,
                axis=1,
                n_groups=self.n_groups,
            )
            # groups 的形状为 (时间, 资产)，沿资产轴聚合后变为
            # (时间, 2)，记录因子值实际形成的最低和最高组号。
            group_bounds[name] = pd.DataFrame(
                {
                    "lowest": groups.min(axis=1),
                    "highest": groups.max(axis=1),
                }
            )
            values = {
                group: returns.where(groups == group).mean(axis=1)
                for group in range(1, self.n_groups + 1)
            }
            output = pd.DataFrame(values)
            output.columns = pd.MultiIndex.from_product(
                [[name], output.columns],
                names=["factor", "quantile"],
            )
            outputs.append(output)
        return QuantileReturnResult(pd.concat(outputs, axis=1), group_bounds)


__all__ = ["QuantileReturnAnalyzer", "QuantileReturnResult"]
