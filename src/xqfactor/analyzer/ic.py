from typing import Dict, List

import pandas as pd

from xqfactor.analyzer.base import AbstractAnalyzer
from xqfactor.config import Config
from xqfactor.factor import AbstractFactor


class ICAnalyzer(AbstractAnalyzer):
    def __init__(
        self,
        returns: pd.DataFrame | AbstractFactor,
        config: Config = None,
        keep_processed_results: bool = False,
    ) -> None:
        super().__init__(config, keep_processed_results)
        self.returns = returns

    def _analyze(self, factors: Dict[str, pd.DataFrame | AbstractFactor]):
        returns = self._get_value(self.returns)
        ics = []
        for name, factor in factors.items():
            factor = self._get_value(factor)
            ic = factor.corrwith(returns, axis=1)
            ic.name = name
            ics.append(ic)
        ic = pd.concat(ics, axis=1)
        return ICAnalysisResult(ic)


class ICAnalysisResult:
    def __init__(self, ic: pd.DataFrame) -> None:
        self._ic = ic

    def ic(self, factor: str | List[str] = None) -> pd.DataFrame:
        if factor is None:
            return self._ic
        elif isinstance(factor, str):
            factor = [factor]
        return self._ic[factor]

    def ic_mean(self, factor: str | List[str] = None) -> pd.Series:
        return self.ic(factor).mean(axis=0)

    def ic_std(self, factor: str | List[str] = None) -> pd.Series:
        return self.ic(factor).std(axis=0, ddof=1)

    def ir(self, factor: str | List[str] = None) -> pd.Series:
        return self.ic_mean(factor).abs() / self.ic_std(factor)

    def gt_ratio(self, factor: str | List[str] = None, value=0, abs=False) -> pd.Series:
        ic = self.ic(factor)
        if abs:
            ic = ic.abs()
        return (ic > value).mean(axis=0)

    def ic_cum(self, factor: str | List[str] = None, method="sum") -> pd.DataFrame:
        if method == "sum":
            return self.ic(factor).cumsum(axis=0)
        if method == "prod":
            return (self.ic(factor) + 1).cumprod(axis=0) - 1

    def summary(self, factor: str | List[str] = None) -> pd.DataFrame:
        # IC值序列的均值大小——因子显著性；
        ic_mean = self.ic_mean(factor)
        # IC 值序列的标准差——因子稳定性；
        ic_std = self.ic_std(factor)
        # IR 比率（IC 值序列均值与标准差的比值的绝对值）——因子有效性；
        ir = self.ir(factor)
        # IC 值序列大于零的占比——因子作用方向是否稳定
        gt_ratio = self.gt_ratio(factor, 0)
        # IC 值序列绝对值大于0.02的占比
        gt_ratio_abs = self.gt_ratio(factor, 0.02, abs=True)
        return pd.DataFrame(
            {
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "ir": ir,
                "gt_zero_ratio": gt_ratio,
                "abs_gt0.02_ratio": gt_ratio_abs,
            }
        )
