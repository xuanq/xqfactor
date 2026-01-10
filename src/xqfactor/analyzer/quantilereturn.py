from typing import Dict, List

import pandas as pd

from xqfactor.analyzer.base import AbstractAnalyzer
from xqfactor.factor import AbstractFactor


class QuantileReturnAnalyzer(AbstractAnalyzer):
    def __init__(
        self,
        returns: pd.DataFrame | AbstractFactor,
        n_groups: int,
        industry: pd.DataFrame | AbstractFactor = None,
        benchmark_returns: pd.DataFrame | AbstractFactor = None,
        config=None,
        keep_processed_results: bool = False,
    ):
        super().__init__(config, keep_processed_results=keep_processed_results)
        self.returns = returns
        self.n_groups = n_groups
        self.benchmark_returns = benchmark_returns
        self.industry = industry

    def _analyze(self, factors: Dict[str, pd.DataFrame | AbstractFactor]):
        returns = self._get_value(self.returns)
        industry = self._get_value(self.industry)
        benchmark_returns = self._get_value(self.benchmark_returns)
        quantile_returns_list = []
        for name, factor in factors.items():
            factor = self._get_value(factor)
            df = factor.stack().to_frame("factor")

            grouper = ["datetime"]

            if industry is not None:
                df["industry"] = industry.stack()
                grouper.append("industry")

            df["factor_quantile"] = df.groupby(grouper, group_keys=False)[
                "factor"
            ].apply(pd.qcut, q=self.n_groups, labels=range(1, self.n_groups + 1))

            df["returns"] = returns.stack()

            quantile_returns = (
                df.groupby(["datetime", "factor_quantile"])
                .agg({"returns": "mean"})
                .unstack(level="factor_quantile")
            )
            quantile_returns.columns = quantile_returns.columns.set_levels(
                [name], level=0
            )
            quantile_returns.columns = quantile_returns.columns.set_names(
                ["factor", "factor_quantile"]
            )
            quantile_returns_list.append(quantile_returns)
        quantile_returns = pd.concat(quantile_returns_list, axis=1)

        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.mean(axis=1)
        else:
            benchmark_returns = None
        return QuantileReturnResult(quantile_returns, benchmark_returns)


class QuantileReturnResult:
    def __init__(self, quantile_returns: pd.DataFrame, benchmark_returns=None):
        self._quantile_returns = quantile_returns
        self.benchmark_returns = benchmark_returns

    @property
    def num_groups(self) -> int:
        return self._quantile_returns.columns.get_level_values("factor_quantile").max()

    def quantile_returns(self, factor: str | List[str] = None) -> pd.DataFrame:
        if factor is None:
            return self._quantile_returns
        elif isinstance(factor, str):
            factor = [factor]
        return self._quantile_returns[factor]

    def long_short(self, factor: str | List[str] = None) -> pd.DataFrame:
        quantile_returns = self.quantile_returns(factor)
        long_shorts = []
        for f in quantile_returns.columns.get_level_values("factor").unique():
            long_short = (
                quantile_returns[(f, self.num_groups)] - quantile_returns[(f, 1)]
            )
            long_short.name = (f, "long_short")
            long_shorts.append(long_short)
        return pd.concat(long_shorts, axis=1)

    def cumulative_returns(self, factor: str | List[str] = None) -> pd.DataFrame:
        return (self.quantile_returns(factor) + 1).cumprod() - 1

    def annualized_returns(self, factor: str | List[str] = None, annualize_factor=252):
        cum_returns = self.cumulative_returns(factor)
        row_num = pd.Series(
            index=cum_returns.index, data=range(1, len(cum_returns) + 1)
        )
        return (cum_returns + 1).pow(annualize_factor / row_num, axis=0) - 1
