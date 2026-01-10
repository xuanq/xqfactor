from typing import Dict, List

import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.api import OLS, WLS

from xqfactor.analyzer.base import AbstractAnalyzer
from xqfactor.factor import AbstractFactor


class RegressionAnalyzer(AbstractAnalyzer):
    def __init__(
        self,
        returns: pd.DataFrame | AbstractFactor,
        model="WLS",
        weights: pd.DataFrame | AbstractFactor = None,
        industry: pd.DataFrame | AbstractFactor = None,
        config=None,
        keep_processed_results: bool = False,
    ) -> None:
        super().__init__(config, keep_processed_results)
        self.returns = returns
        self.weights = weights
        self.industry = industry
        self.model = model

    def _analyze(self, factors: Dict[str, pd.DataFrame | AbstractFactor]):
        returns = self._get_value(self.returns)
        weights = self._get_value(self.weights)
        industry = self._get_value(self.industry)
        reg_res = []
        for name, factor in factors.items():
            factor = self._get_value(factor)
            df = factor.stack().to_frame("factor")

            if industry is not None:
                industry_dummies = pd.get_dummies(industry.stack(), dtype="float")
                df[industry_dummies.columns] = industry_dummies
            x_cols = df.columns.to_list()

            if weights is not None:
                df["weights"] = weights.stack()
            else:
                df["weights"] = 1

            df["returns"] = returns.stack()

            df.dropna(axis=0, inplace=True)

            def cross_sectional_regression(csfactordf, x_cols, model):
                X = csfactordf[x_cols]
                y = csfactordf["returns"]
                if model == "WLS":
                    reg_model = WLS(y, X, weights=csfactordf["weights"])
                elif model == "OLS":
                    reg_model = OLS(y, X)
                return reg_model.fit()

            reg_result = df.groupby("datetime").apply(
                cross_sectional_regression, x_cols, self.model
            )
            reg_result.name = name
            reg_res.append(reg_result)
        reg_results = pd.concat(reg_res, axis=1)
        return RegressionAnalyzerResult(reg_results)


class RegressionAnalyzerResult:
    def __init__(self, reg_results: pd.DataFrame) -> None:
        self._reg_results = reg_results

    def reg_results(self, factor: str | List[str] = None) -> pd.DataFrame:
        if factor is None:
            return self._reg_results
        elif isinstance(factor, str):
            factor = [factor]
        return self._reg_results[factor]

    def coefs(self, factor: str | List[str] = None) -> pd.DataFrame:
        return self.reg_results(factor).map(lambda x: x.params.loc["factor"])

    def se(self, factor: str | List[str] = None) -> pd.DataFrame:
        return self.reg_results(factor).map(lambda x: x.bse.loc["factor"])

    def t(self, factor: str | List[str] = None) -> pd.DataFrame:
        return self.reg_results(factor).map(lambda x: x.tvalues.loc["factor"])

    def num_obs(self, factor: str | List[str] = None) -> pd.DataFrame:
        return self.reg_results(factor).map(lambda x: x.nobs)

    # 因子收益率序列的𝑡值检验
    def ret_ttest(
        self, factor: str | List[str] = None, popmean=0, confidence_level=0.95
    ):
        ret_df = self.coefs(factor)
        res = ttest_1samp(ret_df, popmean)
        stats = pd.Series(index=ret_df.columns, data=res.statistic, name="t_stat")
        pvalue = pd.Series(index=ret_df.columns, data=res.pvalue, name="p_value")
        conf_high = pd.Series(
            index=ret_df.columns,
            data=res.confidence_interval(confidence_level).high,
            name="conf_high",
        )
        conf_low = pd.Series(
            index=ret_df.columns,
            data=res.confidence_interval(confidence_level).low,
            name="conf_low",
        )
        return pd.concat([stats, pvalue, conf_low, conf_high], axis=1)

    @property
    def statistics(self) -> pd.DataFrame:
        return pd.concat([self.coefs(), self.se(), self.t(), self.num_obs()], axis=1)

    def summary(
        self,
        factor: str | List[str] = None,
        t_threshold=2,
        confidence_level=0.95,
        popmean=0,
    ) -> pd.DataFrame:
        t_abs_mean = self.t(factor).abs().mean()
        significant_t_pct = (self.t(factor).abs() > t_threshold).mean()
        t_mean = self.t(factor).mean()
        t_of_t = self.t(factor).mean() / self.t(factor).std()
        ret_mean = self.coefs(factor).mean()
        ret_t = self.ret_ttest(factor, popmean, confidence_level).t_stat
        return pd.DataFrame(
            {
                "t_abs_mean": t_abs_mean,
                "significant_t_pct": significant_t_pct,
                "t_mean": t_mean,
                "t_of_t": t_of_t,
                "ret_mean": ret_mean,
                "ret_t": ret_t,
            }
        )

    def cum_ret(self, factor: str | List[str] = None):
        return (1 + self.coefs(factor)).cumprod() - 1
