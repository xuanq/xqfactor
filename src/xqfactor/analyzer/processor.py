from typing import List

import pandas as pd
from statsmodels.api import OLS

from xqfactor.analyzer.base import BaseProcessor
from xqfactor.factor import AbstractFactor


class Winsorizer(BaseProcessor):
    def mad(self, factor: pd.DataFrame, n, *args, **kwargs) -> pd.DataFrame:
        """
        Median Absolute Deviation, 绝对值差中位数法去极值
        """
        median = factor.median(axis="columns")
        diff_median = factor.sub(median, axis="index").abs().median(axis=1)
        max_range = median + n * diff_median
        min_range = median - n * diff_median
        if isinstance(factor, pd.Series):
            return factor.clip(min_range, max_range)
        else:
            return factor.clip(min_range, max_range, axis=0)


class Normalizer(BaseProcessor):
    def norm(self, factor: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        """数据标准化"""
        return factor.sub(factor.mean(axis=1), axis=0).div(
            factor.std(axis=1, ddof=1), axis=0
        )


class Ranker(BaseProcessor):
    def rank(self, factor: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return factor.rank(axis=1)


class Filler(BaseProcessor):
    def fillna(
        self, factor: pd.DataFrame, fill_value: float = 0, *args, **kwargs
    ) -> pd.DataFrame:
        return factor.fillna(fill_value)


class CSNeutralizer(BaseProcessor):
    def neutralize(
        self,
        factor: pd.DataFrame,
        neutralize_by: List[pd.DataFrame] | List[AbstractFactor],
        dummies: List[bool] = False,
        model="OLS",
        # weights: pd.DataFrame = None,
        **regression_kwargs,
    ) -> pd.DataFrame:
        if isinstance(neutralize_by, pd.DataFrame) or issubclass(
            neutralize_by.__class__, AbstractFactor
        ):
            neutralize_by = [neutralize_by]
        neutralize_by = [self._get_value(factor) for factor in neutralize_by]

        if isinstance(dummies, bool):
            dummies = [dummies]
        if len(neutralize_by) != len(dummies):
            raise ValueError(
                f"neutralize_by and dummies must have the same length, got {len(neutralize_by)} and {len(dummies)}"
            )

        df = factor.stack().to_frame("factor")
        for i in range(len(neutralize_by)):
            if not dummies[i]:
                df[f"neutralize_by_{i}"] = neutralize_by[i].stack()
            else:
                factor_dummies = pd.get_dummies(neutralize_by[i].stack(), dtype="float")
                factor_dummies.columns = [
                    f"neutralize_by_{i}_{j}" for j in factor_dummies.columns
                ]
                df[factor_dummies.columns] = factor_dummies
        df.dropna(axis=0, inplace=True)

        x_cols = df.columns.to_list()
        x_cols.remove("factor")
        if len(x_cols) > 0:

            def get_residuals(factordf, y_col, x_cols):
                X = factordf[x_cols]
                y = factordf[y_col]
                # if model == "WLS":
                #     reg_model = WLS(y, X, weights=factordf["weights"])
                # elif model == "OLS":
                #     reg_model = OLS(y, X)
                reg_model = OLS(y, X)
                reg_result = reg_model.fit()
                resid = reg_result.resid
                return resid

            factor = df.groupby("datetime", group_keys=False).apply(
                get_residuals, "factor", x_cols
            )
        else:
            factor = df["factor"]
        return factor.unstack(level="code")


class Masker(BaseProcessor):
    def mask(
        self,
        factor: pd.DataFrame,
        masked_by: pd.DataFrame | AbstractFactor,
        *args,
        **kwargs,
    ) -> pd.DataFrame:
        masked_by = self._get_value(masked_by)
        masked_by = masked_by.astype(bool)
        return factor.mask(masked_by)
