from typing import Callable, List

import numpy as np
import pandas as pd
from statsmodels.api import OLS

from xqfactor.factor import (
    AbstractFactor,
    BinaryCombinedFactor,
    CombinedFactor,
    ConstantFactor,
    RefFactor,
    UnaryCombinedFactor,
)


def ABS(factor: AbstractFactor):
    return UnaryCombinedFactor(np.abs, factor)


def LOG(factor: AbstractFactor):
    return UnaryCombinedFactor(np.log, factor)


def EXP(factor: AbstractFactor):
    return UnaryCombinedFactor(np.exp, factor)


def EQUAL(factor1: AbstractFactor, factor2: AbstractFactor):
    return BinaryCombinedFactor(np.equal, factor1, factor2)


def SIGN(factor: AbstractFactor):
    return UnaryCombinedFactor(np.sign, factor)


def SIGNEDPOWER(factor: AbstractFactor, c: float):
    return BinaryCombinedFactor(
        lambda x, y: np.sign(x) * np.power(np.abs(x), y), factor, c
    )


def MIN(factor1: AbstractFactor, factor2: AbstractFactor):
    return BinaryCombinedFactor(np.minimum, factor1, factor2)


def FMIN(factor1: AbstractFactor, factor2: AbstractFactor):
    return BinaryCombinedFactor(np.fmin, factor1, factor2)


def MAX(factor1: AbstractFactor, factor2: AbstractFactor):
    return BinaryCombinedFactor(np.maximum, factor1, factor2)


def FMAX(factor1: AbstractFactor, factor2: AbstractFactor):
    return BinaryCombinedFactor(np.fmax, factor1, factor2)


def IF(
    condition: AbstractFactor, true_value: AbstractFactor, false_value: AbstractFactor
):
    if isinstance(true_value, (int, float)):
        true_value = ConstantFactor(true_value)
    if isinstance(false_value, (int, float)):
        false_value = ConstantFactor(false_value)

    def where_func(
        condition: pd.DataFrame, true_value: pd.DataFrame, false_value: pd.DataFrame
    ):
        x = np.where(condition, true_value, false_value)
        return pd.DataFrame(x, index=condition.index, columns=condition.columns)

    return CombinedFactor(where_func, condition, true_value, false_value)


def AS_FLOAT(factor: AbstractFactor):
    return UnaryCombinedFactor(np.float64, factor)


def REF(factor: AbstractFactor, n: int):
    if n == 0:
        return factor
    return RefFactor(factor, n)


def DELAY(factor: AbstractFactor, n: int):
    return REF(factor, n)


def DELTA(factor: AbstractFactor, n: int):
    return BinaryCombinedFactor(np.subtract, factor, REF(factor, n))


def PCT_CHANGE(factor: AbstractFactor, n: int):
    return BinaryCombinedFactor(np.divide, factor, REF(factor, n)) - 1.0


def NOTNA(factor: AbstractFactor):
    return UnaryCombinedFactor(pd.notna, factor)


def MAD(factor: AbstractFactor, n: int):
    def mad(factor: pd.DataFrame, n) -> pd.DataFrame:
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

    return UnaryCombinedFactor(mad, factor, n)


def QUANTILE(factor: AbstractFactor, n_groups: int):
    return UnaryCombinedFactor(pd.qcut, factor, n_groups, labels=range(1, n_groups + 1))


def GROUP_QUANTILE(factor: AbstractFactor, grouper: AbstractFactor, n_groups: int = 5):
    def group_quantile(factor: pd.DataFrame, grouper: pd.DataFrame, n_groups: int = 5):
        df = pd.concat([factor.stack(), grouper.stack()], axis=1).dropna()
        df.columns = ["factor", "grouper"]
        return (
            df.groupby(["datetime", "grouper"], group_keys=False)["factor"]
            .apply(pd.qcut, n_groups, labels=range(1, n_groups + 1))
            .unstack(level="code")
        )

    return BinaryCombinedFactor(group_quantile, factor, grouper, n_groups)


def NORM(factor: AbstractFactor):
    def norm(factor: pd.DataFrame) -> pd.DataFrame:
        """数据标准化"""
        return factor.sub(factor.mean(axis=1), axis=0).div(
            factor.std(axis=1, ddof=1), axis=0
        )

    return UnaryCombinedFactor(norm, factor)


def RANK(factor: AbstractFactor, ascending=True):
    return UnaryCombinedFactor(pd.DataFrame.rank, factor, axis=1, ascending=ascending)


def PROPORTION(factor: AbstractFactor):
    def proportion(factor: pd.DataFrame) -> pd.DataFrame:
        return factor.div(factor.sum(axis=1), axis=0)

    return UnaryCombinedFactor(proportion, factor)


def DIFF(factor: AbstractFactor, n: int = 1):
    if n == 0:
        return factor
    # return UnaryCombinedFactor(pd.DataFrame.diff, factor, n)
    return BinaryCombinedFactor(np.subtract, factor, REF(factor, n))


def CUMPROD(factor: AbstractFactor):
    return UnaryCombinedFactor(pd.DataFrame.cumprod, factor)


def FFILL(factor: AbstractFactor):
    return UnaryCombinedFactor(pd.DataFrame.ffill, factor, inplace=False)


def FILLNA(factor: AbstractFactor, value: AbstractFactor):
    if isinstance(value, (int, float)):
        value = ConstantFactor(value)
    return BinaryCombinedFactor(pd.DataFrame.fillna, factor, value)


def MASK(factor: AbstractFactor, masked_by: AbstractFactor):
    def mask(factor: pd.DataFrame, masked_by: pd.DataFrame) -> pd.DataFrame:
        masked_by = masked_by.astype(bool)
        return factor.mask(masked_by)

    return BinaryCombinedFactor(mask, factor, masked_by)


def CSGROUP(factor: AbstractFactor, grouper: AbstractFactor, func: Callable, args=()):
    def group(
        factor: pd.DataFrame, grouper: pd.DataFrame, func: Callable, args=()
    ) -> pd.DataFrame:
        return (
            factor.stack(dropna=False)
            .groupby(["datetime", grouper.stack()])
            .transform(func, *args)
            .unstack()
        )

    return BinaryCombinedFactor(group, factor, grouper, func, args)


def MINMAXSCALER(factor: AbstractFactor):
    def minmaxscaler(factor: pd.DataFrame) -> pd.DataFrame:
        """
        MinMaxScaler, 最小最大归一化
        """
        return factor.sub(factor.min(axis=1), axis=0).div(
            factor.max(axis=1) - factor.min(axis=1), axis=0
        )

    return UnaryCombinedFactor(minmaxscaler, factor)


def CSNEUTRALIZER(
    factor: AbstractFactor,
    neutralize_by: AbstractFactor | List[AbstractFactor],
    dummies: List[bool] = False,
    model="OLS",
):
    def neutralize(
        factor: pd.DataFrame,
        *neutralize_by: pd.DataFrame,
        dummies: bool | List[bool] = False,
        model="OLS",
        # weights: pd.DataFrame = None,
        **regression_kwargs,
    ) -> pd.DataFrame:
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

    if isinstance(neutralize_by, AbstractFactor):
        neutralize_by = [neutralize_by]
    return CombinedFactor(
        neutralize, factor, *neutralize_by, dummies=dummies, model=model
    )


def BINARY_LABEL(factor: AbstractFactor, top_pct: float = 0.3, bottom_pct: float = 0.3):
    """
    根据因子值在横截面上的排名，将排名靠前的top_pct比例标记为正例（1），
    排名靠后的bottom_pct比例标记为负例（0），其余标记为NaN

    Args:
        factor: 用于排序的因子，通常是收益率因子
        top_pct: 正例比例，默认为0.3（30%）
        bottom_pct: 负例比例，默认为0.3（30%）

    Returns:
        标记为0和1的二元分类标签因子
    """

    def binary_label(
        factor: pd.DataFrame, top_pct: float = 0.3, bottom_pct: float = 0.3
    ) -> pd.DataFrame:
        # 创建结果DataFrame，初始化为NaN
        result = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)

        # 计算每行的分位数阈值 (pandas的quantile默认skipna=True)
        top_thresholds = factor.quantile(1 - top_pct, axis=1)
        bottom_thresholds = factor.quantile(bottom_pct, axis=1)

        # 创建布尔矩阵标识正例和负例
        # 广播操作：将阈值扩展为与factor相同的形状进行比较
        top_mask = factor.ge(top_thresholds, axis=0)
        bottom_mask = factor.le(bottom_thresholds, axis=0)

        # 使用布尔索引一次性赋值
        result[top_mask] = 1.0
        result[bottom_mask] = 0.0

        return result

    return UnaryCombinedFactor(binary_label, factor, top_pct, bottom_pct)


__all__ = [
    "ABS",
    "EXP",
    "LOG",
    "EQUAL",
    "SIGN",
    "SIGNEDPOWER",
    "MIN",
    "FMIN",
    "MAX",
    "FMAX",
    "IF",
    "AS_FLOAT",
    "REF",
    "DELAY",
    "DELTA",
    "PCT_CHANGE",
    "NOTNA",
    "MAD",
    "QUANTILE",
    "GROUP_QUANTILE",
    "NORM",
    "RANK",
    "PROPORTION",
    "DIFF",
    "CUMPROD",
    "FFILL",
    "FILLNA",
    "MASK",
    "CSGROUP",
    "MINMAXSCALER",
    "BINARY_LABEL",
    "CSNEUTRALIZER",
]
