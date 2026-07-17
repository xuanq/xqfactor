"""基于 NumPy/Pandas 实现的内置因子算子。"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from xqfactor.factor import (
    AbstractFactor,
    BinaryCombinedFactor,
    CombinedFactor,
    ConstantFactor,
    RefFactor,
    UnaryCombinedFactor,
)


def _as_factor(value: Any) -> AbstractFactor:
    """将常量或因子统一转换为因子节点。"""
    return value if isinstance(value, AbstractFactor) else ConstantFactor(value)


def _where(
    condition: pd.DataFrame,
    true_value: pd.DataFrame,
    false_value: pd.DataFrame,
) -> pd.DataFrame:
    """根据条件在两个二维因子值之间选择。"""
    values = np.where(condition.astype(bool), true_value, false_value)
    return pd.DataFrame(
        values,
        index=condition.index,
        columns=condition.columns,
    )


def _signed_power(factor: pd.DataFrame, exponent: float) -> pd.DataFrame:
    """计算保持原符号的幂。"""
    return np.sign(factor) * np.power(np.abs(factor), exponent)


def _as_float(factor: pd.DataFrame) -> pd.DataFrame:
    """将因子值转换为浮点类型。"""
    return factor.astype(float)


def _notna(factor: pd.DataFrame) -> pd.DataFrame:
    """返回因子值是否非缺失。"""
    return factor.notna()


def _mad(factor: pd.DataFrame, n: float) -> pd.DataFrame:
    """使用绝对值差中位数法进行横截面去极值。"""
    median = factor.median(axis=1)
    deviation = factor.sub(median, axis=0).abs().median(axis=1)
    return factor.clip(
        lower=median - n * deviation,
        upper=median + n * deviation,
        axis=0,
    )


def _quantile(factor: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    """按每个时间截面的排序比例划分分位数组。"""
    ranks = factor.rank(axis=1, method="first", pct=True)
    groups = np.ceil(ranks * n_groups)
    return groups.clip(lower=1, upper=n_groups)


def _group_quantile(
    factor: pd.DataFrame,
    grouper: pd.DataFrame,
    n_groups: int,
) -> pd.DataFrame:
    """在每个时间和分组内部划分因子分位数组。"""
    result = pd.DataFrame(
        np.nan,
        index=factor.index,
        columns=factor.columns,
    )
    for timestamp in factor.index:
        values = factor.loc[timestamp]
        labels = grouper.loc[timestamp]
        for _, assets in labels.groupby(labels).groups.items():
            selected = values.loc[assets].dropna()
            if selected.empty:
                continue
            ranks = selected.rank(method="first", pct=True)
            result.loc[timestamp, selected.index] = np.ceil(ranks * n_groups).clip(
                lower=1, upper=n_groups
            )
    return result


def _norm(factor: pd.DataFrame) -> pd.DataFrame:
    """执行横截面 z-score 标准化。"""
    return factor.sub(factor.mean(axis=1), axis=0).div(
        factor.std(axis=1, ddof=1),
        axis=0,
    )


def _rank(
    factor: pd.DataFrame,
    method: str,
    ascending: bool,
    pct: bool,
) -> pd.DataFrame:
    """执行横截面排名。"""
    return factor.rank(
        axis=1,
        method=method,
        ascending=ascending,
        pct=pct,
    )


def _proportion(factor: pd.DataFrame) -> pd.DataFrame:
    """将每个横截面的值除以横截面总和。"""
    return factor.div(factor.sum(axis=1), axis=0)


def _cumprod(factor: pd.DataFrame) -> pd.DataFrame:
    """沿时间轴计算累计乘积。"""
    return factor.cumprod()


def _ffill(factor: pd.DataFrame) -> pd.DataFrame:
    """沿时间轴向前填充缺失值。"""
    return factor.ffill()


def _fillna(
    factor: pd.DataFrame,
    fill_value: pd.DataFrame,
) -> pd.DataFrame:
    """使用另一个因子值填充缺失位置。"""
    return factor.where(factor.notna(), fill_value)


def _mask(
    factor: pd.DataFrame,
    masked_by: pd.DataFrame,
) -> pd.DataFrame:
    """将掩码为真的位置设置为缺失值。"""
    return factor.mask(masked_by.astype(bool))


def _cs_group(
    factor: pd.DataFrame,
    grouper: pd.DataFrame,
    *,
    func: Callable[..., Any],
    args: Sequence[Any],
) -> pd.DataFrame:
    """按时间和横截面分组执行转换函数。"""
    factor_series = factor.stack(dropna=False)
    grouper_series = grouper.stack(dropna=False)
    result = factor_series.groupby(
        [
            factor_series.index.get_level_values(0),
            grouper_series,
        ]
    ).transform(func, *args)
    return result.unstack()


def _minmax_scaler(factor: pd.DataFrame) -> pd.DataFrame:
    """执行横截面最小最大归一化。"""
    minimum = factor.min(axis=1)
    maximum = factor.max(axis=1)
    return factor.sub(minimum, axis=0).div(maximum - minimum, axis=0)


def _cs_neutralize(
    factor: pd.DataFrame,
    *controls: pd.DataFrame,
    dummies: bool | tuple[bool, ...] = False,
    model: str = "OLS",
) -> pd.DataFrame:
    """使用逐期横截面回归残差进行中性化。"""
    if model.upper() != "OLS":
        raise ValueError(f"仅支持 OLS，中性化收到 {model}")
    dummy_flags = (dummies,) * len(controls) if isinstance(dummies, bool) else dummies
    if len(dummy_flags) != len(controls):
        raise ValueError("dummies 与 neutralize_by 数量不一致")

    result = pd.DataFrame(
        np.nan,
        index=factor.index,
        columns=factor.columns,
    )
    for timestamp in factor.index:
        design_parts: list[pd.DataFrame] = []
        for index, (control, use_dummies) in enumerate(
            zip(controls, dummy_flags, strict=True)
        ):
            series = control.loc[timestamp]
            if use_dummies:
                design_parts.append(pd.get_dummies(series, dtype=float))
            else:
                design_parts.append(series.rename(f"control_{index}").to_frame())
        design = pd.concat(design_parts, axis=1)
        valid = pd.concat(
            [factor.loc[timestamp].rename("factor"), design],
            axis=1,
        ).dropna()
        if len(valid) <= len(design.columns):
            continue

        # ************************************************************
        # 横截面从 DataFrame (资产数, 1 + 控制变量数) 转换为 NumPy
        # y: (资产数,)，X: (资产数, 控制变量数)，回归后恢复到原资产 columns。
        # ************************************************************
        x = valid.drop(columns="factor").to_numpy(dtype=float)
        y = valid["factor"].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
        residual = valid["factor"] - x @ coefficients
        result.loc[timestamp, residual.index] = residual
    return result


def _binary_label(
    factor: pd.DataFrame,
    top_pct: float,
    bottom_pct: float,
) -> pd.DataFrame:
    """将横截面顶部和底部区间编码为二元标签。"""
    top = factor.ge(factor.quantile(1 - top_pct, axis=1), axis=0)
    bottom = factor.le(factor.quantile(bottom_pct, axis=1), axis=0)
    result = pd.DataFrame(
        np.nan,
        index=factor.index,
        columns=factor.columns,
    )
    return result.mask(top, 1.0).mask(bottom, 0.0)


def ABS(factor: AbstractFactor) -> UnaryCombinedFactor:
    """返回因子绝对值。"""
    return UnaryCombinedFactor(np.abs, factor)


def LOG(factor: AbstractFactor) -> UnaryCombinedFactor:
    """返回因子自然对数。"""
    return UnaryCombinedFactor(np.log, factor)


def EXP(factor: AbstractFactor) -> UnaryCombinedFactor:
    """返回因子指数值。"""
    return UnaryCombinedFactor(np.exp, factor)


def EQUAL(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """比较两个因子是否相等。"""
    return BinaryCombinedFactor(
        np.equal,
        _as_factor(factor1),
        _as_factor(factor2),
    )


def SIGN(factor: AbstractFactor) -> UnaryCombinedFactor:
    """返回因子值符号。"""
    return UnaryCombinedFactor(np.sign, factor)


def SIGNEDPOWER(
    factor: AbstractFactor,
    exponent: float,
) -> UnaryCombinedFactor:
    """计算保持原符号的幂。"""
    return UnaryCombinedFactor(_signed_power, factor, exponent)


def MIN(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """逐元素取最小值。"""
    return BinaryCombinedFactor(
        np.minimum,
        _as_factor(factor1),
        _as_factor(factor2),
    )


def FMIN(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """逐元素取最小值并忽略单侧缺失值。"""
    return BinaryCombinedFactor(
        np.fmin,
        _as_factor(factor1),
        _as_factor(factor2),
    )


def MAX(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """逐元素取最大值。"""
    return BinaryCombinedFactor(
        np.maximum,
        _as_factor(factor1),
        _as_factor(factor2),
    )


def FMAX(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """逐元素取最大值并忽略单侧缺失值。"""
    return BinaryCombinedFactor(
        np.fmax,
        _as_factor(factor1),
        _as_factor(factor2),
    )


def IF(
    condition: AbstractFactor,
    true_value: Any,
    false_value: Any,
) -> CombinedFactor:
    """根据条件在两个因子之间选择值。"""
    return CombinedFactor(
        _where,
        condition,
        _as_factor(true_value),
        _as_factor(false_value),
    )


def AS_FLOAT(factor: AbstractFactor) -> UnaryCombinedFactor:
    """将因子转换为浮点类型。"""
    return UnaryCombinedFactor(_as_float, factor)


def REF(factor: AbstractFactor, n: int) -> AbstractFactor:
    """引用 n 个周期前或后的因子值。"""
    if n == 0:
        return factor
    return RefFactor(factor, n)


def DELAY(factor: AbstractFactor, n: int) -> AbstractFactor:
    """REF 的别名。"""
    return REF(factor, n)


def DELTA(factor: AbstractFactor, n: int) -> AbstractFactor:
    """计算当前值与滞后值的差。"""
    return factor - REF(factor, n)


def PCT_CHANGE(factor: AbstractFactor, n: int) -> AbstractFactor:
    """计算相对 n 个周期前的变化率。"""
    return factor / REF(factor, n) - 1.0


def NOTNA(factor: AbstractFactor) -> UnaryCombinedFactor:
    """返回因子值是否非缺失。"""
    return UnaryCombinedFactor(_notna, factor)


def MAD(factor: AbstractFactor, n: float) -> UnaryCombinedFactor:
    """使用绝对值差中位数法去极值。"""
    return UnaryCombinedFactor(_mad, factor, n)


def QUANTILE(
    factor: AbstractFactor,
    n_groups: int,
) -> UnaryCombinedFactor:
    """按横截面分位数返回组号。"""
    return UnaryCombinedFactor(_quantile, factor, n_groups)


def GROUP_QUANTILE(
    factor: AbstractFactor,
    grouper: AbstractFactor,
    n_groups: int = 5,
) -> BinaryCombinedFactor:
    """在分组变量内部进行横截面分位数分组。"""
    return BinaryCombinedFactor(
        _group_quantile,
        factor,
        grouper,
        n_groups,
    )


def NORM(factor: AbstractFactor) -> UnaryCombinedFactor:
    """执行横截面 z-score 标准化。"""
    return UnaryCombinedFactor(_norm, factor)


def RANK(
    factor: AbstractFactor,
    ascending: bool = True,
    method: str = "average",
    pct: bool = True,
) -> UnaryCombinedFactor:
    """执行横截面排名。"""
    return UnaryCombinedFactor(
        _rank,
        factor,
        method,
        ascending,
        pct,
    )


def PROPORTION(factor: AbstractFactor) -> UnaryCombinedFactor:
    """将横截面值除以横截面总和。"""
    return UnaryCombinedFactor(_proportion, factor)


def DIFF(factor: AbstractFactor, n: int = 1) -> AbstractFactor:
    """计算时间序列差分。"""
    return DELTA(factor, n)


def CUMPROD(factor: AbstractFactor) -> UnaryCombinedFactor:
    """计算时间序列累计乘积。"""
    return UnaryCombinedFactor(_cumprod, factor)


def FFILL(factor: AbstractFactor) -> UnaryCombinedFactor:
    """沿时间轴向前填充缺失值。"""
    return UnaryCombinedFactor(_ffill, factor)


def FILLNA(factor: AbstractFactor, value: Any) -> BinaryCombinedFactor:
    """使用另一个因子或常量填充缺失值。"""
    return BinaryCombinedFactor(
        _fillna,
        factor,
        _as_factor(value),
    )


def MASK(
    factor: AbstractFactor,
    masked_by: AbstractFactor,
) -> BinaryCombinedFactor:
    """将掩码为真的位置设置为缺失值。"""
    return BinaryCombinedFactor(_mask, factor, masked_by)


def CSGROUP(
    factor: AbstractFactor,
    grouper: AbstractFactor,
    func: Callable[..., Any],
    args: Sequence[Any] = (),
) -> CombinedFactor:
    """按横截面分组执行用户函数。"""
    return CombinedFactor(
        _cs_group,
        factor,
        grouper,
        func=func,
        args=tuple(args),
    )


def MINMAXSCALER(factor: AbstractFactor) -> UnaryCombinedFactor:
    """执行横截面最小最大归一化。"""
    return UnaryCombinedFactor(_minmax_scaler, factor)


def CSNEUTRALIZER(
    factor: AbstractFactor,
    neutralize_by: AbstractFactor | list[AbstractFactor],
    dummies: bool | list[bool] = False,
    model: str = "OLS",
) -> CombinedFactor:
    """按一个或多个控制因子执行横截面中性化。"""
    controls = neutralize_by if isinstance(neutralize_by, list) else [neutralize_by]
    dummy_flags = tuple(dummies) if isinstance(dummies, list) else dummies
    return CombinedFactor(
        _cs_neutralize,
        factor,
        *controls,
        dummies=dummy_flags,
        model=model,
    )


def BINARY_LABEL(
    factor: AbstractFactor,
    top_pct: float = 0.3,
    bottom_pct: float = 0.3,
) -> UnaryCombinedFactor:
    """将横截面顶部和底部区间编码为二元标签。"""
    return UnaryCombinedFactor(
        _binary_label,
        factor,
        top_pct,
        bottom_pct,
    )
