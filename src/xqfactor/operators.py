"""后端无关的基础算子和自定义算子构造工具。"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from xqfactor.factor import (
    AbstractFactor,
    BinaryCombinedFactor,
    CombinedFactor,
    ConstantFactor,
    RefFactor,
    RollingWindowFactor,
    UnaryCombinedFactor,
)
from xqfactor.runtime import OperatorSpec


def _spec(
    name: str,
    kind: str,
    *args: Any,
    function: Callable[..., Any] | None = None,
    **kwargs: Any,
) -> OperatorSpec:
    """创建带有稳定参数的算子定义。"""
    return OperatorSpec(
        name, kind, tuple(args), tuple(sorted(kwargs.items())), function
    )


def _binary(name: str, factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """创建双输入基础算子。"""
    return BinaryCombinedFactor(
        _spec(name, "binary"), _as_factor(factor1), _as_factor(factor2)
    )


def _as_factor(value: Any) -> AbstractFactor:
    """将常量或因子统一转换为因子节点。"""
    return value if isinstance(value, AbstractFactor) else ConstantFactor(value)


def ABS(factor: AbstractFactor) -> UnaryCombinedFactor:
    """计算绝对值。"""
    return UnaryCombinedFactor(_spec("abs", "unary"), factor)


def LOG(factor: AbstractFactor) -> UnaryCombinedFactor:
    """计算自然对数。"""
    return UnaryCombinedFactor(_spec("log", "unary"), factor)


def EXP(factor: AbstractFactor) -> UnaryCombinedFactor:
    """计算指数。"""
    return UnaryCombinedFactor(_spec("exp", "unary"), factor)


def EQUAL(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """比较两个因子是否相等。"""
    return _binary("equal", factor1, factor2)


def SIGN(factor: AbstractFactor) -> UnaryCombinedFactor:
    """返回因子值的符号。"""
    return UnaryCombinedFactor(_spec("sign", "unary"), factor)


def SIGNEDPOWER(factor: AbstractFactor, c: float) -> UnaryCombinedFactor:
    """计算保持原符号的幂。"""
    return UnaryCombinedFactor(_spec("signed_power", "unary", c), factor)


def MIN(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """逐元素取最小值。"""
    return _binary("minimum", factor1, factor2)


def FMIN(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """逐元素取最小值并忽略单侧缺失值。"""
    return _binary("fminimum", factor1, factor2)


def MAX(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """逐元素取最大值。"""
    return _binary("maximum", factor1, factor2)


def FMAX(factor1: Any, factor2: Any) -> BinaryCombinedFactor:
    """逐元素取最大值并忽略单侧缺失值。"""
    return _binary("fmaximum", factor1, factor2)


def IF(condition: AbstractFactor, true_value: Any, false_value: Any) -> CombinedFactor:
    """根据条件在两个因子之间选择值。"""
    return CombinedFactor(
        _spec("where", "combined"),
        condition,
        _as_factor(true_value),
        _as_factor(false_value),
    )


def AS_FLOAT(factor: AbstractFactor) -> UnaryCombinedFactor:
    """将因子转换为浮点类型。"""
    return UnaryCombinedFactor(_spec("as_float", "unary"), factor)


def REF(factor: AbstractFactor, n: int) -> AbstractFactor:
    """引用 n 个周期前或后的因子值。"""
    if n == 0:
        return factor
    return RefFactor(factor, n)


def DELAY(factor: AbstractFactor, n: int) -> AbstractFactor:
    """REF 的别名。"""
    return REF(factor, n)


def DELTA(factor: AbstractFactor, n: int) -> BinaryCombinedFactor:
    """计算当前值与滞后值的差。"""
    return _binary("subtract", factor, REF(factor, n))


def PCT_CHANGE(factor: AbstractFactor, n: int) -> UnaryCombinedFactor:
    """计算相对 n 个周期前的变化率。"""
    return UnaryCombinedFactor(_spec("pct_change", "unary", n), factor)


def NOTNA(factor: AbstractFactor) -> UnaryCombinedFactor:
    """返回因子值是否非缺失。"""
    return UnaryCombinedFactor(_spec("notna", "unary"), factor)


def MAD(factor: AbstractFactor, n: float) -> UnaryCombinedFactor:
    """使用绝对值差中位数法进行横截面去极值。"""
    return UnaryCombinedFactor(_spec("mad", "cross_sectional", n), factor)


def QUANTILE(factor: AbstractFactor, n_groups: int) -> UnaryCombinedFactor:
    """按横截面分位数返回分组编号。"""
    return UnaryCombinedFactor(_spec("quantile", "cross_sectional", n_groups), factor)


def GROUP_QUANTILE(
    factor: AbstractFactor, grouper: AbstractFactor, n_groups: int = 5
) -> BinaryCombinedFactor:
    """在分组变量内部进行横截面分位数分组。"""
    return BinaryCombinedFactor(
        _spec("group_quantile", "cross_sectional", n_groups), factor, grouper
    )


def NORM(factor: AbstractFactor) -> UnaryCombinedFactor:
    """执行横截面 z-score 标准化。"""
    return UnaryCombinedFactor(_spec("norm", "cross_sectional"), factor)


def RANK(
    factor: AbstractFactor,
    ascending: bool = True,
    method: str = "average",
    pct: bool = True,
) -> UnaryCombinedFactor:
    """执行横截面排名，默认返回归一化排名。"""
    return UnaryCombinedFactor(
        _spec("rank", "cross_sectional", method, ascending, pct), factor
    )


def PROPORTION(factor: AbstractFactor) -> UnaryCombinedFactor:
    """将横截面值除以横截面总和。"""
    return UnaryCombinedFactor(_spec("proportion", "cross_sectional"), factor)


def DIFF(factor: AbstractFactor, n: int = 1) -> AbstractFactor:
    """计算时间序列差分。"""
    return DELTA(factor, n)


def CUMPROD(factor: AbstractFactor) -> UnaryCombinedFactor:
    """计算时间序列累计乘积。"""
    return UnaryCombinedFactor(_spec("cumprod", "time_series"), factor)


def FFILL(factor: AbstractFactor) -> UnaryCombinedFactor:
    """沿时间轴向前填充缺失值。"""
    return UnaryCombinedFactor(_spec("ffill", "time_series"), factor)


def FILLNA(factor: AbstractFactor, value: Any) -> BinaryCombinedFactor:
    """使用另一个因子或常量填充缺失值。"""
    return _binary("fillna", factor, value)


def MASK(factor: AbstractFactor, masked_by: AbstractFactor) -> BinaryCombinedFactor:
    """将掩码为真的位置设为缺失。"""
    return _binary("mask", factor, masked_by)


def CSGROUP(
    factor: AbstractFactor,
    grouper: AbstractFactor,
    func: Callable[..., Any],
    args: Sequence[Any] = (),
) -> BinaryCombinedFactor:
    """按横截面分组执行用户提供的函数。"""
    return BinaryCombinedFactor(
        _spec("cs_group", "cross_sectional", tuple(args), function=func),
        factor,
        grouper,
    )


def MINMAXSCALER(factor: AbstractFactor) -> UnaryCombinedFactor:
    """执行横截面最小最大归一化。"""
    return UnaryCombinedFactor(_spec("minmax_scaler", "cross_sectional"), factor)


def CSNEUTRALIZER(
    factor: AbstractFactor,
    neutralize_by: AbstractFactor | list[AbstractFactor],
    dummies: bool | list[bool] = False,
    model: str = "OLS",
) -> CombinedFactor:
    """按一个或多个控制因子执行横截面中性化。"""
    controls = neutralize_by if isinstance(neutralize_by, list) else [neutralize_by]
    return CombinedFactor(
        _spec(
            "cs_neutralizer",
            "cross_sectional",
            tuple(dummies) if isinstance(dummies, list) else dummies,
            model=model,
        ),
        factor,
        *controls,
    )


def BINARY_LABEL(
    factor: AbstractFactor, top_pct: float = 0.3, bottom_pct: float = 0.3
) -> UnaryCombinedFactor:
    """将横截面顶部和底部区间编码为二元标签。"""
    return UnaryCombinedFactor(
        _spec("binary_label", "cross_sectional", top_pct, bottom_pct), factor
    )


def custom_unary(
    factor: AbstractFactor,
    function: Callable[..., Any],
    *args: Any,
    name: str = "custom_unary",
    **kwargs: Any,
) -> UnaryCombinedFactor:
    """定义一个由后端接收原始数据并执行的单目算子。"""
    return UnaryCombinedFactor(
        _spec(name, "custom", *args, function=function, **kwargs), factor
    )


def define_operator(
    name: str,
    kind: str,
    function: Callable[..., Any] | None = None,
    *args: Any,
    version: str = "1",
    **kwargs: Any,
) -> OperatorSpec:
    """定义一个可交给 OperatorRegistry 注册的算子。

    输入：名称、算子类别、可选后端函数、位置参数、版本和关键字参数。
    输出：后端无关的 OperatorSpec。
    """
    return OperatorSpec(
        name=name,
        kind=kind,
        args=tuple(args),
        kwargs=tuple(sorted(kwargs.items())),
        function=function,
        version=version,
    )


def custom_binary(
    factor1: AbstractFactor,
    factor2: AbstractFactor,
    function: Callable[..., Any],
    *args: Any,
    name: str = "custom_binary",
    **kwargs: Any,
) -> BinaryCombinedFactor:
    """定义一个由后端执行的双目自定义算子。"""
    return BinaryCombinedFactor(
        _spec(name, "custom", *args, function=function, **kwargs),
        factor1,
        factor2,
    )


def rolling_operator(
    factor: AbstractFactor,
    window: int,
    function: Callable[..., Any],
    *args: Any,
    name: str = "custom_rolling",
    **kwargs: Any,
) -> RollingWindowFactor:
    """定义一个由后端执行的时间窗口自定义算子。"""
    return RollingWindowFactor(
        _spec(name, "rolling", *args, function=function, **kwargs), window, factor
    )
