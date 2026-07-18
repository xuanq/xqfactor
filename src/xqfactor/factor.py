"""使用 Pandas DataFrame 传递数据的因子表达式节点。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

import numpy as np
import pandas as pd

from xqfactor.runtime import (
    CacheKey,
    EvaluationContext,
    ExecutionCache,
    LeafRequest,
    MemoryCache,
    stable_fingerprint,
)


FactorFunction = Callable[..., pd.DataFrame]


def _as_factor(value: Any) -> AbstractFactor:
    """将因子或常量转换为表达式节点。

    输入：因子定义、标量或二维 DataFrame。
    输出：可参与表达式组合的因子节点。
    """
    if isinstance(value, AbstractFactor):
        return value
    return ConstantFactor(value)


def _normalize_frame(
    value: pd.DataFrame,
    context: EvaluationContext,
) -> pd.DataFrame:
    """将因子值对齐到完整计算时间轴和资产轴。

    输入：待校验 DataFrame 和执行上下文。
    输出：形状从输入 ``(T', N')`` 对齐为
    ``(len(time_index), len(universe))`` 的 DataFrame；index 和 columns 分别
    变为 context.time_index 与 context.universe。
    """
    if not isinstance(value, pd.DataFrame):
        raise TypeError(
            f"因子计算结果必须为 pandas.DataFrame，实际为 {type(value).__name__}"
        )
    if value.index.has_duplicates:
        raise ValueError("因子 DataFrame index 不能包含重复值")
    if value.columns.has_duplicates:
        raise ValueError("因子 DataFrame columns 不能包含重复值")
    return value.reindex(index=context.time_index, columns=context.universe)


class AbstractFactor:
    """所有因子表达式的基类。"""

    def __add__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.add, self, _as_factor(other))

    def __sub__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.subtract, self, _as_factor(other))

    def __mul__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.multiply, self, _as_factor(other))

    def __truediv__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.true_divide, self, _as_factor(other))

    def __radd__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.add, _as_factor(other), self)

    def __rsub__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.subtract, _as_factor(other), self)

    def __rmul__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.multiply, _as_factor(other), self)

    def __rtruediv__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.true_divide, _as_factor(other), self)

    def __pow__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.power, self, _as_factor(other))

    def __floordiv__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.floor_divide, self, _as_factor(other))

    def __mod__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.mod, self, _as_factor(other))

    def __gt__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.greater, self, _as_factor(other))

    def __lt__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.less, self, _as_factor(other))

    def __ge__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.greater_equal, self, _as_factor(other))

    def __le__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.less_equal, self, _as_factor(other))

    def __and__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.logical_and, self, _as_factor(other))

    def __or__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.logical_or, self, _as_factor(other))

    def __invert__(self) -> AbstractFactor:
        return UnaryCombinedFactor(np.logical_not, self)

    def __ne__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(np.not_equal, self, _as_factor(other))

    def required_history(self) -> int:
        """返回该因子需要的历史周期数。"""
        return 0

    def definition(self) -> tuple[Any, ...]:
        """返回用于缓存指纹的因子定义。"""
        return (self.__class__.__name__,)

    def fingerprint(self) -> str:
        """返回当前因子定义指纹。"""
        return stable_fingerprint(self.definition())

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """在完整计算时间轴上计算当前节点。"""
        raise NotImplementedError

    def _evaluate(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """执行节点级缓存并返回完整计算时间轴上的 DataFrame。"""
        key = CacheKey(self.fingerprint(), context.fingerprint())
        cached = cache.get(key)
        if cached is not None:
            return cached

        # ************************************************************
        # 每个节点都先在完整时间轴上计算并统一对齐，根节点最后再截取输出区间。
        # ************************************************************
        value = _normalize_frame(self._compute(context, cache), context)
        cache.set(key, value)
        return value.copy(deep=True)

    def evaluate(
        self,
        context: EvaluationContext,
        cache: ExecutionCache | None = None,
    ) -> pd.DataFrame:
        """计算因子并返回最终输出区间。

        输入：显式执行上下文和可选的可复用执行缓存。
        输出：index 为 output_time_index、columns 为 universe 的 DataFrame。
        """
        active_cache = cache if cache is not None else MemoryCache()
        value = self._evaluate(context, active_cache)
        return value.iloc[context.output_start : context.output_end].copy(deep=True)


class _CallableFactor(AbstractFactor):
    """在 callable 被替换时同步更新实现指纹的内部节点基类。"""

    @property
    def func(self) -> FactorFunction:
        """返回当前 DataFrame 计算函数。"""
        return self._func

    @func.setter
    def func(self, value: FactorFunction) -> None:
        """替换计算函数并冻结新实现的定义指纹。"""
        self._func = value
        self._func_fingerprint = stable_fingerprint(value)


class LeafFactor(AbstractFactor):
    """由具体应用提供 resolver 的基础因子。"""

    def __init__(
        self,
        name: str,
        resolver: Callable[[LeafRequest], pd.DataFrame],
        definition_version: str = "1",
    ) -> None:
        """创建叶子因子。

        输入：因子名称、取数函数和数据定义版本。
        输出：不依赖具体数据源的叶子因子节点。
        """
        if not name:
            raise ValueError("name 不能为空")
        self.name = name
        self.resolver = resolver
        self.definition_version = definition_version

    @property
    def resolver(self) -> Callable[[LeafRequest], pd.DataFrame]:
        """返回当前叶子取数函数。"""
        return self._resolver

    @resolver.setter
    def resolver(self, value: Callable[[LeafRequest], pd.DataFrame]) -> None:
        """替换取数函数并冻结新实现的定义指纹。"""
        # resolver 的实现定义在赋值时冻结，避免调用计数等闭包运行状态
        # 改变缓存键；definition_version 仍在每次求值时动态参与指纹。
        self._resolver = value
        self._resolver_fingerprint = stable_fingerprint(value)

    def definition(self) -> tuple[Any, ...]:
        """返回叶子因子的名称、resolver 和版本定义。"""
        return (
            self.__class__.__name__,
            self.name,
            self._resolver_fingerprint,
            self.definition_version,
        )

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """调用应用提供的 resolver 获取完整时间轴上的二维数据。"""
        del cache
        return self.resolver(LeafRequest(self.name, context, self.definition_version))


class FixedFactor(AbstractFactor):
    """将任意因子固定到一个资产后广播到当前资产池。"""

    def __init__(self, factor: AbstractFactor, asset: str | int) -> None:
        """创建固定资产因子。

        输入：待固定的因子表达式和目标资产标识。
        输出：在目标资产上求值、再广播到调用方 universe 的因子节点。
        """
        if not isinstance(asset, (str, int)):
            raise TypeError("asset 必须是 str 或 int")
        self.factor = factor
        self.asset = asset

    def required_history(self) -> int:
        """返回被固定因子的历史需求。"""
        return self.factor.required_history()

    def definition(self) -> tuple[Any, ...]:
        """返回目标资产和被固定因子的定义。"""
        return (self.__class__.__name__, self.asset, self.factor.definition())

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """在单标的上下文中计算因子并广播到调用方资产轴。"""
        # ************************************************************
        # 子因子从调用方 universe=(N,) 切换到固定资产 universe=(1,)，
        # 时间轴、频率、语义、provider 版本和输出切片全部保持不变。
        # ************************************************************
        fixed_context = replace(context, universe=(self.asset,))
        fixed_value = self.factor._evaluate(fixed_context, cache)

        # ************************************************************
        # 子因子结果从 DataFrame (T, 1) 广播为 DataFrame (T, N)；
        # index 保持完整计算时间轴，columns 恢复为调用方 universe。
        # ************************************************************
        fixed_series = fixed_value.iloc[:, 0]
        return pd.concat(
            [fixed_series] * len(context.universe),
            axis=1,
            keys=context.universe,
        )


class UnaryCombinedFactor(_CallableFactor):
    """执行单输入 DataFrame 计算函数的因子节点。"""

    def __init__(
        self,
        func: FactorFunction,
        factor: AbstractFactor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """创建单输入组合因子。"""
        self.func = func
        self.factor = factor
        self.args = args
        self.kwargs = kwargs

    def required_history(self) -> int:
        """返回子因子的历史需求。"""
        return self.factor.required_history()

    def definition(self) -> tuple[Any, ...]:
        """返回计算函数、参数和依赖因子定义。"""
        return (
            self.__class__.__name__,
            self._func_fingerprint,
            self.args,
            self.kwargs,
            self.factor.definition(),
        )

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """计算子因子并执行单输入函数。"""
        value = self.factor._evaluate(context, cache)
        return self.func(value, *self.args, **self.kwargs)


class BinaryCombinedFactor(_CallableFactor):
    """执行双输入 DataFrame 计算函数的因子节点。"""

    def __init__(
        self,
        func: FactorFunction,
        arg1: AbstractFactor,
        arg2: AbstractFactor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """创建双输入组合因子。"""
        self.func = func
        self.arg1 = arg1
        self.arg2 = arg2
        self.args = args
        self.kwargs = kwargs

    def required_history(self) -> int:
        """返回两个输入因子中的最大历史需求。"""
        return max(self.arg1.required_history(), self.arg2.required_history())

    def definition(self) -> tuple[Any, ...]:
        """返回计算函数、参数和两个依赖因子定义。"""
        return (
            self.__class__.__name__,
            self._func_fingerprint,
            self.args,
            self.kwargs,
            self.arg1.definition(),
            self.arg2.definition(),
        )

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """计算两个子因子并执行双输入函数。"""
        value1 = self.arg1._evaluate(context, cache)
        value2 = self.arg2._evaluate(context, cache)
        return self.func(value1, value2, *self.args, **self.kwargs)


class CombinedFactor(_CallableFactor):
    """执行一个或多个 DataFrame 输入的自定义计算函数。"""

    def __init__(
        self,
        func: FactorFunction,
        *factors: AbstractFactor,
        **kwargs: Any,
    ) -> None:
        """创建多输入组合因子。

        输入：与具体因子无关的计算函数、待应用的因子和函数关键字参数。
        输出：可继续参与表达式组合的因子节点。
        """
        if not factors:
            raise ValueError("CombinedFactor 至少需要一个输入因子")
        self.func = func
        self.factors = factors
        self.kwargs = kwargs

    def required_history(self) -> int:
        """返回所有输入因子的最大历史需求。"""
        return max(factor.required_history() for factor in self.factors)

    def definition(self) -> tuple[Any, ...]:
        """返回计算函数、参数和全部依赖因子定义。"""
        return (
            self.__class__.__name__,
            self._func_fingerprint,
            self.kwargs,
            tuple(factor.definition() for factor in self.factors),
        )

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """计算全部子因子并执行组合函数。"""
        values = tuple(factor._evaluate(context, cache) for factor in self.factors)
        return self.func(*values, **self.kwargs)


class RefFactor(AbstractFactor):
    """沿时间轴引用历史或未来值的节点。"""

    def __init__(self, factor: AbstractFactor, periods: int) -> None:
        """创建时间引用节点。"""
        self.factor = factor
        self.periods = periods

    def required_history(self) -> int:
        """返回引用周期和子因子历史需求之和。"""
        return self.factor.required_history() + abs(self.periods)

    def definition(self) -> tuple[Any, ...]:
        """返回引用周期和依赖因子定义。"""
        return (self.__class__.__name__, self.periods, self.factor.definition())

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """沿时间轴移动子因子值。"""
        return self.factor._evaluate(context, cache).shift(self.periods)


class RollingWindowFactor(_CallableFactor):
    """执行单输入时间窗口函数的因子节点。"""

    def __init__(
        self,
        func: Callable[..., pd.DataFrame],
        window: int,
        factor: AbstractFactor,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """创建时间窗口节点。"""
        if window <= 0:
            raise ValueError("window 必须为正整数")
        self.func = func
        self.window = window
        self.factor = factor
        self.args = args
        self.kwargs = kwargs

    def required_history(self) -> int:
        """返回窗口新增的历史需求和子因子历史需求。"""
        return self.factor.required_history() + self.window - 1

    def definition(self) -> tuple[Any, ...]:
        """返回窗口函数、长度、参数和依赖因子定义。"""
        return (
            self.__class__.__name__,
            self._func_fingerprint,
            self.window,
            self.args,
            self.kwargs,
            self.factor.definition(),
        )

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """计算子因子并执行窗口函数。"""
        value = self.factor._evaluate(context, cache)
        return self.func(
            value,
            self.window,
            *self.args,
            **self.kwargs,
        )


class ConstantFactor(AbstractFactor):
    """将常量或二维表包装为因子节点。"""

    def __init__(self, value: Any) -> None:
        """保存待广播的常量或 DataFrame。"""
        self.value = value

    def definition(self) -> tuple[Any, ...]:
        """返回常量定义。"""
        return (self.__class__.__name__, self.value)

    def _compute(
        self,
        context: EvaluationContext,
        cache: ExecutionCache,
    ) -> pd.DataFrame:
        """将常量广播到完整的时间资产二维轴。"""
        del cache
        if isinstance(self.value, pd.DataFrame):
            return self.value.copy(deep=True)
        return pd.DataFrame(
            self.value,
            index=context.time_index,
            columns=context.universe,
        )
