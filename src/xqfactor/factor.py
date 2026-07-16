"""数据源无关的因子表达式图和通用求值节点。"""

from __future__ import annotations

from typing import Any, Callable

from xqfactor.runtime import (
    CacheKey,
    EvaluationContext,
    FactorRuntime,
    FactorValue,
    LeafRequest,
    OperatorSpec,
    stable_fingerprint,
)


def _as_factor(value: Any) -> AbstractFactor:
    """将标量或因子转换为表达式节点。"""
    if isinstance(value, AbstractFactor):
        return value
    return ConstantFactor(value)


class AbstractFactor:
    """所有因子表达式的基类。"""

    def __add__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("add", "binary"), self, _as_factor(other)
        )

    def __sub__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("subtract", "binary"), self, _as_factor(other)
        )

    def __mul__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("multiply", "binary"), self, _as_factor(other)
        )

    def __truediv__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("true_divide", "binary"), self, _as_factor(other)
        )

    def __radd__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("add", "binary"), _as_factor(other), self
        )

    def __rsub__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("subtract", "binary"), _as_factor(other), self
        )

    def __rmul__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("multiply", "binary"), _as_factor(other), self
        )

    def __rtruediv__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("true_divide", "binary"), _as_factor(other), self
        )

    def __pow__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("power", "binary"), self, _as_factor(other)
        )

    def __floordiv__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("floor_divide", "binary"), self, _as_factor(other)
        )

    def __mod__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("mod", "binary"), self, _as_factor(other)
        )

    def __gt__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("greater", "binary"), self, _as_factor(other)
        )

    def __lt__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("less", "binary"), self, _as_factor(other)
        )

    def __ge__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("greater_equal", "binary"), self, _as_factor(other)
        )

    def __le__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("less_equal", "binary"), self, _as_factor(other)
        )

    def __and__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("logical_and", "binary"), self, _as_factor(other)
        )

    def __or__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("logical_or", "binary"), self, _as_factor(other)
        )

    def __invert__(self) -> AbstractFactor:
        return UnaryCombinedFactor(OperatorSpec("logical_not", "unary"), self)

    def __ne__(self, other: Any) -> AbstractFactor:
        return BinaryCombinedFactor(
            OperatorSpec("not_equal", "binary"), self, _as_factor(other)
        )

    def required_history(self) -> int:
        """返回该因子需要的历史周期数。"""
        return 0

    def definition(self) -> tuple[Any, ...]:
        """返回因子定义树，用于缓存和可重复性判断。"""
        return (self.__class__.__name__,)

    def fingerprint(self) -> str:
        """返回因子定义的稳定指纹。"""
        return stable_fingerprint(self.definition())

    def _compute(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """在完整计算时间轴上计算当前节点。"""
        raise NotImplementedError

    def _evaluate(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """执行节点级缓存，并返回未截取的完整计算结果。"""
        key = CacheKey(self.fingerprint(), context.fingerprint(runtime.backend.version))
        cached = runtime.cache.get(key)
        if cached is not None:
            return cached

        # ***** 先递归计算完整时间轴，再由根节点统一截取输出区间。*****
        value = runtime.backend.normalize(self._compute(context, runtime), context)
        runtime.cache.set(key, value)
        return value

    def evaluate(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """在给定上下文和运行时中计算当前因子。

        输入：显式时间轴、资产池、频率以及计算后端和缓存。
        输出：截取到 output 区间的 FactorValue。
        """
        return runtime.evaluate(self, context)


class LeafFactor(AbstractFactor):
    """由具体应用提供取数函数的基础因子。"""

    def __init__(
        self,
        name: str,
        resolver: Callable[[LeafRequest], FactorValue | Any],
        definition_version: str = "1",
    ) -> None:
        """创建用户自定义叶子因子。

        输入：因子名称、接收 LeafRequest 的取数函数以及定义版本。
        输出：一个不依赖任何数据 API 的叶子因子节点。
        """
        if not name:
            raise ValueError("name 不能为空")
        self.name = name
        self.resolver = resolver
        self.definition_version = definition_version

    def definition(self) -> tuple[Any, ...]:
        """返回叶子因子的名称、resolver 和版本定义。"""
        return (
            self.__class__.__name__,
            self.name,
            self.resolver,
            self.definition_version,
        )

    def _compute(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue | Any:
        """调用应用提供的 resolver 获取叶子数据。"""
        request = LeafRequest(self.name, context, self.definition_version)
        return self.resolver(request)


class UnaryCombinedFactor(AbstractFactor):
    """单输入算子节点。"""

    def __init__(self, spec: OperatorSpec, factor: AbstractFactor) -> None:
        self.spec = spec
        self.factor = factor

    def required_history(self) -> int:
        """返回子因子的历史需求。"""
        return self.factor.required_history()

    def definition(self) -> tuple[Any, ...]:
        """返回单输入算子定义。"""
        return (
            self.__class__.__name__,
            self.spec.definition(),
            self.factor.definition(),
        )

    def _compute(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """计算子因子后交给后端执行单目算子。"""
        value = self.factor._evaluate(context, runtime)
        return runtime.backend.apply(self.spec, (value,), context)


class BinaryCombinedFactor(AbstractFactor):
    """双输入算子节点，与数据源无关。"""

    def __init__(
        self,
        spec: OperatorSpec,
        arg1: AbstractFactor,
        arg2: AbstractFactor,
    ) -> None:
        self.spec = spec
        self.arg1 = arg1
        self.arg2 = arg2

    def required_history(self) -> int:
        """返回两个输入因子中的最大历史需求。"""
        return max(self.arg1.required_history(), self.arg2.required_history())

    def definition(self) -> tuple[Any, ...]:
        """返回双输入算子定义。"""
        return (
            self.__class__.__name__,
            self.spec.definition(),
            self.arg1.definition(),
            self.arg2.definition(),
        )

    def _compute(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """计算两个子因子后交给后端执行双目算子。"""
        values = (
            self.arg1._evaluate(context, runtime),
            self.arg2._evaluate(context, runtime),
        )
        return runtime.backend.apply(self.spec, values, context)


class CombinedFactor(AbstractFactor):
    """多输入自定义算子节点。"""

    def __init__(self, spec: OperatorSpec, *factors: AbstractFactor) -> None:
        self.spec = spec
        self.factors = factors

    def required_history(self) -> int:
        """返回所有输入因子的最大历史需求。"""
        return max((factor.required_history() for factor in self.factors), default=0)

    def definition(self) -> tuple[Any, ...]:
        """返回多输入算子定义。"""
        return (
            self.__class__.__name__,
            self.spec.definition(),
            tuple(factor.definition() for factor in self.factors),
        )

    def _compute(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """计算所有子因子后交给后端执行多目算子。"""
        values = tuple(factor._evaluate(context, runtime) for factor in self.factors)
        return runtime.backend.apply(self.spec, values, context)


class RefFactor(AbstractFactor):
    """沿时间轴引用历史或未来值的节点。"""

    def __init__(self, factor: AbstractFactor, periods: int) -> None:
        self.factor = factor
        self.periods = periods

    def required_history(self) -> int:
        """返回引用周期和子因子历史需求之和。"""
        return self.factor.required_history() + abs(self.periods)

    def definition(self) -> tuple[Any, ...]:
        """返回引用节点定义。"""
        return (self.__class__.__name__, self.periods, self.factor.definition())

    def _compute(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """计算子因子并由后端沿时间轴移动。"""
        value = self.factor._evaluate(context, runtime)
        return runtime.backend.shift(value, self.periods, context)


class RollingWindowFactor(AbstractFactor):
    """单输入时间窗口算子节点。"""

    def __init__(self, spec: OperatorSpec, window: int, factor: AbstractFactor) -> None:
        """创建窗口节点。

        输入：窗口算子定义、正整数窗口长度和输入因子。
        输出：一个等待后端执行的窗口因子节点。
        """
        if window <= 0:
            raise ValueError("window 必须为正整数")
        self.spec = spec
        self.window = window
        self.factor = factor

    def required_history(self) -> int:
        """返回窗口新增的历史需求和子因子历史需求。"""
        return self.factor.required_history() + self.window - 1

    def definition(self) -> tuple[Any, ...]:
        """返回窗口节点定义。"""
        return (
            self.__class__.__name__,
            self.spec.definition(),
            self.window,
            self.factor.definition(),
        )

    def _compute(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """计算子因子并交给后端执行窗口运算。"""
        value = self.factor._evaluate(context, runtime)
        return runtime.backend.rolling(self.spec, value, self.window, context)


class ConstantFactor(AbstractFactor):
    """将标量或后端常量包装为因子节点。"""

    def __init__(self, value: Any) -> None:
        self.value = value

    def definition(self) -> tuple[Any, ...]:
        """返回常量定义。"""
        return (self.__class__.__name__, self.value)

    def _compute(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> FactorValue:
        """将常量广播到当前上下文的二维轴。"""
        return runtime.backend.constant(self.value, context)


# 保留旧名称的结构别名，避免外部应用必须立即改动所有 import。
OperatorNode = CombinedFactor
