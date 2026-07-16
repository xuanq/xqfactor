"""xqfactor 的无数据源运行时协议和执行缓存实现。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
from threading import RLock
from typing import Any, Callable, Mapping, Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from xqfactor.factor import AbstractFactor


AssetId = str | int


def _stable_value(value: Any) -> Any:
    """将常见参数转换为可稳定序列化的结构。

    输入：任意因子参数、执行选项或可调用对象。
    输出：可以用于生成稳定指纹的 Python 基础结构。
    """
    if is_dataclass(value):
        return _stable_value(asdict(value))
    if callable(value):
        version = getattr(value, "__xqfactor_version__", "1")
        return {
            "callable": f"{value.__module__}.{value.__qualname__}",
            "version": version,
        }
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item) for item in value]
    if isinstance(value, set):
        return sorted(_stable_value(item) for item in value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {
        "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "repr": repr(value),
    }


def stable_fingerprint(value: Any) -> str:
    """生成参数或因子定义的稳定 SHA-256 指纹。

    输入：需要参与缓存身份计算的对象。
    输出：十六进制 SHA-256 字符串。
    """
    payload = json.dumps(
        _stable_value(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FactorValue:
    """因子值的统一二维轴契约。

    ``data`` 由具体后端持有，可以是 Pandas、Polars、NumPy 或 PyTorch 对象。
    ``time_index`` 和 ``assets`` 描述 data 的两条逻辑轴，后端负责保证实际数据
    与它们一致。
    """

    data: Any
    time_index: tuple[Any, ...]
    assets: tuple[AssetId, ...]
    metadata: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_index", tuple(self.time_index))
        object.__setattr__(self, "assets", tuple(self.assets))
        object.__setattr__(self, "metadata", tuple(self.metadata))


@dataclass(frozen=True)
class EvaluationContext:
    """一次因子求值所需的显式上下文。

    ``time_index`` 应包含因子所需的历史数据，``output_start`` 到 ``output_end``
    定义最终返回区间。核心只消费已经准备好的时间轴，不查询任何交易日历。
    """

    time_index: tuple[Any, ...]
    universe: tuple[AssetId, ...]
    frequency: str
    output_start: int = 0
    output_end: int | None = None
    semantics: tuple[tuple[str, Any], ...] = ()
    provider_version: str = "default"

    def __post_init__(self) -> None:
        time_index = tuple(self.time_index)
        universe = tuple(self.universe)
        if not time_index:
            raise ValueError("time_index 不能为空")
        if not universe:
            raise ValueError("universe 不能为空")
        output_end = len(time_index) if self.output_end is None else self.output_end
        if not 0 <= self.output_start <= output_end <= len(time_index):
            raise ValueError("output_start/output_end 超出 time_index 范围")
        object.__setattr__(self, "time_index", time_index)
        object.__setattr__(self, "universe", universe)
        object.__setattr__(self, "output_end", output_end)
        object.__setattr__(self, "semantics", tuple(self.semantics))

    @property
    def output_time_index(self) -> tuple[Any, ...]:
        """返回最终结果对应的时间轴。"""
        return self.time_index[self.output_start : self.output_end]

    @property
    def start_time(self) -> Any:
        """返回最终结果起始时间。"""
        return self.output_time_index[0]

    @property
    def end_time(self) -> Any:
        """返回最终结果结束时间。"""
        return self.output_time_index[-1]

    def fingerprint(self, backend_version: str) -> str:
        """生成包含执行范围、资产池和后端版本的缓存上下文指纹。"""
        return stable_fingerprint(
            {
                "time_index": self.time_index,
                "universe": self.universe,
                "frequency": self.frequency,
                "output_start": self.output_start,
                "output_end": self.output_end,
                "semantics": self.semantics,
                "provider_version": self.provider_version,
                "backend_version": backend_version,
            }
        )


@dataclass(frozen=True)
class LeafRequest:
    """叶子因子的取数请求。"""

    factor_name: str
    context: EvaluationContext
    definition_version: str


@dataclass(frozen=True)
class OperatorSpec:
    """算子定义及其后端无关的元数据。"""

    name: str
    kind: str
    args: tuple[Any, ...] = ()
    kwargs: tuple[tuple[str, Any], ...] = ()
    function: Callable[..., Any] | None = None
    version: str = "1"

    def definition(self) -> tuple[Any, ...]:
        """返回用于定义指纹和后端分派的不可变描述。"""
        return (
            self.name,
            self.kind,
            self.args,
            self.kwargs,
            self.function,
            self.version,
        )


class OperatorRegistry:
    """保存应用注册的算子定义，避免名称和语义版本散落在业务代码中。"""

    def __init__(self) -> None:
        """创建空的算子注册表。"""
        self._operators: dict[str, OperatorSpec] = {}

    def register(self, spec: OperatorSpec, overwrite: bool = False) -> OperatorSpec:
        """注册算子。

        输入：算子定义和是否允许覆盖同名定义。
        输出：已注册的算子定义；重复注册且不允许覆盖时抛出 ValueError。
        """
        if spec.name in self._operators and not overwrite:
            raise ValueError(f"算子名称重复: {spec.name}")
        self._operators[spec.name] = spec
        return spec

    def get(self, name: str) -> OperatorSpec:
        """按名称读取算子定义。"""
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"未注册算子: {name}") from exc

    @property
    def names(self) -> tuple[str, ...]:
        """返回已注册算子名称。"""
        return tuple(self._operators)


@dataclass(frozen=True)
class CacheKey:
    """一次因子节点在特定执行上下文中的缓存键。"""

    factor_fingerprint: str
    context_fingerprint: str


class ExecutionCache(Protocol):
    """执行缓存协议，可由应用替换为任意存储实现。"""

    def get(self, key: CacheKey) -> FactorValue | None:
        """读取缓存；未命中时返回 None。"""

    def set(self, key: CacheKey, value: FactorValue) -> None:
        """写入一个已完成计算的因子值。"""

    def clear(self) -> None:
        """清理缓存内容。"""


class MemoryCache:
    """线程安全的有界内存 LRU 执行缓存。"""

    def __init__(self, maxsize: int = 128) -> None:
        """创建缓存。

        输入：``maxsize`` 为最多保存的因子节点结果数量。
        输出：无，实例化一个空缓存。
        """
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正整数")
        self.maxsize = maxsize
        self._values: OrderedDict[CacheKey, FactorValue] = OrderedDict()
        self._lock = RLock()

    def get(self, key: CacheKey) -> FactorValue | None:
        """读取缓存并更新 LRU 顺序。"""
        with self._lock:
            value = self._values.get(key)
            if value is not None:
                self._values.move_to_end(key)
            return value

    def set(self, key: CacheKey, value: FactorValue) -> None:
        """写入缓存并淘汰最久未使用的结果。"""
        with self._lock:
            self._values[key] = value
            self._values.move_to_end(key)
            while len(self._values) > self.maxsize:
                self._values.popitem(last=False)

    def clear(self) -> None:
        """清理所有缓存结果。"""
        with self._lock:
            self._values.clear()


class ComputeBackend(Protocol):
    """因子运算后端协议。"""

    name: str
    version: str

    def normalize(self, value: Any, context: EvaluationContext) -> FactorValue:
        """将后端原始值标准化为 FactorValue。"""

    def constant(self, value: Any, context: EvaluationContext) -> FactorValue:
        """将常量广播到上下文的二维轴。"""

    def apply(
        self,
        spec: OperatorSpec,
        inputs: Sequence[FactorValue],
        context: EvaluationContext,
    ) -> FactorValue:
        """对一个或多个因子值执行算子。"""

    def shift(
        self, value: FactorValue, periods: int, context: EvaluationContext
    ) -> FactorValue:
        """沿时间轴移动因子值。"""

    def rolling(
        self,
        spec: OperatorSpec,
        value: FactorValue,
        window: int,
        context: EvaluationContext,
    ) -> FactorValue:
        """执行时间序列窗口算子。"""

    def slice(
        self,
        value: FactorValue,
        start: int,
        end: int,
        context: EvaluationContext,
    ) -> FactorValue:
        """截取最终输出区间。"""


class FactorRuntime:
    """绑定计算后端和执行缓存的因子运行时。"""

    def __init__(
        self, backend: ComputeBackend, cache: ExecutionCache | None = None
    ) -> None:
        """创建运行时。

        输入：计算后端和可选执行缓存。
        输出：可传给 ``AbstractFactor.evaluate`` 的运行时对象。
        """
        self.backend = backend
        self.cache = cache if cache is not None else MemoryCache()

    def evaluate(
        self, factor: AbstractFactor, context: EvaluationContext
    ) -> FactorValue:
        """计算因子并截取上下文声明的最终输出区间。"""
        value = factor._evaluate(context, self)
        return self.backend.slice(
            value, context.output_start, context.output_end, context
        )
