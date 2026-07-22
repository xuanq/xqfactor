"""xqfactor 的稳定定义指纹和执行缓存。"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, is_dataclass
from functools import partial
import hashlib
import json
import marshal
from threading import RLock
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd


def _stable_callable(value: Any, stack: tuple[int, ...]) -> Any:
    """将 callable 转换为稳定定义。

    输入：函数、偏函数、绑定方法或 callable 对象，以及当前递归路径。
    输出：包含实现、参数和闭包定义的可序列化结构。
    """
    value_id = id(value)
    if value_id in stack:
        return {"recursive_callable": stack.index(value_id)}
    nested_stack = (*stack, value_id)

    if isinstance(value, partial):
        return {
            "partial": _stable_callable(value.func, nested_stack),
            "args": _stable_value(value.args, nested_stack),
            "keywords": _stable_value(value.keywords, nested_stack),
        }

    bound_function = getattr(value, "__func__", None)
    bound_instance = getattr(value, "__self__", None)
    if bound_function is not None and bound_instance is not None:
        return {
            "bound_method": _stable_callable(bound_function, nested_stack),
            "instance": _stable_value(bound_instance, nested_stack),
        }

    module = getattr(value, "__module__", value.__class__.__module__)
    name = getattr(
        value,
        "__qualname__",
        getattr(value, "__name__", value.__class__.__qualname__),
    )
    definition: dict[str, Any] = {
        "callable": f"{module}.{name}",
        "version": getattr(value, "__xqfactor_version__", "1"),
    }
    code = getattr(value, "__code__", None)
    if code is not None:
        # ************************************************************
        # 函数字节码区分同名但实现不同的函数；defaults、kwdefaults 和
        # closure 递归区分由同一工厂函数生成的参数化闭包。
        # ************************************************************
        definition["code"] = hashlib.sha256(marshal.dumps(code)).hexdigest()
        definition["defaults"] = _stable_value(
            getattr(value, "__defaults__", None), nested_stack
        )
        definition["kwdefaults"] = _stable_value(
            getattr(value, "__kwdefaults__", None), nested_stack
        )
        closure = getattr(value, "__closure__", None)
        if closure is None:
            definition["closure"] = None
        else:
            closure_values: list[Any] = []
            for cell in closure:
                try:
                    cell_value = cell.cell_contents
                except ValueError:
                    cell_value = {"empty_cell": True}
                closure_values.append(_stable_value(cell_value, nested_stack))
            definition["closure"] = closure_values
        return definition

    state = getattr(value, "__dict__", None)
    if state:
        definition["state"] = _stable_value(state, nested_stack)
    return definition


def _stable_value(value: Any, stack: tuple[int, ...] = ()) -> Any:
    """将对象转换为可稳定序列化的结构。

    输入：因子定义、函数、参数或执行选项。
    输出：可交给 JSON 序列化的基础结构。
    """
    if is_dataclass(value):
        return _stable_value(asdict(value), stack)
    if callable(value):
        return _stable_callable(value, stack)
    if isinstance(value, pd.DataFrame):
        # ************************************************************
        # DataFrame 从二维值及两条轴转换为 JSON 可序列化结构；不能使用会
        # 截断中间行列的 repr，否则未展示区域不同的数据会发生指纹碰撞。
        # ************************************************************
        return {
            "dataframe": _stable_value(value.to_numpy(dtype=object).tolist(), stack),
            "index": _stable_value(value.index.tolist(), stack),
            "index_names": _stable_value(value.index.names, stack),
            "columns": _stable_value(value.columns.tolist(), stack),
            "column_names": _stable_value(value.columns.names, stack),
            "dtypes": [str(dtype) for dtype in value.dtypes],
        }
    if isinstance(value, pd.Series):
        return {
            "series": _stable_value(value.to_numpy(dtype=object).tolist(), stack),
            "index": _stable_value(value.index.tolist(), stack),
            "index_names": _stable_value(value.index.names, stack),
            "name": _stable_value(value.name, stack),
            "dtype": str(value.dtype),
        }
    if isinstance(value, np.ndarray):
        return {
            "ndarray": _stable_value(value.tolist(), stack),
            "shape": value.shape,
            "dtype": str(value.dtype),
        }
    if isinstance(value, Mapping):
        # 映射保留 key 的类型和值，避免整数 1 与字符串 "1" 被统一转成
        # 同一个 JSON 对象键；键值对按稳定键序列排序以消除插入顺序影响。
        items = [
            [_stable_value(key, stack), _stable_value(item, stack)]
            for key, item in value.items()
        ]
        return {
            "mapping": sorted(
                items,
                key=lambda pair: json.dumps(pair[0], sort_keys=True),
            )
        }
    if isinstance(value, (tuple, list)):
        return [_stable_value(item, stack) for item in value]
    if isinstance(value, set):
        items = [_stable_value(item, stack) for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {
        "type": f"{value.__class__.__module__}.{value.__class__.__qualname__}",
        "repr": repr(value),
    }


def stable_fingerprint(value: Any) -> str:
    """生成对象定义的稳定 SHA-256 指纹。

    输入：需要参与缓存身份计算的对象。
    输出：十六进制 SHA-256 字符串。
    """
    payload = json.dumps(
        _stable_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheKey:
    """因子节点在特定上下文中的执行缓存键。"""

    factor_fingerprint: str
    context_fingerprint: str


class ExecutionCache(Protocol):
    """DataFrame 执行缓存协议。"""

    def get(self, key: CacheKey) -> pd.DataFrame | None:
        """读取缓存；未命中时返回 None。"""

    def set(self, key: CacheKey, value: pd.DataFrame) -> None:
        """写入一个已完成计算的因子值。"""

    def clear(self) -> None:
        """清理缓存内容。"""


class MemoryCache:
    """线程安全的有界 DataFrame LRU 执行缓存。"""

    def __init__(self, maxsize: int = 128) -> None:
        """创建缓存。

        输入：最多保存的因子节点结果数量。
        输出：空的内存缓存实例。
        """
        if maxsize <= 0:
            raise ValueError("maxsize 必须为正整数")
        self.maxsize = maxsize
        self._values: OrderedDict[CacheKey, pd.DataFrame] = OrderedDict()
        self._lock = RLock()

    def get(self, key: CacheKey) -> pd.DataFrame | None:
        """读取缓存副本并更新 LRU 顺序。"""
        with self._lock:
            value = self._values.get(key)
            if value is None:
                return None
            self._values.move_to_end(key)
            return value.copy(deep=True)

    def set(self, key: CacheKey, value: pd.DataFrame) -> None:
        """写入 DataFrame 副本并淘汰最久未使用的结果。"""
        with self._lock:
            self._values[key] = value.copy(deep=True)
            self._values.move_to_end(key)
            while len(self._values) > self.maxsize:
                self._values.popitem(last=False)

    def clear(self) -> None:
        """清理所有缓存结果。"""
        with self._lock:
            self._values.clear()
