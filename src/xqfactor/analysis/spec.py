"""因子检验的声明式规范，不包含 Pandas 或统计库实现。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ProcessorSpec:
    """描述一个因子预处理步骤。"""

    name: str
    parameters: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class AnalysisSpec:
    """描述一个因子分析步骤及其所需输入。"""

    name: str
    required_inputs: tuple[str, ...] = ()
    parameters: tuple[tuple[str, Any], ...] = ()


class AnalysisResult(Protocol):
    """具体分析后端返回结果时应遵守的最小协议。"""

    @property
    def data(self) -> Any:
        """返回后端结果对象。"""


class AnalysisPipeline:
    """保存检验步骤顺序，具体执行交给后端。"""

    def __init__(self) -> None:
        """创建空的检验管道。"""
        self._components: list[tuple[str, ProcessorSpec | AnalysisSpec]] = []

    def append(self, name: str, component: ProcessorSpec | AnalysisSpec) -> None:
        """追加检验组件。

        输入：唯一组件名称和处理/分析规范。
        输出：无；重复名称会抛出 ValueError。
        """
        if any(existing_name == name for existing_name, _ in self._components):
            raise ValueError(f"检验组件名称重复: {name}")
        self._components.append((name, component))

    @property
    def components(self) -> tuple[tuple[str, ProcessorSpec | AnalysisSpec], ...]:
        """返回按注册顺序排列的不可变组件列表。"""
        return tuple(self._components)
