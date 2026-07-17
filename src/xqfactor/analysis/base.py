"""因子检验器的基础输入解析和抽象接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import pandas as pd

from xqfactor.factor import AbstractFactor
from xqfactor.runtime import EvaluationContext, ExecutionCache, MemoryCache


FactorInput = pd.DataFrame | AbstractFactor


class AbstractAnalyzer(ABC):
    """统一解析因子表达式和附加输入的检验器基类。"""

    def __init__(self, **inputs: FactorInput | None) -> None:
        """创建检验器并保存需要随 factors 一起求值的附加输入。

        输入：名称到因子表达式、DataFrame 或 None 的映射。
        输出：尚未执行统计逻辑的检验器实例。
        """
        self._inputs = {
            name: value for name, value in inputs.items() if value is not None
        }

    @staticmethod
    def _resolve_factor(
        value: FactorInput,
        context: EvaluationContext | None,
        cache: ExecutionCache | None,
    ) -> pd.DataFrame:
        """将因子表达式或现成 DataFrame 转换为检验输入。

        输入：因子表达式或二维因子值，以及可选执行上下文和缓存。
        输出：index 为时间、columns 为资产的 DataFrame 副本。
        """
        if isinstance(value, AbstractFactor):
            if context is None:
                raise ValueError("检验因子表达式时必须提供 context")
            return value.evaluate(context, cache)
        return value.copy(deep=True)

    @classmethod
    def _resolve_factors(
        cls,
        factors: Mapping[str, FactorInput],
        context: EvaluationContext | None,
        cache: ExecutionCache | None,
    ) -> dict[str, pd.DataFrame]:
        """批量解析名称到因子值的映射。

        输入：待求值的因子映射、可选执行上下文和共享缓存。
        输出：名称到二维 DataFrame 的新字典。
        """
        return {
            name: cls._resolve_factor(factor, context, cache)
            for name, factor in factors.items()
        }

    def analyze(
        self,
        factors: Mapping[str, FactorInput],
        *,
        context: EvaluationContext | None = None,
        cache: ExecutionCache | None = None,
    ) -> Any:
        """统一求值主因子和附加输入后执行统计逻辑。

        输入：待检验因子映射，以及因子表达式求值所需的可选上下文和缓存。
        输出：由具体检验器 `_analyze` 定义的统计结果。
        """
        # ************************************************************
        # 主因子和 returns、weights 等附加因子共用同一个 cache，保证同一
        # 表达式节点在一次检验中只求值一次；所有结果形状均为 (时间, 资产)。
        # ************************************************************
        active_cache = cache if cache is not None else MemoryCache()
        resolved_factors = self._resolve_factors(factors, context, active_cache)
        resolved_inputs = self._resolve_factors(
            self._inputs,
            context,
            active_cache,
        )
        return self._analyze(resolved_factors, **resolved_inputs)

    @abstractmethod
    def _analyze(
        self,
        factors: Mapping[str, pd.DataFrame],
        **inputs: pd.DataFrame,
    ) -> Any:
        """使用已经求值的二维因子值执行具体统计逻辑。

        输入：待检验因子 DataFrame 映射，以及已求值的附加输入。
        输出：由具体检验器定义的统计结果。
        """
