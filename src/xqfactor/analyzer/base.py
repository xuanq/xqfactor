from __future__ import annotations

from typing import Callable, Dict, OrderedDict, Tuple

import pandas as pd

from xqfactor import AbstractFactor
from xqfactor.config import Config, global_config


class AbstractAnalyzer:
    def __init__(
        self, config: Config = None, keep_processed_results: bool = False
    ) -> None:
        if config is None:
            config = global_config
        self.config = config
        self.keep_processed_results = keep_processed_results

        self.processors: OrderedDict[str, "BaseProcessor"] = OrderedDict()
        self.processed_results: Dict[Tuple[str, str], pd.DataFrame] = {}

    def use_config(self, config: Config):
        self.config = config
        for processor in self.processors.values():
            processor.use_config(config)
        return self

    def register_processor(self, name: str, processor: "BaseProcessor"):
        self.processors[name] = processor.use_config(self.config)

    def process(
        self, factor_name: str, factor: pd.DataFrame | AbstractFactor
    ) -> pd.DataFrame:
        factor = self._get_value(factor)
        for processor_name, processor in self.processors.items():
            factor = processor.process(factor)
            if self.keep_processed_results:
                self.processed_results[(factor_name, processor_name)] = factor
        return factor

    def _analyze(self, factors: Dict[str, pd.DataFrame | AbstractFactor]):
        raise NotImplementedError

    def analyze(self, factors: Dict[str, pd.DataFrame | AbstractFactor]):
        factors = {name: self.process(name, factor) for name, factor in factors.items()}
        result = self._analyze(factors)

        if self.keep_processed_results:
            # 将字典的tuple key展开为多级索引的前几层，再加上子DataFrame自身的索引
            dfs_with_multiindex = []
            for tuple_key, df in self.processed_results.items():
                # 复制子DataFrame
                df_with_multi = df.copy()
                # 创建多级索引：将tuple_key的每个元素转为列表 + 子DataFrame原索引
                # 确保所有层级都是可迭代对象
                levels = [[key_part] for key_part in tuple_key] + [df.index]
                df_with_multi.index = pd.MultiIndex.from_product(
                    levels,
                    # 为各级索引命名
                    names=["factor", "processor", "datetime"],
                )

                dfs_with_multiindex.append(df_with_multi)

            # 3. 合并所有DataFrame
            combined_df = pd.concat(dfs_with_multiindex)
            result.intermediate_results = combined_df

        return result

    def _get_value(self, data: pd.DataFrame | AbstractFactor) -> pd.DataFrame:
        """Convert AbstractFactor to its value if needed."""
        if issubclass(data.__class__, AbstractFactor):
            data = data._compute(self.config)
        return data


class BaseProcessor(AbstractAnalyzer):
    def __init__(
        self,
        method: str,
        *args,
        config=None,
        **kwargs,
    ) -> None:
        super().__init__(config)
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def process(self, factor: pd.DataFrame | AbstractFactor) -> pd.DataFrame:
        factor = self._get_value(factor)
        method: Callable = getattr(self, self.method)
        return method(factor, *self.args, **self.kwargs)
