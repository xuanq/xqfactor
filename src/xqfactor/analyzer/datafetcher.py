import pandas as pd

from xqfactor.analyzer.base import AbstractAnalyzer
from xqfactor.config import Config
from xqfactor.factor import AbstractFactor


class DataFetcher(AbstractAnalyzer):
    def __init__(self, config: Config = None) -> None:
        super().__init__(config)

    def process(self, factor: pd.DataFrame | AbstractFactor) -> pd.DataFrame:
        return self._get_value(factor)
