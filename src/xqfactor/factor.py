from functools import lru_cache
from typing import Callable, Tuple
from warnings import warn

import numpy as np
import pandas as pd

from xqfactor.config import Config, get_api, global_config


class AbstractFactor:
    shift = 0
    idx = pd.DatetimeIndex([])
    config = global_config

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.add, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.add, self, other)

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.subtract, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.subtract, self, other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.multiply, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.multiply, self, other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.true_divide, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.true_divide, self, other)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.add, ConstantFactor(other), self)
        return BinaryCombinedFactor(np.add, other, self)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.subtract, ConstantFactor(other), self)
        return BinaryCombinedFactor(np.subtract, other, self)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.multiply, ConstantFactor(other), self)
        return BinaryCombinedFactor(np.multiply, other, self)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.true_divide, ConstantFactor(other), self)
        return BinaryCombinedFactor(np.true_divide, other, self)

    def __pow__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.power, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.power, self, other)

    def __floordiv__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.floor_divide, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.floor_divide, self, other)

    def __mod__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.mod, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.mod, self, other)

    def __gt__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.greater, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.greater, self, other)

    def __lt__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.less, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.less, self, other)

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.greater_equal, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.greater_equal, self, other)

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.less_equal, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.less_equal, self, other)

    def __and__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.logical_and, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.logical_and, self, other)

    def __or__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.logical_or, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.logical_or, self, other)

    def __invert__(self):
        return UnaryCombinedFactor(np.logical_not, self)

    def __ne__(self, other):
        if isinstance(other, (int, float)):
            return BinaryCombinedFactor(np.not_equal, self, ConstantFactor(other))
        return BinaryCombinedFactor(np.not_equal, self, other)

    def use_config(self, config: Config):
        self.config = config
        return self

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def value(self):
        return self._compute(config=self.config)


class LeafFactor(AbstractFactor):
    def __init__(self, name, api="default"):
        self.name = name
        self.api = api

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        start_time: pd.Timestamp = config.get_option("start_time")
        end_time: pd.Timestamp = config.get_option("end_time")
        frequency: str = config.get_option("frequency")
        codes: Tuple[str] = config.get_option("universe")
        index: pd.DatetimeIndex = config.get_option("index")
        if start_time in index:
            start_loc = index.get_loc(start_time) - refs[0]
        else:
            start_loc = index.get_loc(index.asof(start_time)) - refs[0] + 1
        query_start_time = index[start_loc - 1]
        if end_time in index:
            end_loc = index.get_loc(end_time) + refs[1]
        else:
            end_loc = index.get_loc(index.asof(end_time)) + refs[1]
        query_end_time = index[end_loc]
        index = index[start_loc : end_loc + 1]

        if frequency in ("W", "ME"):
            query_frequency = "D"
        else:
            query_frequency = frequency

        api = get_api(self.api)
        data = api.get_factor(
            factors=self.name,
            codes=codes,
            start_time=query_start_time,
            end_time=query_end_time,
            frequency=query_frequency,
            panel=False,
        )
        if data.empty:
            warn(f"Factor {self.name} is empty")
            data = pd.DataFrame(index=index, columns=codes)
            data.index.name = "datetime"
            data.columns.name = "code"
            return data

        data = data["value"]
        if frequency == "tick":
            data = data.reset_index()
            data["has_newer"] = data.duplicated(
                subset=["datetime", "code"], keep="last"
            )
            data.loc[data["has_newer"], "datetime"] = data.loc[
                data["has_newer"], "datetime"
            ] - pd.Timedelta(seconds=0.5)
            data = (
                data.drop(columns="has_newer")
                .set_index(["datetime", "code"])["value"]
                .unstack(-1)
            )
            data = data.resample(
                f"{config.tick_freq}s", closed="right", label="right"
            ).last()
        else:
            data = data.unstack(level=-1)
        return data.reindex(columns=codes, index=index)


class UnaryCombinedFactor(AbstractFactor):
    def __init__(self, func, factor: AbstractFactor, *args, **extra_kwargs):
        self._func = func
        self._factor = factor
        self._args = args
        self._extra_kwargs = extra_kwargs

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        value = self._factor._compute(config, refs)
        return self._func(value, *self._args, **self._extra_kwargs)


class BinaryCombinedFactor(AbstractFactor):
    def __init__(
        self,
        func,
        arg1: AbstractFactor,
        arg2: AbstractFactor,
        *extra_args,
        **extra_kwargs,
    ):
        self._func = func
        self._arg1 = arg1
        self._arg2 = arg2
        self._extra_args = extra_args
        self._extra_kwargs = extra_kwargs

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        return self._func(
            self._arg1._compute(config, refs),
            self._arg2._compute(config, refs),
            *self._extra_args,
            **self._extra_kwargs,
        )


class CombinedFactor(AbstractFactor):
    def __init__(
        self,
        func: Callable[..., pd.DataFrame],
        *factors: AbstractFactor,
        **extra_kwargs,
    ):
        self._func = func
        self._factors = factors
        self._extra_kwargs = extra_kwargs

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        return self._func(
            *[factor._compute(config, refs) for factor in self._factors],
            **self._extra_kwargs,
        )


class RefFactor(AbstractFactor):
    def __init__(self, factor: AbstractFactor, n: int):
        self._factor = factor
        self._n = n

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        left_ref, right_ref = refs
        if self._n > 0:
            left_ref += self._n
            values = self._factor._compute(config, (left_ref, right_ref))
            rolled_values = values.shift(self._n)[self._n :]
        elif self._n < 0:
            right_ref -= self._n
            values = self._factor._compute(config, (left_ref, right_ref))
            rolled_values = values.shift(self._n)[: self._n]
        else:
            rolled_values = self._factor._compute(config, (left_ref, right_ref))
        return rolled_values


class RollingWindowFactor(AbstractFactor):
    def __init__(
        self,
        func: Callable[[pd.DataFrame, int], pd.DataFrame],
        window: int,
        factor: AbstractFactor,
    ):
        self._func = func
        self._window = window
        self._factor = factor

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        left_ref, right_ref = refs
        left_ref += self._window
        values = self._factor._compute(config, (left_ref, right_ref))
        values = self._func(values, self._window)
        return values[self._window :]


class ConstantFactor(AbstractFactor):
    def __init__(self, value):
        self._value = value

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        return self._value


class SingleLeafFactor(LeafFactor):
    """
    返回一个指定标的时序，映射到universe上
    所有标的在界面上的值都是相同的，通常适用于业绩基准的场景
    """

    def __init__(self, name, code: str, api="default"):
        super().__init__(name, api)

        if not isinstance(code, str):
            raise ValueError("code must be str")
        self.code = code

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        start_time: pd.Timestamp = config.get_option("start_time")
        end_time: pd.Timestamp = config.get_option("end_time")
        frequency: str = config.get_option("frequency")
        codes: Tuple[str] = config.get_option("universe")
        index: pd.DatetimeIndex = config.get_option("index")
        if start_time in index:
            start_loc = index.get_loc(start_time) - refs[0]
        else:
            start_loc = index.get_loc(index.asof(start_time)) - refs[0] + 1
        query_start_time = index[start_loc - 1]
        if end_time in index:
            end_loc = index.get_loc(end_time) + refs[1]
        else:
            end_loc = index.get_loc(index.asof(end_time)) + refs[1]
        query_end_time = index[end_loc]
        index = index[start_loc : end_loc + 1]

        if frequency in ("W", "ME"):
            query_frequency = "D"
        else:
            query_frequency = frequency

        api = get_api(self.api)
        data = api.get_factor(
            factors=self.name,
            codes=self.code,
            start_time=query_start_time,
            end_time=query_end_time,
            frequency=query_frequency,
            panel=False,
        )["value"]

        if frequency == "tick":
            data = data.reset_index()
            data["has_newer"] = data.duplicated(
                subset=["datetime", "code"], keep="last"
            )
            data.loc[data["has_newer"], "datetime"] = data.loc[
                data["has_newer"], "datetime"
            ] - pd.Timedelta(seconds=0.5)
            data = (
                data.drop(columns="has_newer")
                .set_index(["datetime", "code"])["value"]
                .unstack(-1)
            )
            data = (
                data.resample(f"{config.tick_freq}s", closed="right", label="right")
                .last()
                .reindex(index=index)
            )
        else:
            data = data.unstack(level=-1).reindex(index=index)

        if len(data.columns) != 1:
            raise ValueError(f"must be a single column dataframe,got {data.columns}")
        data = data.squeeze()
        return pd.DataFrame({col: data for col in codes})


class ObjectedLeafFactor(LeafFactor):
    """
    双主键属性，锁定单一主键后的因子值
    例如：
        持仓：（锁定客户号后，该客户对universe里的标的的时许状态
        基差：确定某一基础标的后，universe相对于其的差值
    """

    def __init__(self, name: str, object: str | int, api="default"):
        super().__init__(name, api)
        self.object = object

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        start_time: pd.Timestamp = config.get_option("start_time")
        end_time: pd.Timestamp = config.get_option("end_time")
        frequency: str = config.get_option("frequency")
        codes: Tuple[str] = config.get_option("universe")
        index: pd.DatetimeIndex = config.get_option("index")
        if start_time in index:
            start_loc = index.get_loc(start_time) - refs[0]
        else:
            start_loc = index.get_loc(index.asof(start_time)) - refs[0] + 1
        query_start_time = index[start_loc - 1]
        if end_time in index:
            end_loc = index.get_loc(end_time) + refs[1]
        else:
            end_loc = index.get_loc(index.asof(end_time)) + refs[1]
        query_end_time = index[end_loc]
        index = index[start_loc : end_loc + 1]

        if frequency in ("W", "ME"):
            query_frequency = "D"
        else:
            query_frequency = frequency

        api = get_api(self.api)
        data = api.get_dualkey_factor(
            factors=self.name,
            codes=codes,
            objects=self.object,
            start_time=query_start_time,
            end_time=query_end_time,
            frequency=query_frequency,
            panel=False,
        ).reset_index(level="object", drop=True)["value"]

        if frequency == "tick":
            data = data.reset_index()
            data["has_newer"] = data.duplicated(
                subset=["datetime", "code"], keep="last"
            )
            data.loc[data["has_newer"], "datetime"] = data.loc[
                data["has_newer"], "datetime"
            ] - pd.Timedelta(seconds=0.5)
            data = (
                data.drop(columns="has_newer")
                .set_index(["datetime", "code"])["value"]
                .unstack(-1)
            )
            data = data.resample(
                f"{config.tick_freq}s", closed="right", label="right"
            ).last()
        else:
            data = data.unstack(level=-1)

        return data.reindex(columns=codes, index=index)


class ListedFactor(AbstractFactor):
    """
    获取universe里的标的当天的上市状态
    """

    def __init__(self, secuinfo: pd.DataFrame):
        self.secuinfo = secuinfo

    @lru_cache(maxsize=8)
    def _compute(self, config: Config, refs: Tuple = (0, 0)):
        start_time: pd.Timestamp = config.get_option("start_time")
        end_time: pd.Timestamp = config.get_option("end_time")
        codes: Tuple[str] = config.get_option("universe")
        index: pd.DatetimeIndex = config.get_option("index")
        if start_time in index:
            start_loc = index.get_loc(start_time) - refs[0]
        else:
            start_loc = index.get_loc(index.asof(start_time)) - refs[0] + 1
        if end_time in index:
            end_loc = index.get_loc(end_time) + refs[1]
        else:
            end_loc = index.get_loc(index.asof(end_time)) + refs[1]
        index = index[start_loc : end_loc + 1]

        secuinfo = self.secuinfo.set_index("code").filter(items=codes, axis=0)
        LISTED = {}
        for date in index:
            is_listed = (secuinfo.listed_date <= date) & (secuinfo.delisted_date > date)
            LISTED[date] = is_listed
        listing_df = pd.concat(LISTED, axis=1).T.reindex(columns=codes, index=index)
        listing_df.columns.name = "code"
        return listing_df


__all__ = [
    "LeafFactor",
    "SingleLeafFactor",
    "ObjectedLeafFactor",
    "ListedFactor",
    "CombinedFactor",
    "RefFactor",
    "RollingWindowFactor",
    "UnaryCombinedFactor",
    "BinaryCombinedFactor",
    "ConstantFactor",
]
