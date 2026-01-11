import math
from functools import lru_cache
from typing import Dict, Literal, Tuple

import pandas as pd
from xqdata.dataapi import DataApi


class Api:
    def __init__(self):
        self._api: Dict[str, DataApi] = {}

    def add_api(self, api: DataApi, name="default"):
        self._api[name] = api

    def get_api(self, name="default"):
        return self._api[name]

    def remove_api(self, name="default"):
        self._api.pop(name)


global_api = Api()


def add_api(api: DataApi, name="default"):
    global_api.add_api(api, name)


def remove_api(name="default"):
    global_api.remove_api(name)


def get_api(name="default"):
    return global_api.get_api(name)


class Config:
    def __init__(self):
        self.start_time: pd.Timestamp = None
        self.end_time: pd.Timestamp = None
        self.api: str = "default"
        self.frequency = "D"
        self.calendar_type: Literal["tradeday_cn", "natural"] = "tradeday_cn"
        self.trading_minutes = [("09:30:00", "11:30:00"), ("13:00:00", "15:00:00")]
        self.exclude = None
        self.tick_freq = 3  # 股票每3秒一个tick,期货每0.5秒一个tick
        self.offset = 60
        self.universe: Tuple[str] = None
        self._index_inited = False

    @property
    def trading_minute_1d(self):
        """根据trading_minutes计算每天交易时间"""
        return sum(
            [
                (
                    (
                        pd.to_datetime(period[1]) - pd.to_datetime(period[0])
                    ).total_seconds()
                    / 60
                )
                % 1440
                for period in self.trading_minutes
            ]
        )

    @lru_cache(maxsize=1)
    def calendar(self):
        if self.calendar_type == "tradeday_cn":
            # 获取交易日
            api = get_api(self.api)
            calendar = api.get_info("tradedays")
            non_tradedays = calendar[~calendar.is_tradeday].index
            ashare_holiday = non_tradedays[~non_tradedays.weekday.isin([5, 6])]
            return pd.offsets.CustomBusinessDay(holidays=ashare_holiday)
        elif self.calendar_type == "natural":
            return pd.offsets.Day(1)

    @lru_cache(maxsize=1)
    def index(self):
        start_date = self.start_time.normalize()
        end_date = self.end_time.normalize()
        if self.frequency == "tick":
            tick_in_1min = 60 / self.tick_freq
            days_ref = math.ceil(self.offset / (self.trading_minute_1d * tick_in_1min))
            true_start_date = start_date - self.calendar() * days_ref
            true_end_date = end_date + self.calendar() * days_ref
            trade_days = pd.date_range(
                true_start_date, true_end_date, freq=self.calendar()
            )
            trading_tick_list = []
            for day in trade_days:
                for time in self.trading_minutes:
                    trading_tick_list.append(
                        pd.Series(
                            pd.date_range(
                                start=day + pd.Timedelta(time[0]),
                                end=day + pd.Timedelta(time[1]),
                                freq=f"{self.tick_freq}s",
                                inclusive="right",
                            )
                        )
                    )
            index = pd.DatetimeIndex(pd.concat(trading_tick_list))

        elif self.frequency == "min":
            days_ref = math.ceil(self.offset / self.trading_minute_1d)
            true_start_date = start_date - self.calendar() * days_ref
            true_end_date = end_date + self.calendar() * days_ref
            trade_days = pd.date_range(
                true_start_date, true_end_date, freq=self.calendar()
            )

            trading_min_list = []
            for day in trade_days:
                for time in self.trading_minutes:
                    trading_min_list.append(
                        pd.Series(
                            pd.date_range(
                                start=day + pd.Timedelta(time[0]),
                                end=day + pd.Timedelta(time[1]),
                                freq="min",
                                inclusive="right",
                            )
                        )
                    )
            index = pd.DatetimeIndex(pd.concat(trading_min_list))
        elif self.frequency == "D":
            true_start_date = start_date - self.calendar() * self.offset
            true_end_date = end_date + self.calendar() * self.offset
            index = pd.date_range(true_start_date, true_end_date, freq=self.calendar())
        elif self.frequency == "W":
            true_start_date = start_date - self.offset * self.calendar() * 5
            true_end_date = end_date + self.offset * self.calendar() * 5
            daily_index = pd.date_range(
                true_start_date, true_end_date, freq=self.calendar()
            )
            week_ends = (
                pd.Series(index=daily_index, data=daily_index)
                .groupby(pd.Grouper(freq="W"))
                .last()
                .dropna()
            )
            index = pd.DatetimeIndex(week_ends)
        elif self.frequency == "ME":
            true_start_date = start_date - self.offset * self.calendar() * 23
            true_end_date = end_date + self.offset * self.calendar() * 23
            daily_index = pd.date_range(
                true_start_date, true_end_date, freq=self.calendar()
            )
            month_ends = (
                pd.Series(index=daily_index, data=daily_index)
                .groupby(pd.Grouper(freq="ME"))
                .last()
                .dropna()
            )
            index = pd.DatetimeIndex(month_ends)

        if self.exclude is not None:
            index = index[~index.isin(self.exclude)]

        self._index_inited = True
        index.name = "datetime"
        return index

    def set_option(self, option_name, value):
        if hasattr(self, option_name):
            if option_name in ("start_time", "end_time"):
                value = pd.Timestamp(value)
            if option_name == "universe":
                if isinstance(value, str):
                    value = (value,)
                value = tuple(value)
            setattr(self, option_name, value)
            if self._index_inited:
                self.index.cache_clear()
        else:
            raise ValueError(f"{option_name} is not a valid option.")

    def get_option(self, option_name):
        if hasattr(self, option_name):
            if callable(getattr(self, option_name)):
                return getattr(self, option_name)()
            else:
                return getattr(self, option_name)
        else:
            raise ValueError(f"{option_name} is not a valid option.")

    def __hash__(self):
        return hash(
            (
                self.start_time,
                self.end_time,
                self.frequency,
                self.universe,
                self.calendar_type,
                self.offset,
            )
        )

    def __eq__(self, other):
        if isinstance(other, Config):
            return hash(self) == hash(other)


# 创建一个全局配置实例
global_config = Config()


def set_option(option_name, value):
    global_config.set_option(option_name, value)


def get_option(option_name):
    return global_config.get_option(option_name)
