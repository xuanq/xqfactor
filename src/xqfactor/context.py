"""因子求值上下文、叶子请求和上下文构造协议。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Literal, Protocol, Sequence, runtime_checkable
from warnings import catch_warnings, simplefilter

import pandas as pd
from pandas.tseries.frequencies import to_offset


AssetId = str | int
DateInput = str | date | datetime | pd.Timestamp
Market = Literal["cn", "hk", "us"]
Frequency = str
_NON_CANONICAL_FREQUENCY_ALIASES = {
    "d": "D",
    "1d": "D",
    "1m": "ME",
    "W": "W-SUN",
    "M": "ME",
    "T": "min",
}


def validate_frequency(frequency: Frequency) -> Frequency:
    """校验频率是否为 Pandas 规范 freqstr。

    输入：待校验的 Pandas 频率字符串。
    输出：与输入相同的规范 freqstr；别名、弃用值或非法值会抛出 ValueError。
    """
    if not isinstance(frequency, str) or not frequency:
        raise ValueError("frequency 必须是非空 Pandas freqstr")
    alias_target = _NON_CANONICAL_FREQUENCY_ALIASES.get(frequency)
    if alias_target is not None:
        raise ValueError(
            "frequency 必须使用 Pandas 规范 freqstr："
            f"{frequency!r} 应改为 {alias_target!r}"
        )
    try:
        # ************************************************************
        # Pandas 2.x 会对 M、T 等旧别名发出 FutureWarning 后返回新 freqstr；
        # 这里屏蔽警告并统一通过字符串比较给出项目自身的稳定错误信息。
        # ************************************************************
        with catch_warnings():
            simplefilter("ignore", FutureWarning)
            canonical = to_offset(frequency).freqstr
    except ValueError as error:
        raise ValueError(
            f"frequency 不是有效的 Pandas freqstr：{frequency!r}"
        ) from error
    if frequency != canonical:
        raise ValueError(
            "frequency 必须使用 Pandas 规范 freqstr："
            f"{frequency!r} 应改为 {canonical!r}"
        )
    return frequency


@dataclass(frozen=True)
class EvaluationContext:
    """一次因子求值使用的时间轴、资产池和数据语义。"""

    time_index: tuple[Any, ...]
    universe: tuple[AssetId, ...]
    frequency: Frequency
    output_start: int = 0
    output_end: int | None = None
    semantics: tuple[tuple[str, Any], ...] = ()
    provider_version: str = "default"

    def __post_init__(self) -> None:
        """标准化不可变字段并校验频率及最终输出区间。"""
        time_index = tuple(self.time_index)
        universe = tuple(self.universe)
        if not time_index:
            raise ValueError("time_index 不能为空")
        if not universe:
            raise ValueError("universe 不能为空")
        validate_frequency(self.frequency)
        output_end = len(time_index) if self.output_end is None else self.output_end
        if not 0 <= self.output_start < output_end <= len(time_index):
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

    def fingerprint(self) -> str:
        """生成包含求值范围和数据版本的上下文指纹。"""
        from xqfactor.runtime import stable_fingerprint

        return stable_fingerprint(
            {
                "time_index": self.time_index,
                "universe": self.universe,
                "frequency": self.frequency,
                "output_start": self.output_start,
                "output_end": self.output_end,
                "semantics": self.semantics,
                "provider_version": self.provider_version,
            }
        )


@dataclass(frozen=True)
class LeafRequest:
    """叶子因子的取数请求。"""

    factor_name: str
    context: EvaluationContext
    definition_version: str


@runtime_checkable
class EvaluationContextBuilder(Protocol):
    """由具体数据源实现的求值上下文构造协议。"""

    def build(
        self,
        start_date: DateInput,
        end_date: DateInput,
        universe: Sequence[AssetId],
        market: Market = "cn",
        type: str = "stock",
        frequency: Frequency = "D",
        history_period: int = 0,
        future_period: int = 0,
    ) -> EvaluationContext:
        """构造包含完整计算轴和最终输出切片的求值上下文。

        输入：日期范围、资产池、市场、资产类型、Pandas 规范频率及两侧扩展量。
        输出：可直接传给因子表达式求值的 EvaluationContext。
        """
        ...
