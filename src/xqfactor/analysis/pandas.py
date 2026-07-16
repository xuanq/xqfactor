"""基于 Pandas、SciPy 和 statsmodels 的可选因子检验实现。"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.api import OLS, WLS

from xqfactor.factor import AbstractFactor
from xqfactor.runtime import EvaluationContext, FactorRuntime, FactorValue


FactorInput = pd.DataFrame | FactorValue | AbstractFactor


class Processor(Protocol):
    """Pandas 因子预处理器协议。"""

    def use_runtime(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> Processor:
        """绑定计算因子时使用的上下文和运行时。"""

    def process(self, factor: pd.DataFrame) -> pd.DataFrame:
        """处理一个二维因子值并返回相同轴结构的数据。"""


class PandasProcessor:
    """需要读取其他因子时可复用的 Pandas 预处理器基类。"""

    def __init__(self) -> None:
        """创建尚未绑定运行时的预处理器。"""
        self.context: EvaluationContext | None = None
        self.runtime: FactorRuntime | None = None

    def use_runtime(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> PandasProcessor:
        """绑定执行上下文和运行时并返回自身。"""
        self.context = context
        self.runtime = runtime
        return self

    def _resolve(self, value: FactorInput) -> pd.DataFrame:
        """将因子定义或 FactorValue 转换为 Pandas DataFrame。"""
        if isinstance(value, AbstractFactor):
            if self.context is None or self.runtime is None:
                raise ValueError("处理器读取因子前必须绑定 context 和 runtime")
            return value.evaluate(self.context, self.runtime).data
        if isinstance(value, FactorValue):
            return value.data
        return value


class Winsorizer(PandasProcessor):
    """使用绝对值差中位数法进行横截面去极值。"""

    def __init__(self, n: float = 3.0) -> None:
        """创建去极值处理器；n 表示中位绝对偏差倍数。"""
        super().__init__()
        self.n = n

    def process(self, factor: pd.DataFrame) -> pd.DataFrame:
        """按每个时间截面限制因子值上下界。"""
        median = factor.median(axis=1)
        deviation = factor.sub(median, axis=0).abs().median(axis=1)
        return factor.clip(
            lower=median - self.n * deviation,
            upper=median + self.n * deviation,
            axis=0,
        )


class Normalizer(PandasProcessor):
    """执行横截面 z-score 标准化。"""

    def process(self, factor: pd.DataFrame) -> pd.DataFrame:
        """将每个时间截面的均值变为零、样本标准差变为一。"""
        return factor.sub(factor.mean(axis=1), axis=0).div(
            factor.std(axis=1, ddof=1), axis=0
        )


class Ranker(PandasProcessor):
    """执行横截面排名。"""

    def __init__(
        self, ascending: bool = True, method: str = "average", pct: bool = True
    ) -> None:
        """创建排名处理器。"""
        super().__init__()
        self.ascending = ascending
        self.method = method
        self.pct = pct

    def process(self, factor: pd.DataFrame) -> pd.DataFrame:
        """返回每个时间截面的排名。"""
        return factor.rank(
            axis=1, ascending=self.ascending, method=self.method, pct=self.pct
        )


class Filler(PandasProcessor):
    """使用固定值填充缺失值。"""

    def __init__(self, value: float = 0.0) -> None:
        """创建缺失值填充处理器。"""
        super().__init__()
        self.value = value

    def process(self, factor: pd.DataFrame) -> pd.DataFrame:
        """填充因子缺失值。"""
        return factor.fillna(self.value)


class Masker(PandasProcessor):
    """根据另一个因子生成的布尔掩码隐藏数据。"""

    def __init__(self, masked_by: FactorInput) -> None:
        """创建掩码处理器。"""
        super().__init__()
        self.masked_by = masked_by

    def process(self, factor: pd.DataFrame) -> pd.DataFrame:
        """将掩码为真的位置设置为缺失值。"""
        return factor.mask(self._resolve(self.masked_by).astype(bool))


class CSNeutralizer(PandasProcessor):
    """使用横截面 OLS 回归残差进行中性化。"""

    def __init__(
        self,
        neutralize_by: FactorInput | list[FactorInput],
        dummies: bool | list[bool] = False,
    ) -> None:
        """创建中性化处理器。"""
        super().__init__()
        self.neutralize_by = (
            neutralize_by if isinstance(neutralize_by, list) else [neutralize_by]
        )
        self.dummies = dummies

    def process(self, factor: pd.DataFrame) -> pd.DataFrame:
        """逐时间截面回归并返回残差。"""
        controls = [self._resolve(value) for value in self.neutralize_by]
        dummies = self.dummies
        if isinstance(dummies, bool):
            dummies = [dummies] * len(controls)
        if len(dummies) != len(controls):
            raise ValueError("dummies 与 neutralize_by 数量不一致")

        result = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
        for timestamp in factor.index:
            design_parts: list[pd.DataFrame] = []
            for index, (control, use_dummies) in enumerate(
                zip(controls, dummies, strict=True)
            ):
                series = control.loc[timestamp]
                if use_dummies:
                    design_parts.append(pd.get_dummies(series, dtype=float))
                else:
                    design_parts.append(series.rename(f"control_{index}").to_frame())
            design = pd.concat(design_parts, axis=1)
            valid = pd.concat(
                [factor.loc[timestamp].rename("factor"), design], axis=1
            ).dropna()
            if len(valid) <= len(design.columns):
                continue
            model = OLS(valid["factor"], valid.drop(columns="factor")).fit()
            result.loc[timestamp, model.resid.index] = model.resid
        return result


class AbstractAnalyzer:
    """支持顺序预处理的 Pandas 因子检验器基类。"""

    def __init__(
        self,
        context: EvaluationContext | None = None,
        runtime: FactorRuntime | None = None,
        keep_processed_results: bool = False,
    ) -> None:
        """创建检验器并保存可选执行环境。"""
        self.context = context
        self.runtime = runtime
        self.keep_processed_results = keep_processed_results
        self.processors: OrderedDict[str, Processor] = OrderedDict()
        self.processed_results: dict[tuple[str, str], pd.DataFrame] = {}

    def use_runtime(
        self, context: EvaluationContext, runtime: FactorRuntime
    ) -> AbstractAnalyzer:
        """绑定执行上下文和运行时。"""
        self.context = context
        self.runtime = runtime
        for processor in self.processors.values():
            processor.use_runtime(context, runtime)
        return self

    def register_processor(self, name: str, processor: Processor) -> None:
        """按顺序注册一个名称唯一的预处理器。"""
        if name in self.processors:
            raise ValueError(f"处理器名称重复: {name}")
        if self.context is not None and self.runtime is not None:
            processor.use_runtime(self.context, self.runtime)
        self.processors[name] = processor

    def _resolve(self, value: FactorInput) -> pd.DataFrame:
        """将因子定义或 FactorValue 转换为 Pandas DataFrame。"""
        if isinstance(value, AbstractFactor):
            if self.context is None or self.runtime is None:
                raise ValueError("检验器读取因子前必须绑定 context 和 runtime")
            return value.evaluate(self.context, self.runtime).data
        if isinstance(value, FactorValue):
            return value.data
        return value

    def process(self, factor_name: str, factor: FactorInput) -> pd.DataFrame:
        """依次执行当前检验器注册的预处理器。"""
        result = self._resolve(factor)
        for processor_name, processor in self.processors.items():
            result = processor.process(result)
            if self.keep_processed_results:
                self.processed_results[(factor_name, processor_name)] = result
        return result

    def _analyze(self, factors: Mapping[str, pd.DataFrame]) -> Any:
        """由自定义检验器实现最终统计逻辑。"""
        raise NotImplementedError

    def analyze(self, factors: Mapping[str, FactorInput]) -> Any:
        """预处理所有输入因子后执行检验。"""
        processed = {
            name: self.process(name, factor) for name, factor in factors.items()
        }
        return self._analyze(processed)


class DataFetcher(AbstractAnalyzer):
    """只负责将因子定义计算为 Pandas DataFrame。"""

    def fetch(self, factor: FactorInput) -> pd.DataFrame:
        """计算或提取一个因子值。"""
        return self._resolve(factor)


class ICAnalysisResult:
    """IC 序列及常用统计结果。"""

    def __init__(self, ic: pd.DataFrame) -> None:
        """保存各因子的逐期 IC。"""
        self._ic = ic

    @property
    def data(self) -> pd.DataFrame:
        """返回逐期 IC DataFrame。"""
        return self._ic

    def ic(self, factor: str | list[str] | None = None) -> pd.DataFrame:
        """返回全部或指定因子的 IC。"""
        if factor is None:
            return self._ic
        columns = [factor] if isinstance(factor, str) else factor
        return self._ic[columns]

    def summary(self, factor: str | list[str] | None = None) -> pd.DataFrame:
        """返回 IC 均值、标准差、IR 和方向稳定性。"""
        ic = self.ic(factor)
        return pd.DataFrame(
            {
                "ic_mean": ic.mean(),
                "ic_std": ic.std(ddof=1),
                "ir": ic.mean().abs() / ic.std(ddof=1),
                "gt_zero_ratio": (ic > 0).mean(),
                "abs_gt0.02_ratio": (ic.abs() > 0.02).mean(),
            }
        )


class ICAnalyzer(AbstractAnalyzer):
    """计算因子与给定收益率之间的逐期横截面相关系数。"""

    def __init__(
        self,
        returns: FactorInput,
        context: EvaluationContext | None = None,
        runtime: FactorRuntime | None = None,
        keep_processed_results: bool = False,
    ) -> None:
        """创建 IC 检验器。"""
        super().__init__(context, runtime, keep_processed_results)
        self.returns = returns

    def _analyze(self, factors: Mapping[str, pd.DataFrame]) -> ICAnalysisResult:
        """计算每个输入因子的逐期 IC。"""
        returns = self._resolve(self.returns)
        values = {
            name: factor.corrwith(returns, axis=1) for name, factor in factors.items()
        }
        return ICAnalysisResult(pd.DataFrame(values))


class QuantileReturnResult:
    """分组收益分析结果。"""

    def __init__(self, quantile_returns: pd.DataFrame) -> None:
        """保存因子分组收益。"""
        self._quantile_returns = quantile_returns

    @property
    def data(self) -> pd.DataFrame:
        """返回分组收益 DataFrame。"""
        return self._quantile_returns

    def long_short(self) -> pd.DataFrame:
        """返回各因子最高组减最低组的收益。"""
        results: list[pd.Series] = []
        for factor in self._quantile_returns.columns.get_level_values(0).unique():
            frame = self._quantile_returns[factor]
            series = frame[frame.columns.max()] - frame[frame.columns.min()]
            series.name = factor
            results.append(series)
        return pd.concat(results, axis=1)


class QuantileReturnAnalyzer(AbstractAnalyzer):
    """计算因子的横截面分组收益。"""

    def __init__(
        self,
        returns: FactorInput,
        n_groups: int = 5,
        context: EvaluationContext | None = None,
        runtime: FactorRuntime | None = None,
    ) -> None:
        """创建分组收益检验器。"""
        super().__init__(context, runtime)
        self.returns = returns
        self.n_groups = n_groups

    def _analyze(self, factors: Mapping[str, pd.DataFrame]) -> QuantileReturnResult:
        """计算每个因子的分组平均收益。"""
        returns = self._resolve(self.returns)
        outputs: list[pd.DataFrame] = []
        for name, factor in factors.items():
            groups = factor.apply(
                lambda row: pd.qcut(
                    row,
                    self.n_groups,
                    labels=range(1, self.n_groups + 1),
                    duplicates="drop",
                ),
                axis=1,
            )
            values: dict[int, pd.Series] = {}
            for group in range(1, self.n_groups + 1):
                values[group] = returns.where(groups == group).mean(axis=1)
            output = pd.DataFrame(values)
            output.columns = pd.MultiIndex.from_product(
                [[name], output.columns], names=["factor", "quantile"]
            )
            outputs.append(output)
        return QuantileReturnResult(pd.concat(outputs, axis=1))


class RegressionAnalysisResult:
    """逐期横截面回归结果。"""

    def __init__(self, results: pd.DataFrame) -> None:
        """保存各期拟合结果对象。"""
        self._results = results

    @property
    def data(self) -> pd.DataFrame:
        """返回原始回归结果对象。"""
        return self._results

    def coefficients(self) -> pd.DataFrame:
        """返回各因子的逐期回归系数。"""
        return self._results.map(lambda result: result.params.loc["factor"])

    def summary(self) -> pd.DataFrame:
        """返回平均因子收益和跨期 t 检验结果。"""
        coefficients = self.coefficients()
        test = ttest_1samp(coefficients, 0, nan_policy="omit")
        return pd.DataFrame(
            {
                "return_mean": coefficients.mean(),
                "t_stat": test.statistic,
                "p_value": test.pvalue,
            },
            index=coefficients.columns,
        )


class RegressionAnalyzer(AbstractAnalyzer):
    """执行逐期横截面 OLS 或 WLS 因子收益回归。"""

    def __init__(
        self,
        returns: FactorInput,
        weights: FactorInput | None = None,
        model: str = "WLS",
        context: EvaluationContext | None = None,
        runtime: FactorRuntime | None = None,
    ) -> None:
        """创建横截面回归检验器。"""
        super().__init__(context, runtime)
        self.returns = returns
        self.weights = weights
        self.model = model.upper()

    def _analyze(self, factors: Mapping[str, pd.DataFrame]) -> RegressionAnalysisResult:
        """分别回归每个输入因子并返回逐期拟合对象。"""
        returns = self._resolve(self.returns)
        weights = self._resolve(self.weights) if self.weights is not None else None
        outputs: dict[str, pd.Series] = {}
        for name, factor in factors.items():
            fitted: dict[Any, Any] = {}
            for timestamp in factor.index:
                frame = pd.DataFrame(
                    {
                        "factor": factor.loc[timestamp],
                        "returns": returns.loc[timestamp],
                    }
                )
                if weights is not None:
                    frame["weights"] = weights.loc[timestamp]
                frame = frame.dropna()
                if frame.empty:
                    continue
                if self.model == "OLS":
                    result = OLS(frame["returns"], frame[["factor"]]).fit()
                elif self.model == "WLS":
                    weight_values = (
                        frame["weights"] if "weights" in frame else np.ones(len(frame))
                    )
                    result = WLS(
                        frame["returns"], frame[["factor"]], weights=weight_values
                    ).fit()
                else:
                    raise ValueError(f"不支持的回归模型: {self.model}")
                fitted[timestamp] = result
            outputs[name] = pd.Series(fitted)
        return RegressionAnalysisResult(pd.DataFrame(outputs))


__all__ = [
    "AbstractAnalyzer",
    "CSNeutralizer",
    "DataFetcher",
    "Filler",
    "ICAnalysisResult",
    "ICAnalyzer",
    "Masker",
    "Normalizer",
    "PandasProcessor",
    "QuantileReturnAnalyzer",
    "QuantileReturnResult",
    "Ranker",
    "RegressionAnalysisResult",
    "RegressionAnalyzer",
    "Winsorizer",
]
