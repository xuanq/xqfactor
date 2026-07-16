"""基于 Pandas/NumPy 的参考计算后端。

该模块是可选后端，不被 xqfactor 核心模块自动导入。安装 Pandas extra 后，
可通过 ``FactorRuntime(PandasBackend())`` 执行因子表达式。
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from xqfactor.runtime import EvaluationContext, FactorValue, OperatorSpec


class PandasBackend:
    """使用 Pandas DataFrame 表达二维因子值的参考后端。"""

    name = "pandas"
    version = "1"

    def _frame(self, value: FactorValue, context: EvaluationContext) -> pd.DataFrame:
        """将 FactorValue 转换为完整计算轴上的 DataFrame。"""
        data = value.data
        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, pd.Series):
            if len(data) == len(context.time_index):
                frame = pd.DataFrame(
                    {asset: data.to_numpy() for asset in context.universe},
                    index=context.time_index,
                )
            elif len(data) == len(context.universe):
                frame = pd.DataFrame(
                    np.broadcast_to(
                        data.to_numpy(), (len(context.time_index), len(data))
                    ),
                    index=context.time_index,
                    columns=context.universe,
                )
            else:
                raise ValueError("Series 长度无法映射到时间轴或资产轴")
        else:
            array = np.asarray(data)
            if array.ndim == 0:
                array = np.full(
                    (len(context.time_index), len(context.universe)), array.item()
                )
            elif array.ndim == 1:
                if len(array) == len(context.time_index):
                    array = np.repeat(array[:, None], len(context.universe), axis=1)
                elif len(array) == len(context.universe):
                    array = np.repeat(array[None, :], len(context.time_index), axis=0)
                else:
                    raise ValueError("一维数组长度无法映射到时间轴或资产轴")
            if array.ndim != 2:
                raise ValueError("因子值必须是二维数据")
            frame = pd.DataFrame(
                array, index=context.time_index, columns=context.universe
            )
        return frame.reindex(index=context.time_index, columns=context.universe)

    def normalize(self, value: Any, context: EvaluationContext) -> FactorValue:
        """将 DataFrame、Series、数组或标量标准化为 FactorValue。

        输入：后端原始结果或已经包装的 FactorValue。
        输出：时间轴为 context.time_index、列为 context.universe 的二维因子值。
        """
        if isinstance(value, FactorValue):
            frame = self._frame(value, context)
        else:
            frame = self._frame(
                FactorValue(value, context.time_index, context.universe), context
            )
        return FactorValue(frame, context.time_index, context.universe)

    def constant(self, value: Any, context: EvaluationContext) -> FactorValue:
        """将标量广播为完整二维 DataFrame。"""
        frame = pd.DataFrame(
            value,
            index=context.time_index,
            columns=context.universe,
        )
        return FactorValue(frame, context.time_index, context.universe)

    def _apply_builtin(
        self,
        spec: OperatorSpec,
        frames: Sequence[pd.DataFrame],
    ) -> Any:
        """执行参考后端内置算子。"""
        name = spec.name
        if name in {
            "add",
            "subtract",
            "multiply",
            "true_divide",
            "power",
            "floor_divide",
            "mod",
            "greater",
            "less",
            "greater_equal",
            "less_equal",
            "logical_and",
            "logical_or",
            "not_equal",
            "equal",
        }:
            operation = {
                "add": np.add,
                "subtract": np.subtract,
                "multiply": np.multiply,
                "true_divide": np.true_divide,
                "power": np.power,
                "floor_divide": np.floor_divide,
                "mod": np.mod,
                "greater": np.greater,
                "less": np.less,
                "greater_equal": np.greater_equal,
                "less_equal": np.less_equal,
                "logical_and": np.logical_and,
                "logical_or": np.logical_or,
                "not_equal": np.not_equal,
                "equal": np.equal,
            }[name]
            return operation(frames[0], frames[1])
        if name in {"abs", "log", "exp", "sign", "logical_not"}:
            return {
                "abs": np.abs,
                "log": np.log,
                "exp": np.exp,
                "sign": np.sign,
                "logical_not": np.logical_not,
            }[name](frames[0])
        if name == "signed_power":
            exponent = spec.args[0]
            return np.sign(frames[0]) * np.power(np.abs(frames[0]), exponent)
        if name in {"minimum", "maximum", "fminimum", "fmaximum"}:
            operation = {
                "minimum": np.minimum,
                "maximum": np.maximum,
                "fminimum": np.fmin,
                "fmaximum": np.fmax,
            }[name]
            return operation(frames[0], frames[1])
        if name == "where":
            return np.where(frames[0].astype(bool), frames[1], frames[2])
        if name == "as_float":
            return frames[0].astype(float)
        if name == "notna":
            return frames[0].notna()
        if name == "pct_change":
            periods = spec.args[0]
            return frames[0].div(frames[0].shift(periods)) - 1.0
        if name == "mad":
            median = frames[0].median(axis=1)
            deviation = frames[0].sub(median, axis=0).abs().median(axis=1)
            return frames[0].clip(
                lower=median - spec.args[0] * deviation,
                upper=median + spec.args[0] * deviation,
                axis=0,
            )
        if name == "norm":
            return (
                frames[0]
                .sub(frames[0].mean(axis=1), axis=0)
                .div(frames[0].std(axis=1, ddof=1), axis=0)
            )
        if name == "rank":
            method, ascending, pct = spec.args
            return frames[0].rank(axis=1, method=method, ascending=ascending, pct=pct)
        if name == "proportion":
            return frames[0].div(frames[0].sum(axis=1), axis=0)
        if name == "cumprod":
            return frames[0].cumprod()
        if name == "ffill":
            return frames[0].ffill()
        if name == "fillna":
            return frames[0].where(frames[0].notna(), frames[1])
        if name == "mask":
            return frames[0].mask(frames[1].astype(bool))
        if name == "minmax_scaler":
            minimum = frames[0].min(axis=1)
            maximum = frames[0].max(axis=1)
            return frames[0].sub(minimum, axis=0).div(maximum - minimum, axis=0)
        if name == "quantile":
            groups = spec.args[0]
            return frames[0].apply(
                lambda row: pd.qcut(
                    row, groups, labels=range(1, groups + 1), duplicates="drop"
                ),
                axis=1,
            )
        if name == "binary_label":
            top_pct, bottom_pct = spec.args
            top = frames[0].ge(frames[0].quantile(1 - top_pct, axis=1), axis=0)
            bottom = frames[0].le(frames[0].quantile(bottom_pct, axis=1), axis=0)
            result = pd.DataFrame(
                np.nan, index=frames[0].index, columns=frames[0].columns
            )
            result = result.mask(top, 1).mask(bottom, 0)
            return result
        if name == "group_quantile":
            groups = spec.args[0]
            factor, grouper = frames
            output = pd.DataFrame(np.nan, index=factor.index, columns=factor.columns)
            for timestamp in factor.index:
                values = factor.loc[timestamp]
                labels = grouper.loc[timestamp]
                for _, assets in labels.groupby(labels).groups.items():
                    selected = values.loc[assets].dropna()
                    if len(selected) >= groups:
                        output.loc[timestamp, selected.index] = pd.qcut(
                            selected,
                            groups,
                            labels=range(1, groups + 1),
                            duplicates="drop",
                        )
            return output
        if name == "cs_group":
            function = spec.function
            if function is None:
                raise ValueError("cs_group 必须提供 function")
            result = (
                frames[0]
                .stack(dropna=False)
                .groupby(
                    [
                        frames[0].stack(dropna=False).index.get_level_values(0),
                        frames[1].stack(dropna=False),
                    ]
                )
                .transform(function, *spec.args[0])
            )
            return result.unstack()
        if name == "cs_neutralizer":
            return self._neutralize(frames, spec)
        raise KeyError(f"PandasBackend 不支持算子: {name}")

    def apply(
        self,
        spec: OperatorSpec,
        inputs: Sequence[FactorValue],
        context: EvaluationContext,
    ) -> FactorValue:
        """执行内置或用户自定义算子并标准化结果。"""
        frames = tuple(self._frame(value, context) for value in inputs)
        if spec.function is not None and spec.name not in {
            "cs_group",
            "custom_unary",
            "custom_binary",
            "custom_combined",
        }:
            result = spec.function(*frames, *spec.args, **dict(spec.kwargs))
        elif spec.name in {
            "custom_unary",
            "custom_binary",
            "custom_combined",
            "cs_group",
        }:
            if spec.function is None:
                raise ValueError(f"{spec.name} 缺少 function")
            result = spec.function(*frames, *spec.args, **dict(spec.kwargs))
        else:
            result = self._apply_builtin(spec, frames)
        return self.normalize(result, context)

    def _neutralize(
        self, frames: Sequence[pd.DataFrame], spec: OperatorSpec
    ) -> pd.DataFrame:
        """按横截面回归控制变量并返回残差。"""
        controls = list(frames[1:])
        dummies = spec.args[0]
        model = dict(spec.kwargs).get("model", "OLS")
        if isinstance(dummies, bool):
            dummies = [dummies] * len(controls)
        if len(dummies) != len(controls):
            raise ValueError("dummies 与 neutralize_by 数量不一致")
        result = pd.DataFrame(np.nan, index=frames[0].index, columns=frames[0].columns)
        for timestamp in frames[0].index:
            y = frames[0].loc[timestamp]
            design_parts = []
            for control, use_dummies in zip(controls, dummies):
                series = control.loc[timestamp]
                if use_dummies:
                    design_parts.append(pd.get_dummies(series, dtype=float))
                else:
                    design_parts.append(series.rename("control").to_frame())
            design = pd.concat(design_parts, axis=1)
            valid = pd.concat([y.rename("y"), design], axis=1).dropna()
            if len(valid) <= len(design.columns):
                continue
            if model.upper() != "OLS":
                raise ValueError(f"PandasBackend 仅支持 OLS，中性化收到 {model}")
            x = valid.drop(columns="y").to_numpy(dtype=float)
            coefficients, *_ = np.linalg.lstsq(
                x, valid["y"].to_numpy(dtype=float), rcond=None
            )
            residual = valid["y"] - x @ coefficients
            result.loc[timestamp, residual.index] = residual
        return result

    def shift(
        self, value: FactorValue, periods: int, context: EvaluationContext
    ) -> FactorValue:
        """沿时间轴移动 DataFrame，缺失位置填充为 NaN。"""
        return self.normalize(self._frame(value, context).shift(periods), context)

    def rolling(
        self,
        spec: OperatorSpec,
        value: FactorValue,
        window: int,
        context: EvaluationContext,
    ) -> FactorValue:
        """执行自定义窗口函数；函数输入为 DataFrame 和窗口长度。"""
        frame = self._frame(value, context)
        if spec.function is None:
            raise ValueError("窗口算子必须提供 function")
        result = spec.function(frame, window, *spec.args, **dict(spec.kwargs))
        return self.normalize(result, context)

    def slice(
        self,
        value: FactorValue,
        start: int,
        end: int,
        context: EvaluationContext,
    ) -> FactorValue:
        """截取 DataFrame 的输出区间并同步更新时间轴。"""
        frame = self._frame(value, context).iloc[start:end]
        return FactorValue(frame, context.time_index[start:end], context.universe)
