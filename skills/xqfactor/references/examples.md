# xqfactor 使用示例

## 目录

- [依赖与初始化](#依赖与初始化)
- [使用 RQData 定义后复权 CLOSE](#使用-rqdata-定义后复权-close)
- [组合 RETURNS 因子并执行](#组合-returns-因子并执行)
- [使用 Polars 自定义算子](#使用-polars-自定义算子)
- [使用 PyTorch 自定义算子](#使用-pytorch-自定义算子)
- [先标准化再进行 IC 检验](#先标准化再进行-ic-检验)
- [自定义检验器](#自定义检验器)
- [窗口与缓存注意事项](#窗口与缓存注意事项)

## 依赖与初始化

在使用 xqfactor 的应用项目中安装依赖：

```bash
uv add "xqfactor[pandas,analysis]"
uv add rqdatac
uv add polars
uv add torch
```

`rqdatac`、Polars 和 PyTorch 都是应用依赖，不要加入 xqfactor 核心依赖。

```python
import pandas as pd
import rqdatac

from xqfactor import EvaluationContext, FactorRuntime, MemoryCache
from xqfactor.backends import PandasBackend


rqdatac.init()

runtime = FactorRuntime(
    backend=PandasBackend(),
    cache=MemoryCache(maxsize=256),
)
```

## 使用 RQData 定义后复权 CLOSE

RQData 的 `get_price` 使用 `adjust_type="post"` 获取股票和 ETF 的后复权行情。
`expect_df=True` 时返回 Pandas DataFrame；多标的日线通常使用
`(order_book_id, date)` MultiIndex。

```python
from typing import Any

import pandas as pd
import rqdatac

from xqfactor import LeafFactor
from xqfactor.runtime import LeafRequest


def _to_rq_frequency(frequency: str) -> str:
    """将应用频率转换为 rqdatac.get_price 支持的频率。

    输入：xqfactor EvaluationContext 中的频率字符串。
    输出：RQData 使用的频率字符串。
    """
    return {
        "D": "1d",
        "W": "1w",
        "min": "1m",
    }.get(frequency, frequency)


def load_close(request: LeafRequest) -> pd.DataFrame:
    """读取后复权收盘价并转换为时间乘资产的二维表。

    输入：包含完整计算时间轴、universe 和频率的叶子请求。
    输出：index 为完整时间轴、columns 为 universe 的 DataFrame。
    """
    context = request.context
    semantics: dict[str, Any] = dict(context.semantics)
    data = rqdatac.get_price(
        order_book_ids=list(context.universe),
        start_date=context.time_index[0],
        end_date=context.time_index[-1],
        frequency=_to_rq_frequency(context.frequency),
        fields="close",
        adjust_type="post",
        skip_suspended=False,
        expect_df=True,
        market=semantics.get("market", "cn"),
    )

    if data is None or data.empty:
        return pd.DataFrame(
            index=pd.Index(context.time_index, name="datetime"),
            columns=context.universe,
            dtype=float,
        )

    # ************************************************************
    # RQData 结果从 MultiIndex Series：
    # (order_book_id, date/datetime) -> value
    # 转换为二维 DataFrame：
    # index=date/datetime，columns=order_book_id。
    # ************************************************************
    close = data["close"].unstack("order_book_id")
    close.index = pd.to_datetime(close.index)
    return close.reindex(
        index=pd.DatetimeIndex(context.time_index),
        columns=context.universe,
    )


CLOSE = LeafFactor(
    name="close",
    resolver=load_close,
    definition_version="rqdata-post-v1",
)
```

如果使用港股，把上下文语义设置为 `(("market", "hk"),)`；resolver 会把它传给
`rqdatac.get_price`。

## 组合 RETURNS 因子并执行

`PCT_CHANGE(X, n)` 的语义是 `X / REF(X, n) - 1`。

```python
import pandas as pd

from xqfactor import EvaluationContext, PCT_CHANGE


RETURNS = PCT_CHANGE(CLOSE, 1)

# time_index 包含 2025-01-02 这一期历史值。
# output_start=1 表示最终结果从 2025-01-03 开始返回。
context = EvaluationContext(
    time_index=tuple(
        pd.to_datetime(
            [
                "2025-01-02",
                "2025-01-03",
                "2025-01-06",
                "2025-01-07",
            ]
        )
    ),
    universe=("000001.XSHE", "600000.XSHG"),
    frequency="D",
    output_start=1,
    semantics=(("market", "cn"), ("adjust_type", "post")),
    provider_version="rqdata-2025-07",
)

returns_value = RETURNS.evaluate(context, runtime)
returns_df = returns_value.data
```

在同一个 `runtime` 中再次计算 `CLOSE`、`RETURNS` 或依赖它们的其他表达式时，可以
复用已经读取的叶子值和中间结果。

## 使用 Polars 自定义算子

当前参考执行后端是 `PandasBackend`，因此自定义函数接收 Pandas DataFrame。可以在
函数内部转换为 Polars，计算后再恢复相同的时间轴和资产轴。

```python
import pandas as pd
import polars as pl

from xqfactor import custom_unary


def polars_log1p(frame: pd.DataFrame) -> pd.DataFrame:
    """使用 Polars 计算 log(1 + x)。

    输入：形状为 (时间数, 资产数) 的 Pandas DataFrame。
    输出：形状、index 和 columns 与输入一致的 Pandas DataFrame。
    """
    index_name = frame.index.name or "datetime"

    # ************************************************************
    # 数据形状保持 (时间数, 资产数) 不变。
    # reset_index 临时把时间 index 变为普通列，计算后再恢复。
    # ************************************************************
    polars_frame = pl.from_pandas(frame.rename_axis(index_name).reset_index())
    transformed = polars_frame.with_columns(
        (pl.exclude(index_name) + 1).log()
    )
    return transformed.to_pandas().set_index(index_name).reindex_like(frame)


POLARS_LOG_CLOSE = custom_unary(
    CLOSE,
    polars_log1p,
    name="polars_log1p",
)

polars_result = POLARS_LOG_CLOSE.evaluate(context, runtime).data
```

## 使用 PyTorch 自定义算子

```python
import pandas as pd
import torch

from xqfactor import custom_unary


def torch_sigmoid(frame: pd.DataFrame) -> pd.DataFrame:
    """使用 PyTorch 对因子值执行 sigmoid。

    输入：形状为 (时间数, 资产数) 的 Pandas DataFrame。
    输出：形状、index 和 columns 与输入一致的 Pandas DataFrame。
    """
    # ************************************************************
    # DataFrame (T, N) -> Tensor (T, N) -> DataFrame (T, N)。
    # 时间轴和资产轴在转换前后保持不变。
    # ************************************************************
    tensor = torch.as_tensor(frame.to_numpy(), dtype=torch.float64)
    transformed = torch.sigmoid(tensor).numpy()
    return pd.DataFrame(
        transformed,
        index=frame.index,
        columns=frame.columns,
    )


TORCH_SIGMOID_CLOSE = custom_unary(
    CLOSE,
    torch_sigmoid,
    name="torch_sigmoid",
)

torch_result = TORCH_SIGMOID_CLOSE.evaluate(context, runtime).data
```

如果需要真正以 Polars 或 PyTorch 对象贯穿整个表达式图，应实现新的
`ComputeBackend`，而不是在每个算子中反复转换。

## 先标准化再进行 IC 检验

下面先计算下一期收益，再对被检验因子执行横截面标准化，最后计算逐期 IC。

```python
from xqfactor.analysis.pandas import ICAnalyzer, Normalizer


raw_returns = RETURNS.evaluate(context, runtime).data
forward_returns = raw_returns.shift(-1)

analyzer = ICAnalyzer(
    returns=forward_returns,
    context=context,
    runtime=runtime,
    keep_processed_results=True,
)
analyzer.register_processor("normalization", Normalizer())

ic_result = analyzer.analyze({"returns": RETURNS})
ic_series = ic_result.data
ic_summary = ic_result.summary()

normalized_returns = analyzer.processed_results[
    ("returns", "normalization")
]
```

处理器按注册顺序运行。需要先去极值再标准化时：

```python
from xqfactor.analysis.pandas import Normalizer, Winsorizer


analyzer.register_processor("winsorization", Winsorizer(n=3.0))
analyzer.register_processor("normalization", Normalizer())
```

## 自定义检验器

继承 `AbstractAnalyzer` 并实现 `_analyze`。传入 `_analyze` 的因子已经完成求值和所有
预处理，因此类型是 `dict[str, pd.DataFrame]`。

```python
from typing import Mapping

import pandas as pd

from xqfactor.analysis.pandas import AbstractAnalyzer, Normalizer


class CoverageAnalyzer(AbstractAnalyzer):
    """统计每个因子的非缺失数据覆盖率。"""

    def _analyze(
        self,
        factors: Mapping[str, pd.DataFrame],
    ) -> pd.Series:
        """计算全部时间和资产上的非缺失比例。

        输入：名称到二维因子值的映射。
        输出：index 为因子名、value 为覆盖率的 Series。
        """
        return pd.Series(
            {
                name: factor.notna().to_numpy().mean()
                for name, factor in factors.items()
            },
            name="coverage",
        )


coverage_analyzer = CoverageAnalyzer(context=context, runtime=runtime)
coverage_analyzer.register_processor("normalization", Normalizer())
coverage = coverage_analyzer.analyze(
    {
        "returns": RETURNS,
        "polars_log_close": POLARS_LOG_CLOSE,
    }
)
```

自定义检验器可以直接返回 Series、DataFrame、dataclass 或其他业务结果对象；如果希望
与通用 `AnalysisResult` 协议兼容，结果对象应提供 `data` 属性。

## 窗口与缓存注意事项

自定义窗口算子：

```python
import pandas as pd

from xqfactor import rolling_operator


def rolling_mean(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """计算时间序列移动平均。"""
    return frame.rolling(window).mean()


MA20 = rolling_operator(
    CLOSE,
    window=20,
    function=rolling_mean,
    name="ma20",
)
```

构造上下文时至少提供 19 个额外历史周期，并把 `output_start` 设置到目标输出起点。
可以通过 `MA20.required_history()` 查询表达式需要的历史周期数。

缓存键包含：

- 因子定义和 resolver 版本；
- 完整时间轴与输出切片；
- universe 和 frequency；
- semantics；
- provider 版本；
- backend 版本。

不同 universe、频率或时间区间不会共享缓存。长期全市场数据存储不属于 xqfactor。
