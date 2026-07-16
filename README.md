# xqfactor

`xqfactor` 是数据源无关的因子表达式、算子契约和执行缓存框架。

核心包不依赖 `xqdata`、RQData、Pandas 或任何具体数据存储。应用项目通过
`LeafFactor` 提供实际取数逻辑，通过 `FactorRuntime` 注入计算后端
和执行缓存。

## 基本用法

Pandas 参考后端属于可选依赖：

```bash
uv add "xqfactor[pandas]"
```

```python
import pandas as pd

from xqfactor import EvaluationContext, FactorRuntime, LeafFactor, RANK
from xqfactor.backends import PandasBackend


def load_close(request):
    """由具体应用负责从 API、数据库或本地文件读取数据。"""
    return pd.DataFrame(
        [[1.0, 2.0], [2.0, 1.0]],
        index=request.context.time_index,
        columns=request.context.universe,
    )


factor = RANK(LeafFactor("close", load_close))
context = EvaluationContext(
    time_index=("2025-01-01", "2025-01-02"),
    universe=("000001.SZ", "000002.SZ"),
    frequency="D",
)
result = factor.evaluate(context, FactorRuntime(PandasBackend()))
```

## 职责边界

- 本项目负责因子表达式图、算子定义、执行上下文、后端协议和内存执行缓存。
- 具体基础因子、在线 API、DolphinDB、Parquet、DuckDB 和全量市场数据由应用项目负责。
- 本项目的执行缓存只针对相同执行上下文复用叶子数据和中间因子，不是本地数据仓库。
- Pandas/NumPy 参考后端和统计检验实现通过可选依赖使用。

## 代码阅读路径

从 `LeafFactor` 创建叶子因子后，算子函数在 `operators.py` 中构造表达式
节点；调用 `factor.evaluate()` 进入 `FactorRuntime`，运行时先按因子定义和上下文
查询 `MemoryCache`，未命中时递归计算子节点，交给具体后端执行，最后返回输出时间区间。
