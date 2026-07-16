---
name: xqfactor
description: Use the xqfactor Python project to define data-source-independent factors, compose built-in or custom operators, execute factor graphs with explicit contexts and caches, add Pandas/Polars/PyTorch calculations, and build IC, quantile-return, regression, preprocessing, or custom factor analyses. Trigger when an agent needs to integrate xqfactor into a research project, create LeafFactor resolvers for RQData or another source, extend a compute backend, diagnose factor evaluation or caching, or write and test factor-analysis workflows.
---

# xqfactor

## Goal

Use this repository as a data-source-independent factor expression and analysis framework. Keep
data acquisition in application-owned `LeafFactor` resolvers; keep expression execution, operator
semantics, axis handling and execution caching in xqfactor.

## Start Here

1. Read the repository `AGENTS.md`.
2. Inspect `src/xqfactor/__init__.py` before assuming an API is public.
3. Read `references/examples.md` for complete runnable patterns.
4. Install only the extras needed by the consuming project:

```bash
uv add "xqfactor[pandas]"
uv add "xqfactor[analysis]"
```

Add application-owned dependencies such as `rqdatac`, `polars` or `torch` in the consuming
project, not to xqfactor core.

## Core Model

Build workflows from these objects:

- `LeafFactor(name, resolver, definition_version="1")`: define application-owned data loading.
- `EvaluationContext`: declare the complete calculation time axis, universe, frequency, output
  slice, semantic options and provider version.
- `FactorRuntime(backend, cache)`: bind a compute backend and execution cache.
- `FactorValue`: carry backend data plus explicit time and asset axes.
- `PandasBackend`: execute built-in and custom operators with Pandas/NumPy.
- `MemoryCache`: reuse leaf and intermediate values only for an identical factor/context/backend
  identity.

Treat `time_index` as the full calculation axis. Include all history needed by `REF`, rolling
operators or return calculations. Use `output_start`/`output_end` to remove the history rows from
the final result.

## Define Leaf Factors

Pass all external I/O through the resolver:

```python
CLOSE = LeafFactor("close", load_close, definition_version="rqdata-post-v1")
```

Make the resolver return a two-dimensional value whose rows follow
`request.context.time_index` and whose columns follow `request.context.universe`. Reindex before
returning. Change `definition_version` or `EvaluationContext.provider_version` whenever source
semantics change and old cache entries must not be reused.

Do not add RQData, database or local-file logic to xqfactor core. See the RQData post-adjusted close
example in `references/examples.md`.

## Compose Operators

Use exported operators such as `PCT_CHANGE`, `REF`, `RANK`, `NORM`, `MAD`, `IF` and
`CSNEUTRALIZER`. Arithmetic and comparisons create expression nodes:

```python
RETURNS = PCT_CHANGE(CLOSE, 1)
ALPHA = RANK(RETURNS) * -1
```

Use:

- `custom_unary` for one input;
- `custom_binary` for two inputs;
- `rolling_operator` for a historical window;
- `define_operator` and `OperatorRegistry` when an application needs named operator metadata.

With `PandasBackend`, custom functions receive Pandas DataFrames. Convert to Polars or PyTorch
inside the custom function and return a value that the backend can normalize. Preserve row and
column order. Read the Polars and PyTorch examples before implementing this bridge.

## Execute Factors

Create one runtime and reuse it across related calculations:

```python
runtime = FactorRuntime(PandasBackend(), MemoryCache(maxsize=256))
value = RETURNS.evaluate(context, runtime)
```

Reuse the runtime to share leaf and intermediate cache entries. Expect a cache miss when the
universe, time axis, frequency, output slice, semantic options, provider version, backend version
or factor definition changes.

Do not use xqfactor execution cache as a market data store. Persistent full-market Parquet,
DuckDB, Redis or remote cache implementations belong to the consuming application and should only
implement the `ExecutionCache` protocol if needed.

## Analyze Factors

Import optional implementations from `xqfactor.analysis.pandas`:

```python
from xqfactor.analysis.pandas import ICAnalyzer, Normalizer
```

Register processors in required order before calling `analyze`. For normalization followed by IC:

```python
analyzer.register_processor("normalization", Normalizer())
result = analyzer.analyze({"returns": RETURNS})
```

Subclass `AbstractAnalyzer` and implement `_analyze(factors)` to define a custom analyzer. Inputs
have already been evaluated and processed into Pandas DataFrames.

## Extend Carefully

- Add a built-in operator specification in `operators.py`.
- Add its implementation to every backend that claims support.
- Export it from `xqfactor.__init__`.
- Test axis order, NaN behavior, dtype, history requirements and cache identity.
- Keep optional libraries out of core imports.
- Add or update examples when changing public APIs.

## Validate Work

Run:

```bash
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run python -m compileall -q src
uv build
```

When editing this skill, also run:

```bash
uv run --with pyyaml python /Users/xuanqi/.codex/skills/.system/skill-creator/scripts/quick_validate.py skills/xqfactor
```

## Reference

Read `references/examples.md` for:

- RQData post-adjusted `CLOSE`;
- `RETURNS = PCT_CHANGE(CLOSE, 1)`;
- Polars and PyTorch custom operators;
- normalization followed by IC analysis;
- a custom analyzer;
- runtime, history and cache patterns.
