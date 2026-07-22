import pandas as pd
import pytest

from xqfactor import CacheKey, MemoryCache


def test_memory_cache_rejects_invalid_size() -> None:
    """缓存容量必须为正整数。"""
    with pytest.raises(ValueError):
        MemoryCache(maxsize=0)


def test_memory_cache_returns_independent_dataframe() -> None:
    """调用方修改缓存读取结果时不应污染缓存内容。"""
    cache = MemoryCache()
    key = CacheKey("factor", "context")
    original = pd.DataFrame([[1.0]], index=["t0"], columns=["A"])
    cache.set(key, original)

    loaded = cache.get(key)
    assert loaded is not None
    loaded.loc["t0", "A"] = 2.0

    reloaded = cache.get(key)
    assert reloaded is not None
    assert reloaded.loc["t0", "A"] == 1.0
