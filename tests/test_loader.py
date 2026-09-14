"""
数据加载器测试（不依赖网络）

I. 测试重点

1. 复权参数归一化：`"raw"` 必须转换成 akshare 接受的 `""`，
   否则接口会返回空表并抛出难以排查的 `KeyError: 'date'`
2. 列名标准化与中文列映射
3. 代理绕过上下文必须完整还原环境变量
4. 缓存命中与「网络失败降级到缓存」的行为
5. 异常表结构必须给出可读的错误信息

@module tests.test_loader
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from qlearn.data import loader as loader_module
from qlearn.data.loader import (
    _load_cached_or_fetch,
    _no_proxy_context,
    _normalize,
    _normalize_adjust,
    _to_sina_symbol,
)

# ----------------------------------------------------------------------
# 复权参数
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_value,expected",
    [
        ("", ""),
        (None, ""),
        ("raw", ""),
        ("RAW", ""),
        ("bfq", ""),
        ("qfq", "qfq"),
        ("QFQ", "qfq"),
        ("hfq", "hfq"),
        (" qfq ", "qfq"),
    ],
)
def test_normalize_adjust(raw_value, expected):
    assert _normalize_adjust(raw_value) == expected


def test_normalize_adjust_rejects_unknown():
    with pytest.raises(ValueError, match="未知的复权方式"):
        _normalize_adjust("adjusted")


# ----------------------------------------------------------------------
# 代码 -> 新浪前缀
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,expected",
    [
        ("600519", "sh600519"),
        ("000001", "sz000001"),
        ("300750", "sz300750"),
        ("833171", "bj833171"),
    ],
)
def test_to_sina_symbol(code, expected):
    assert _to_sina_symbol(code) == expected


def test_to_sina_symbol_rejects_unknown():
    with pytest.raises(ValueError, match="无法识别的 A 股代码"):
        _to_sina_symbol("99999999")


# ----------------------------------------------------------------------
# 列标准化
# ----------------------------------------------------------------------


def test_normalize_maps_chinese_columns():
    """akshare 返回的中文列名必须被映射为英文标准列。"""
    raw = pd.DataFrame(
        {
            "日期": ["2024-01-02", "2024-01-03"],
            "开盘": [10.0, 10.5],
            "收盘": [10.5, 11.0],
            "最高": [10.8, 11.2],
            "最低": [9.9, 10.3],
            "成交量": [1e6, 1.2e6],
            "成交额": [1e7, 1.25e7],
            "涨跌幅": [1.5, 4.76],
            "换手率": [0.01, 0.012],
        }
    )
    normalized = _normalize("date", raw)

    assert isinstance(normalized.index, pd.DatetimeIndex)
    for column in ("open", "high", "low", "close", "volume", "amount", "pct_chg", "turnover"):
        assert column in normalized.columns

    assert normalized.loc["2024-01-02", "close"] == 10.5
    assert normalized.index.is_monotonic_increasing


def test_normalize_fills_missing_standard_columns():
    """缺少某些字段时必须按标准列补 NaN，保证下游列结构稳定。"""
    raw = pd.DataFrame(
        {"日期": ["2024-01-02"], "开盘": [10.0], "收盘": [10.5],
         "最高": [10.8], "最低": [9.9]}
    )
    normalized = _normalize("date", raw)
    # 成交量缺失应被补齐为 NaN 而不是报错
    assert "volume" in normalized.columns
    assert pd.isna(normalized.loc["2024-01-02", "volume"])


def test_normalize_raises_readable_error_on_bad_schema():
    """表结构异常时必须给出可读错误，而不是 KeyError: 'date'。"""
    bogus = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(ValueError, match="表结构不符合预期"):
        _normalize("date", bogus)


def test_normalize_raises_on_empty_frame():
    with pytest.raises(ValueError, match="表结构不符合预期"):
        _normalize("date", pd.DataFrame())


# ----------------------------------------------------------------------
# 代理绕过
# ----------------------------------------------------------------------


def test_no_proxy_context_clears_and_restores(monkeypatch):
    """代理绕过上下文必须在退出后完整还原环境变量。"""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:7897")
    monkeypatch.setenv("NO_PROXY", "localhost")
    # 记录现场：运行环境本身可能已设置 no_proxy，需按原值还原
    original_no_proxy = os.environ.get("no_proxy")

    with _no_proxy_context():
        assert os.environ.get("HTTPS_PROXY") is None
        assert os.environ.get("http_proxy") is None
        assert os.environ["no_proxy"] == "*"
        assert os.environ["NO_PROXY"] == "*"

    assert os.environ["HTTPS_PROXY"] == "http://127.0.0.1:7897"
    assert os.environ["http_proxy"] == "http://127.0.0.1:7897"
    assert os.environ["NO_PROXY"] == "localhost"
    assert os.environ.get("no_proxy") == original_no_proxy


def test_no_proxy_context_restores_even_on_exception(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy:8080")
    with pytest.raises(RuntimeError, match="boom"), _no_proxy_context():
        raise RuntimeError("boom")
    assert os.environ["HTTPS_PROXY"] == "http://proxy:8080"


# ----------------------------------------------------------------------
# 缓存与降级
# ----------------------------------------------------------------------


def _raw_frame(start: str, periods: int) -> pd.DataFrame:
    """构造与 akshare 返回结构一致的原始表（含 date 列，未设索引）。

    真实数据源返回的就是这种形态，`_load_cached_or_fetch` 会对其做标准化。
    """
    index = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "date": index,
            "开盘": 10.0,
            "最高": 10.5,
            "最低": 9.5,
            "收盘": 10.2,
            "成交量": 1e6,
            "成交额": 1e7,
            "涨跌幅": 0.0,
            "换手率": 0.01,
        }
    )


def _cached_frame(start: str, periods: int) -> pd.DataFrame:
    """构造 parquet 缓存文件内容（索引为 DatetimeIndex）。"""
    return _normalize("date", _raw_frame(start, periods))


def test_cache_hit_avoids_fetch(tmp_path):
    """缓存覆盖请求区间时，不得发起任何网络请求。"""
    path = tmp_path / "cache.parquet"
    _cached_frame("2024-01-01", 100).to_parquet(path)

    def fetcher():  # pragma: no cover - 不应被调用
        raise AssertionError("缓存命中时不应调用数据源")

    result = _load_cached_or_fetch(path, "2024-02-01", "2024-03-01", fetcher, use_cache=True)
    assert not result.empty
    assert result.index.min() >= pd.Timestamp("2024-02-01")


def test_cache_miss_triggers_fetch_and_writes_cache(tmp_path):
    path = tmp_path / "cache.parquet"
    calls = {"n": 0}

    def fetcher():
        calls["n"] += 1
        return _raw_frame("2024-01-01", 50)

    result = _load_cached_or_fetch(path, "2024-01-01", "2024-03-31", fetcher, use_cache=True)

    assert calls["n"] == 1
    assert path.exists(), "拉取成功后应写入缓存"
    assert not result.empty
    assert not result.index.duplicated().any()


def test_incremental_merge_on_partial_cache(tmp_path):
    """缓存只覆盖部分区间时，必须合并新旧数据而非重复拉取。"""
    path = tmp_path / "cache.parquet"
    _cached_frame("2024-01-01", 20).to_parquet(path)

    def fetcher():
        return _raw_frame("2024-02-01", 20)

    result = _load_cached_or_fetch(path, "2024-01-01", "2024-03-31", fetcher, use_cache=True)

    merged = pd.read_parquet(path)
    assert not merged.index.duplicated().any(), "合并后不应有重复日期"
    assert merged.index.is_monotonic_increasing
    assert not result.empty


def test_fetch_failure_degrades_to_cache(tmp_path):
    """刷新失败但存在缓存时，应降级使用缓存并给出警告。"""
    path = tmp_path / "cache.parquet"
    _cached_frame("2024-01-01", 60).to_parquet(path)

    def failing():
        raise ConnectionError("模拟网络中断")

    # 请求区间必须超出缓存覆盖范围，否则会在缓存命中时提前返回
    with pytest.warns(UserWarning, match="降级使用本地缓存"):
        result = _load_cached_or_fetch(
            path, "2024-01-15", "2024-12-31", failing, use_cache=True
        )
    assert not result.empty


def test_fetch_failure_without_cache_raises(tmp_path):
    """没有任何数据可降级时，必须抛出异常而不是返回空表。"""
    path = tmp_path / "missing.parquet"

    def failing():
        raise ConnectionError("模拟网络中断")

    with pytest.raises(ConnectionError):
        _load_cached_or_fetch(path, "2024-01-01", "2024-03-31", failing, use_cache=True)


def test_set_proxy_bypass_toggles_global_flag(monkeypatch):
    """全局开关必须能被切换，并影响请求路径的选择。"""
    original = loader_module.BYPASS_PROXY
    try:
        loader_module.set_proxy_bypass(False)
        assert loader_module.BYPASS_PROXY is False
        loader_module.set_proxy_bypass(True)
        assert loader_module.BYPASS_PROXY is True
    finally:
        loader_module.set_proxy_bypass(original)


def test_cache_path_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "RAW_DIR", tmp_path / "raw")
    path = loader_module._cache_path("stock", "600519_qfq")

    assert isinstance(path, Path)
    assert path.parent.exists()
    assert path.name == "600519_qfq.parquet"
