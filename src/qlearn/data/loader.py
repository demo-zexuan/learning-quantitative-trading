"""
A 股行情数据加载器（基于 akshare）

I. 数据源说明

1. 个股日线: ak.stock_zh_a_hist，支持前复权 / 后复权 / 不复权
2. 指数日线: ak.index_zh_a_hist
3. akshare 为免费公开数据源，无需 token，但**不保证**数据质量与可用性
   (1) 接口可能因上游变更而失效
   (2) 复权因子、停牌处理与商业数据源存在差异
   (3) 教学 / 研究可用，生产环境务必自建数据管道

II. 缓存策略

1. 以 parquet 落盘到 data/raw，key 为「标的 + 复权方式」
2. 命中缓存且区间覆盖请求区间时，直接切片返回，不发网络请求
3. 区间不覆盖时增量合并：新老数据 concat 后按索引去重（保留新值）再落盘

III. 复权方式

1. "qfq" 前复权: 适合技术指标与回测，价格连续可交易性最好
2. "hfq" 后复权: 适合长期收益率计算
3. ""    不复权: 真实成交价，但除权日会出现价格跳空，会污染信号

@module qlearn.data.loader
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from qlearn.config import RAW_DIR
from qlearn.data.panel import PricePanel

__all__ = ["fetch_daily", "fetch_index_daily", "load_panel", "clear_cache", "set_proxy_bypass"]

# 网络重试配置：免费行情接口抖动是常态，必须重试
_RETRY_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 1.0


# ----------------------------------------------------------------------
# 代理绕过
# ----------------------------------------------------------------------

#: 是否对国内数据源绕过系统代理，默认开启。
#: 若关闭，requests 会遵循 HTTP_PROXY / HTTPS_PROXY 环境变量。
BYPASS_PROXY: bool = True

_PROXY_ENV_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "http_proxy",
    "https_proxy",
    "ALL_PROXY",
    "all_proxy",
    "NO_PROXY",
    "no_proxy",
)


def set_proxy_bypass(enabled: bool) -> None:
    """全局开关：是否对国内数据源绕过系统代理。

    I. 为什么需要绕过

    1. 本模块访问的行情接口（东方财富等）全部是**国内站点**
    2. 若系统配置了海外代理（如 Clash / V2Ray 监听 127.0.0.1:7897），
       requests 会默认走代理，而代理往往无法访问国内行情接口，
       表现为 `requests.exceptions.ProxyError`
    3. 此时直连国内接口反而更快更稳定

    Args:
        enabled: True 表示绕过代理。
    """
    global BYPASS_PROXY
    BYPASS_PROXY = bool(enabled)


@contextmanager
def _no_proxy_context():
    """在上下文内临时清空代理相关环境变量并设置 `no_proxy=*`。

    I. 实现说明

    1. requests 在**每次请求时**读取 `os.environ`，因此临时修改环境变量有效
    2. 设置 `no_proxy=*` 是让 requests 的 should_bypass_proxies 直接放行，
       清空代理变量则是双保险
    3. 退出时完整还原现场，不对调用方产生副作用
    """
    saved = {key: os.environ.get(key) for key in _PROXY_ENV_KEYS}

    for key in _PROXY_ENV_KEYS:
        os.environ.pop(key, None)
    os.environ["no_proxy"] = "*"
    os.environ["NO_PROXY"] = "*"

    try:
        yield
    finally:
        for key in ("no_proxy", "NO_PROXY"):
            os.environ.pop(key, None)
        for key, value in saved.items():
            if value is not None:
                os.environ[key] = value


# akshare 中文列名 -> 英文标准列名
_COLUMN_MAPPING: dict[str, str] = {
    "日期": "date",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "涨跌幅": "pct_chg",
    "换手率": "turnover",
}

# 引擎与面板需要的标准列
_STANDARD_COLUMNS: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_chg",
    "turnover",
)


def _cache_path(kind: str, key: str) -> Path:
    """构造缓存文件路径。

    Args:
        kind: 数据类型标识，如 "stock" / "index"。
        key: 用于区分缓存文件的键，如 "000001_qfq"。
    """
    directory = RAW_DIR / kind
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{key}.parquet"


def _normalize(fallback_date_column: str, frame: pd.DataFrame) -> pd.DataFrame:
    """将 akshare 返回表标准化为英文列名 + DatetimeIndex。

    I. 处理步骤

    1. 重命名中文列
    2. 日期列转 datetime 并设为索引
    3. 补齐缺失的标准列（填 NaN），保证下游列结构稳定
    4. 去重、排序

    Args:
        fallback_date_column: 当列映射后没有 date 列时使用的原始列名。
        frame: akshare 原始返回表。
    """
    # I. 重命名
    renamed = frame.rename(columns=_COLUMN_MAPPING)

    # II. 日期列处理
    if "date" not in renamed.columns:
        renamed = renamed.rename(columns={fallback_date_column: "date"})
    if "date" not in renamed.columns or len(renamed) == 0:
        raise ValueError(
            f"数据源返回的表结构不符合预期（shape={frame.shape}，"
            f"columns={list(frame.columns)[:10]}）。"
            "常见原因：复权参数非法、代码不存在、接口已变更。"
        )
    renamed["date"] = pd.to_datetime(renamed["date"])
    renamed = renamed.set_index("date").sort_index()

    # III. 对齐标准列
    for name in _STANDARD_COLUMNS:
        if name not in renamed.columns:
            renamed[name] = pd.NA
    renamed = renamed.loc[:, list(_STANDARD_COLUMNS)]

    # IV. 去重（上游偶发重复行）
    renamed = renamed[~renamed.index.duplicated(keep="last")]
    renamed.index.name = "date"
    return renamed


def _read_cache(path: Path) -> pd.DataFrame | None:
    """读取 parquet 缓存，不存在或损坏时返回 None。"""
    if not path.exists():
        return None
    try:
        cached = pd.read_parquet(path)
    except Exception:  # noqa: BLE001 - 缓存损坏不应阻塞主流程，直接重建
        return None
    cached.index = pd.to_datetime(cached.index)
    return cached.sort_index()


def _write_cache(path: Path, frame: pd.DataFrame) -> None:
    """写入 parquet 缓存。"""
    frame.to_parquet(path, compression="snappy")


def _normalize_adjust(adjust: str | None) -> str:
    """把对外的复权标识统一为 akshare 接受的取值。

    I. 为什么需要转换

    akshare 只接受 `""`（不复权）/ `"qfq"` / `"hfq"` 三种取值。
    若把 `"raw"` 直接透传，两个数据源都会返回空表，最终表现为
    一个莫名其妙的 `KeyError: 'date'`，极难排查。

    II. 接受的输入

    1. `""` / `"raw"` / `"none"` / `"bfq"` -> 不复权，统一为 `""`
    2. `"qfq"` / `"hfq"` -> 原样返回

    Args:
        adjust: 用户传入的复权标识。

    Returns:
        akshare 可接受的复权取值。

    Raises:
        ValueError: 无法识别的取值。
    """
    value = (adjust or "").strip().lower()
    if value in ("", "raw", "none", "bfq"):
        return ""
    if value in ("qfq", "hfq"):
        return value
    raise ValueError(f"未知的复权方式: {adjust!r}，可选 'qfq' / 'hfq' / 'raw'")


def _to_sina_symbol(code: str) -> str:
    """把 6 位 A 股代码转换为新浪接口所需的市场前缀格式。

    I. 映射规则

    1. 6 开头 -> 上交所 (sh)，如 600519 -> sh600519
    2. 0 / 3 开头 -> 深交所 (sz)，如 000001 -> sz000001
    3. 4 / 8 开头 -> 北交所 (bj)
    """
    if code.startswith("6"):
        return f"sh{code}"
    if code.startswith(("0", "3")):
        return f"sz{code}"
    if code.startswith(("4", "8")):
        return f"bj{code}"
    raise ValueError(f"无法识别的 A 股代码: {code}")


def _call_with_retry(func, description: str, attempts: int = _RETRY_ATTEMPTS):
    """带指数退避的重试包装。

    I. 为什么必须重试

    免费公开行情接口（东财、新浪）在高峰期经常出现
    `RemoteDisconnected` / 超时 / 限流，这是常态而非异常。
    没有重试的数据管道在批量加载几十只股票时几乎必然中断。

    Args:
        func: 无参可调用对象。
        description: 用于错误信息的描述文本。
        attempts: 总尝试次数。

    Raises:
        RuntimeError: 所有重试均失败。
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - 需要捕获所有网络层异常以触发重试
            last_error = exc
            if attempt < attempts:
                # 线性退避：1s, 2s, 3s...
                time.sleep(_RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"{description} 重试 {attempts} 次后仍失败: {last_error}")


def _fetch_raw_stock(symbol: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """拉取个股日线原始数据。

    I. 数据源优先级

    1. 东方财富（ak.stock_zh_a_hist）：字段最全，含涨跌幅与换手率
    2. 新浪（ak.stock_zh_a_daily）：备用源，字段略少但更稳定
       (1) 新浪接口不接受复权参数为空字符串以外的值差异，此处统一映射
       (2) 两个源都失败时才向上抛错，由缓存降级机制兜底

    II. 代理处理

    两个数据源都是国内站点，默认绕过系统代理，见 `set_proxy_bypass`。
    """
    import akshare as ak

    def primary() -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust=adjust,
        )

    def fallback() -> pd.DataFrame:
        return ak.stock_zh_a_daily(
            symbol=_to_sina_symbol(symbol),
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
            adjust=adjust,
        )

    def runner() -> pd.DataFrame:
        # 先试主源，失败再试备用源
        try:
            return _call_with_retry(primary, f"东方财富个股接口({symbol})", attempts=2)
        except RuntimeError:
            return _call_with_retry(fallback, f"新浪个股接口({symbol})")

    return _with_bypass(runner) if BYPASS_PROXY else runner()


def _fetch_raw_index(symbol: str, start: str, end: str) -> pd.DataFrame:
    """拉取指数日线原始数据。

    I. 数据源优先级

    1. 东方财富（ak.index_zh_a_hist）
    2. 新浪（ak.stock_zh_index_daily），需要 sh / sz 前缀
    """
    import akshare as ak

    prefix = "sh" if symbol.startswith(("0", "9")) else "sz"

    def primary() -> pd.DataFrame:
        return ak.index_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start.replace("-", ""),
            end_date=end.replace("-", ""),
        )

    def fallback() -> pd.DataFrame:
        return ak.stock_zh_index_daily(symbol=f"{prefix}{symbol}")

    def runner() -> pd.DataFrame:
        try:
            return _call_with_retry(primary, f"东方财富指数接口({symbol})", attempts=2)
        except RuntimeError:
            return _call_with_retry(fallback, f"新浪指数接口({symbol})")

    return _with_bypass(runner) if BYPASS_PROXY else runner()


def _with_bypass(request):
    """在「不走代理」的上下文中执行请求。"""
    with _no_proxy_context():
        return request()


def _load_cached_or_fetch(
    path: Path,
    start: str,
    end: str,
    fetcher,
    use_cache: bool,
) -> pd.DataFrame:
    """带缓存的通用加载逻辑。

    I. 流程

    1. 尝试读取缓存
       (1) 缓存区间完全覆盖请求区间 -> 直接切片返回
       (2) 缓存区间不足 -> 走增量合并
    2. 无缓存或未启用缓存 -> 全量拉取
    3. 拉取失败且存在缓存 -> 降级返回缓存数据（离线可用）

    Args:
        path: 缓存文件路径。
        start: 请求起始日期。
        end: 请求结束日期。
        fetcher: 无参可调用对象，返回标准化后的 DataFrame。
        use_cache: 是否启用缓存。
    """
    cached = _read_cache(path) if use_cache else None

    # I.1 缓存完全覆盖请求区间，直接返回
    if (
        cached is not None
        and len(cached) > 0
        and cached.index.min() <= pd.Timestamp(start)
        and cached.index.max() >= pd.Timestamp(end)
    ):
        return cached.loc[start:end]

    # II. 拉取新数据
    try:
        fresh = fetcher()
        fresh = _normalize("date", fresh)
    except Exception as exc:  # noqa: BLE001 - 网络/上游异常统一降级处理
        # III. 拉取失败时若存在缓存则降级使用，保证离线可复现
        if cached is not None and len(cached) > 0:
            import warnings

            warnings.warn(
                f"数据拉取失败（{exc}），降级使用本地缓存 {path.name}。"
                "结果可能不包含最新交易日。",
                stacklevel=2,
            )
            return cached.loc[start:end]
        raise

    # I.2 增量合并
    if cached is not None and len(cached) > 0:
        combined = pd.concat([cached, fresh])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = fresh

    if use_cache:
        _write_cache(path, combined)

    return combined.loc[start:end]


def fetch_daily(
    symbol: str,
    start: str = "2020-01-01",
    end: str | None = None,
    adjust: str = "qfq",
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取单个 A 股标的的日线行情。

    Args:
        symbol: 6 位股票代码，如 "600519"（贵州茅台）。
        start: 起始日期，格式 "YYYY-MM-DD"。
        end: 结束日期，None 表示今天。
        adjust: 复权方式，"qfq" / "hfq" / "raw"（raw 等价于不复权）。
        use_cache: 是否使用本地 parquet 缓存。

    Returns:
        索引为 DatetimeIndex 的行情表，列见 `_STANDARD_COLUMNS`。
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    mode = _normalize_adjust(adjust)
    path = _cache_path("stock", f"{symbol}_{mode or 'raw'}")
    return _load_cached_or_fetch(
        path,
        start,
        end,
        lambda: _fetch_raw_stock(symbol, start, end, mode),
        use_cache,
    )


def fetch_index_daily(
    symbol: str = "000300",
    start: str = "2020-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """获取 A 股指数日线行情。

    Args:
        symbol: 6 位指数代码，如 "000300"（沪深 300）、"000905"（中证 500）。
        start: 起始日期。
        end: 结束日期，None 表示今天。
        use_cache: 是否使用本地 parquet 缓存。
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    path = _cache_path("index", symbol)
    return _load_cached_or_fetch(
        path,
        start,
        end,
        lambda: _fetch_raw_index(symbol, start, end),
        use_cache,
    )


def load_panel(
    symbols: Sequence[str],
    start: str = "2020-01-01",
    end: str | None = None,
    adjust: str = "qfq",
    use_cache: bool = True,
    min_observations: int = 0,
) -> PricePanel:
    """批量加载多标的行情并组装为 PricePanel。

    I. 处理流程

    1. 逐个标的调用 `fetch_daily`
       (1) 单个标的失败不中断整体流程，记录后跳过
       (2) 全部失败则抛出异常，避免返回空面板导致下游误判
    2. 组装为 PricePanel（自动对齐时间轴）
    3. 可选剔除有效样本过少的标的（新股 / 长期停牌）

    Args:
        symbols: 标的代码序列。
        start: 起始日期。
        end: 结束日期，None 表示今天。
        adjust: 复权方式。
        use_cache: 是否使用缓存。
        min_observations: 见 `PricePanel.drop_inactive_symbols`，0 表示不剔除。

    Returns:
        构造完成的 PricePanel。
    """
    import warnings

    frames: dict[str, pd.DataFrame] = {}
    failures: list[str] = []

    # I. 逐个标的加载
    for symbol in symbols:
        try:
            frames[symbol] = fetch_daily(
                symbol, start=start, end=end, adjust=adjust, use_cache=use_cache
            )
        except Exception as exc:  # noqa: BLE001 - 单标的失败不应中断批量加载
            failures.append(f"{symbol}({exc})")

    # II. 全部失败则直接报错
    if not frames:
        raise RuntimeError(f"所有标的均加载失败: {failures}")

    if failures:
        warnings.warn(f"以下标的加载失败已跳过: {failures}", stacklevel=2)

    panel = PricePanel.from_frames(frames)

    # III. 可选剔除样本不足的标的
    if min_observations > 0:
        panel = panel.drop_inactive_symbols(min_observations)

    return panel


def clear_cache(kind: str | None = None) -> list[Path]:
    """删除本地行情缓存。

    Args:
        kind: 只清理指定类型（"stock" / "index"），None 表示全部。

    Returns:
        被删除的文件路径列表。
    """
    removed: list[Path] = []
    targets: Iterable[Path]
    targets = [RAW_DIR / kind] if kind else [RAW_DIR / "stock", RAW_DIR / "index"]
    for directory in targets:
        if not directory.exists():
            continue
        for file in sorted(directory.glob("*.parquet")):
            file.unlink()
            removed.append(file)
    return removed
