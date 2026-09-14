"""
股票池（Universe）管理

I. 用途

定义「在哪些标的上做策略」。股票池的选择本身就是一个重大决策，
对回测结果的影响往往大于策略参数调优。

II. 数据来源

1. 指数成分股：akshare 的 `index_stock_cons_csindex`
2. 全量 A 股列表：akshare 的 `stock_info_a_code_name`
3. 内置样本池：离线可用，用于教学与冒烟测试

III. 幸存者偏差警告 ⚠️

用「今天的指数成分股」回测「过去 5 年」，等于提前知道了哪些股票会被纳入指数。
正确做法是使用**历史成分股快照**（point-in-time constituents）。
akshare 免费接口不提供历史快照，因此本模块的结果仅适合教学，
实盘研究请购买带历史成分的数据或自行从指数公司公告重建。

@module qlearn.data.universe
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qlearn.config import DATA_DIR

__all__ = [
    "BUILTIN_UNIVERSES",
    "get_index_constituents",
    "get_all_a_share_codes",
    "load_universe",
]

# 成分股快照缓存目录
_UNIVERSE_CACHE_DIR = DATA_DIR / "universe"

# 内置样本池：离线可用。代码均为常见大市值 A 股，仅用于教学演示。
BUILTIN_UNIVERSES: dict[str, list[str]] = {
    # 消费 + 金融 + 新能源，大市值、流动性好
    "demo_large_cap": [
        "600519",  # 贵州茅台
        "000858",  # 五粮液
        "601318",  # 中国平安
        "600036",  # 招商银行
        "000001",  # 平安银行
        "300750",  # 宁德时代
        "002594",  # 比亚迪
        "600276",  # 恒瑞医药
        "000333",  # 美的集团
        "601899",  # 紫金矿业
    ],
    # 银行板块，低波动、高股息，适合练均值回归
    "demo_banks": [
        "600036",  # 招商银行
        "601398",  # 工商银行
        "601288",  # 农业银行
        "601988",  # 中国银行
        "600000",  # 浦发银行
    ],
    # 宽基 ETF，用于替代个股做更干净的指数级回测
    "demo_etf": [
        "510300",  # 沪深 300 ETF
        "510500",  # 中证 500 ETF
        "159915",  # 创业板 ETF
        "512880",  # 证券 ETF
        "512000",  # 券商 ETF
    ],
}


def get_all_a_share_codes(use_cache: bool = True) -> pd.DataFrame:
    """获取全量 A 股代码与名称。

    Args:
        use_cache: 是否使用本地 JSON 缓存（全量列表每日变化不大）。

    Returns:
        含 code、name 两列的 DataFrame。
    """
    _UNIVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file: Path = _UNIVERSE_CACHE_DIR / "all_a_share.json"

    # I. 命中缓存
    if use_cache and cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return pd.DataFrame(cached)

    # II. 拉取并写缓存
    import akshare as ak

    frame = ak.stock_info_a_code_name()
    frame = frame.rename(columns={"code": "code", "name": "name"})
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if use_cache:
        cache_file.write_text(
            frame.to_json(orient="records", force_ascii=False), encoding="utf-8"
        )
    return frame


def get_index_constituents(index_code: str = "000300", use_cache: bool = True) -> list[str]:
    """获取指数成分股代码列表。

    ⚠️ 返回的是**当前**成分股，不含历史快照，存在幸存者偏差。

    Args:
        index_code: 指数代码，如 "000300"（沪深 300）、"000905"（中证 500）。
        use_cache: 是否使用本地 JSON 缓存。

    Returns:
        6 位股票代码列表。
    """
    _UNIVERSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file: Path = _UNIVERSE_CACHE_DIR / f"index_{index_code}.json"

    # I. 命中缓存
    if use_cache and cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    # II. 拉取并写缓存
    import akshare as ak

    frame = ak.index_stock_cons_csindex(symbol=index_code)
    codes = frame["成分券代码"].astype(str).str.zfill(6).tolist()
    codes = sorted(set(codes))

    if use_cache:
        cache_file.write_text(json.dumps(codes, ensure_ascii=False), encoding="utf-8")
    return codes


def load_universe(name_or_codes: str | list[str]) -> list[str]:
    """统一入口：按名称取内置池，或按指数代码取成分股，或直接透传代码列表。

    I. 解析优先级

    1. 传入 list -> 直接作为股票池（校验格式）
    2. 传入内置池名称（见 `BUILTIN_UNIVERSES`）-> 返回内置列表
    3. 传入 6 位数字且形如指数代码 -> 尝试拉取指数成分股
    4. 其他 -> 报错并提示可用选项

    Args:
        name_or_codes: 股票池名称、指数代码或代码列表。

    Returns:
        6 位股票代码列表。
    """
    # I. 直接传入列表
    if isinstance(name_or_codes, list):
        return [_normalize_code(code) for code in name_or_codes]

    # II. 内置股票池
    if name_or_codes in BUILTIN_UNIVERSES:
        return list(BUILTIN_UNIVERSES[name_or_codes])

    # III. 指数成分股
    if name_or_codes.isdigit() and len(name_or_codes) == 6:
        return get_index_constituents(name_or_codes)

    # IV. 未知输入
    available = ", ".join(sorted(BUILTIN_UNIVERSES))
    raise ValueError(
        f"无法解析股票池 '{name_or_codes}'。可用内置池: {available}；"
        "或传入 6 位指数代码（如 000300），或直接传入代码列表。"
    )


def _normalize_code(code: str) -> str:
    """将股票代码规范化为 6 位字符串。"""
    normalized = str(code).strip().zfill(6)
    if not normalized.isdigit() or len(normalized) != 6:
        raise ValueError(f"非法的股票代码: {code!r}，应为 6 位数字")
    return normalized
