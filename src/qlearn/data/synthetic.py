"""
合成行情数据生成器

⚠️ 用途声明：本模块产生的数据**没有任何真实市场信息**，仅用于

1. 在无网络环境下跑通「数据 -> 策略 -> 回测 -> 绩效」全链路
2. 单元测试的确定性输入（固定随机种子）
3. 验证回测引擎的会计恒等式是否成立

**绝对不要**基于合成数据得出任何关于策略有效性的结论。

I. 价格过程

1. 共同市场因子 r_m ~ N(mu_m / 252, (sigma_m / sqrt(252))^2)
2. 个股对数收益 r_i = alpha_i + beta_i * r_m + eps_i，其中 eps_i ~ N(0, sigma_i^2 / 252)
3. 收盘价 close_t = close_0 * exp(cumsum(r))

II. 由收盘价反推 OHLCV

1. open  = 前收 * (1 + 开盘跳空噪声)
2. high  = max(open, close) * (1 + 上影噪声)
3. low   = min(open, close) * (1 - 下影噪声)
4. volume ~ 对数正态分布

III. 自相关旋钮（教学核心）

真实市场的收益率并非独立同分布。通过 `autocorrelation` 参数可以把收益序列
变成 AR(1) 过程，从而构造出"有结构"的合成市场：

1. autocorrelation = 0.00 -> 随机游走，任何策略在扣费后都应亏损
   (1) 这是检验回测引擎是否在"造假"的最佳基准
2. autocorrelation > 0    -> 收益存在惯性，趋势/动量策略应当有效
3. autocorrelation < 0    -> 收益存在反转，均值回归策略应当有效

@module qlearn.data.synthetic
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qlearn.data.panel import PricePanel

__all__ = ["make_synthetic_panel"]

# 一年按 252 个交易日折算
TRADING_DAYS_PER_YEAR = 252


def _apply_ar1(noise: np.ndarray, phi: float, mean: float = 0.0) -> np.ndarray:
    """对白噪声施加 AR(1) 滤波，生成具有自相关结构的序列。

    I. 递推式

    y_t = mean + phi * (y_{t-1} - mean) + sqrt(1 - phi^2) * (eps_t - mean)

    其中 sqrt(1 - phi^2) 这一项用于**保持边缘方差不变**，
    使得调整 phi 时不会连带改变波动率，参数之间相互正交，便于教学对比。

    Args:
        noise: 形状为 (T,) 或 (T, N) 的白噪声，沿第 0 轴递推。
        phi: 一阶自相关系数，取值 (-0.99, 0.99)。
        mean: 序列均值。

    Returns:
        与输入同形状、具有 AR(1) 结构的序列。
    """
    if not (-0.99 < phi < 0.99):
        raise ValueError(f"autocorrelation 必须落在 (-0.99, 0.99) 区间，当前为 {phi}")
    if phi == 0.0:
        return noise

    centered = noise - mean
    out = np.empty_like(centered)
    out[0] = centered[0]
    scale = np.sqrt(1.0 - phi * phi)
    for t in range(1, centered.shape[0]):
        out[t] = phi * out[t - 1] + scale * centered[t]
    return out + mean


def make_synthetic_panel(
    symbols: list[str] | None = None,
    start: str = "2020-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
    annual_drift: float = 0.04,
    market_annual_vol: float = 0.20,
    stock_annual_vol: float = 0.35,
    market_beta: float = 0.90,
    initial_price: float = 20.0,
    suspension_prob: float = 0.0,
    autocorrelation: float = 0.0,
) -> PricePanel:
    """生成多标的合成行情面板。

    I. 参数说明

    1. annual_drift: 市场因子的年化漂移率
    2. market_annual_vol: 市场因子的年化波动率
    3. stock_annual_vol: 个股特质波动的年化波动率
    4. market_beta: 个股对市场因子的暴露中枢
    5. suspension_prob: 单日停牌概率，用于测试停牌处理逻辑
    6. autocorrelation: 收益的一阶自相关系数
       (1) 0 表示随机游走（有效市场），扣费后所有策略都应亏损
       (2) 正数引入趋势性，动量/趋势策略应当有效
       (3) 负数引入反转性，均值回归策略应当有效

    Args:
        symbols: 标的代码列表，默认生成 5 只。
        start: 起始日期（含）。
        end: 结束日期（含）。
        seed: 随机种子，保证结果可复现。

    Returns:
        构造完成的 PricePanel。
    """
    # I. 准备时间轴与随机数发生器
    # 1. 使用工作日近似交易日，跳过周末
    dates = pd.bdate_range(start=start, end=end, name="date")
    if len(dates) < 2:
        raise ValueError("日期区间过短，至少需要 2 个交易日")

    # 2. 固定种子，保证同一 seed 下结果完全一致
    rng = np.random.default_rng(seed)

    codes = symbols if symbols is not None else [f"SYN{i:03d}" for i in range(1, 6)]
    n_days = len(dates)
    n_stocks = len(codes)

    # II. 生成共同市场因子
    # 1. 日度漂移与波动由年化参数折算
    market_mu = annual_drift / TRADING_DAYS_PER_YEAR
    market_sigma = market_annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)

    # 2. 市场因子日对数收益（可选 AR(1) 结构）
    market_ret = _apply_ar1(
        rng.normal(market_mu, market_sigma, size=n_days), autocorrelation, mean=market_mu
    )

    # III. 生成个股收益
    # 1. 每只股票有独立的 beta 与特质波动，围绕中枢小幅扰动
    betas = market_beta + rng.normal(0.0, 0.15, size=n_stocks)
    idiosyncratic_sigma = stock_annual_vol / np.sqrt(TRADING_DAYS_PER_YEAR)

    # 2. r_i = beta_i * r_m + eps_i（此处 alpha 置零，不人为制造"稳赚"的股票）
    shocks = _apply_ar1(
        rng.normal(0.0, idiosyncratic_sigma, size=(n_days, n_stocks)), autocorrelation
    )
    log_returns = market_ret[:, None] * betas[None, :] + shocks

    # IV. 累积为价格序列
    close = initial_price * np.exp(np.cumsum(log_returns, axis=0))

    # V. 由收盘价反推 OHLCV
    # 1. 开盘价：前一日收盘价叠加跳空噪声，首日直接用收盘价
    gap = rng.normal(0.0, 0.004, size=(n_days, n_stocks))
    open_px = np.empty_like(close)
    open_px[0] = close[0]
    open_px[1:] = close[:-1] * (1.0 + gap[1:])

    # 2. 影线：high / low 必须包住 open 与 close，否则是不合法的 K 线
    upper_shadow = np.abs(rng.normal(0.0, 0.006, size=(n_days, n_stocks)))
    lower_shadow = np.abs(rng.normal(0.0, 0.006, size=(n_days, n_stocks)))
    high = np.maximum(open_px, close) * (1.0 + upper_shadow)
    low = np.minimum(open_px, close) * (1.0 - lower_shadow)

    # 3. 成交量：对数正态分布，与当日波动幅度正相关
    base_volume = rng.lognormal(mean=14.0, sigma=0.4, size=(n_days, n_stocks))
    volume = base_volume * (1.0 + 5.0 * np.abs(log_returns))

    # VI. 注入停牌日（可选）
    # 1. 停牌日所有价格与成交量置为 NaN，由 tradable 掩码屏蔽交易
    if suspension_prob > 0.0:
        suspended = rng.random(size=(n_days, n_stocks)) < suspension_prob
        for array in (open_px, high, low, close, volume):
            array[suspended] = np.nan

    # VII. 组装为 PricePanel
    columns = pd.Index(codes, name="symbol")
    fields = {
        "open": pd.DataFrame(open_px, index=dates, columns=columns),
        "high": pd.DataFrame(high, index=dates, columns=columns),
        "low": pd.DataFrame(low, index=dates, columns=columns),
        "close": pd.DataFrame(close, index=dates, columns=columns),
        "volume": pd.DataFrame(volume, index=dates, columns=columns),
        "amount": pd.DataFrame(close * volume, index=dates, columns=columns),
    }
    return PricePanel(fields)
