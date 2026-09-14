"""
绩效与风险指标

I. 指标分类

1. 收益类: 累计收益、年化收益（CAGR）、月度/年度收益表
2. 风险类: 年化波动率、最大回撤、下行波动、VaR / CVaR
3. 风险调整收益: Sharpe、Sortino、Calmar、信息比率
4. 相对基准: Alpha、Beta、超额收益
5. 交易特征: 胜率、盈亏比

II. 关于年化的说明 ⚠️

1. 波动率与 Sharpe 使用 252 个交易日/年折算
2. CAGR 使用**真实日历天数**折算，而非交易日数量
   (1) 两者口径不同是有意为之：净值增长发生在日历时间上
   (2) 若用交易日折算长期 CAGR，会系统性高估收益

III. 常见误读

1. Sharpe 高不代表策略好，可能只是样本期太短或存在幸存者偏差
2. 最大回撤必须结合恢复时间一起看，否则无法判断资金占用成本
3. 胜率高但盈亏比极低的策略会"钝刀割肉"，单看胜率会得出错误结论

@module qlearn.metrics.performance
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "TRADING_DAYS_PER_YEAR",
    "total_return",
    "cagr",
    "annualized_volatility",
    "sharpe_ratio",
    "sortino_ratio",
    "drawdown_series",
    "max_drawdown",
    "calmar_ratio",
    "win_rate",
    "value_at_risk",
    "conditional_value_at_risk",
    "alpha_beta",
    "information_ratio",
    "rolling_sharpe",
    "yearly_returns",
    "monthly_returns_table",
    "summary",
]

TRADING_DAYS_PER_YEAR = 252


# ----------------------------------------------------------------------
# 内部工具
# ----------------------------------------------------------------------


def _clean(series: pd.Series) -> pd.Series:
    """去掉 NaN 与 inf，避免污染统计量。"""
    return series.replace([np.inf, -np.inf], np.nan).dropna()


def _to_returns(equity: pd.Series) -> pd.Series:
    """由净值曲线导出日收益率序列。"""
    return equity.pct_change().dropna()


def _is_degenerate(series: pd.Series, min_obs: int = 2) -> bool:
    """判断序列是否样本不足或为常数，此类情况所有统计量均无意义。"""
    if len(series) < min_obs:
        return True
    return bool(np.isclose(series.std(ddof=1), 0.0, atol=1e-12, equal_nan=True))


# ----------------------------------------------------------------------
# 收益类指标
# ----------------------------------------------------------------------


def total_return(equity: pd.Series) -> float:
    """累计收益率 = 期末净值 / 期初净值 - 1。"""
    series = _clean(equity)
    if len(series) < 2 or series.iloc[0] == 0:
        return float("nan")
    return float(series.iloc[-1] / series.iloc[0] - 1.0)


def cagr(equity: pd.Series) -> float:
    """年化复合收益率（Compound Annual Growth Rate）。

    I. 计算方式

    1. 使用真实日历天数而非交易日数量
    2. 公式: (期末 / 期初) ^ (365.25 / 实际天数) - 1

    注意：样本期短于 1 年时该指标极不稳定，仅供参考。
    """
    series = _clean(equity)
    if len(series) < 2 or series.iloc[0] <= 0:
        return float("nan")

    # I. 实际持有天数
    days = (series.index[-1] - series.index[0]).days
    if days <= 0:
        return float("nan")

    # II. 按日历天数年化
    growth = series.iloc[-1] / series.iloc[0]
    if growth <= 0:
        return float("nan")
    return float(growth ** (365.25 / days) - 1.0)


# ----------------------------------------------------------------------
# 风险类指标
# ----------------------------------------------------------------------


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """年化波动率 = 日收益标准差 * sqrt(年交易日数)。"""
    series = _clean(returns)
    if _is_degenerate(series):
        return float("nan")
    return float(series.std(ddof=1) * np.sqrt(periods_per_year))


def drawdown_series(equity: pd.Series) -> pd.Series:
    """回撤序列（负数或 0，单位为比例）。

    定义: drawdown_t = equity_t / max(equity_0..t) - 1
    """
    series = _clean(equity)
    if series.empty:
        return pd.Series(dtype=float)
    return series / series.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float:
    """最大回撤（负数，例如 -0.35 表示最大回撤 35%）。"""
    drawdown = drawdown_series(equity)
    if drawdown.empty:
        return float("nan")
    return float(drawdown.min())


def max_drawdown_duration(equity: pd.Series) -> int:
    """最长回撤持续天数（从创新高到下一次创新高之间的最长自然日数）。"""
    series = _clean(equity)
    if len(series) < 2:
        return 0

    at_high = series >= series.cummax()
    # 每段回撤的起点是上一个新高，终点是下一个新高
    high_dates = series.index[at_high]
    if len(high_dates) < 2:
        return int((series.index[-1] - series.index[0]).days)

    gaps = np.diff(high_dates.values).astype("timedelta64[D]").astype(int)
    return int(gaps.max())


def value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """历史模拟法 VaR（返回负数，表示日收益的第 level 分位）。

    Args:
        returns: 日收益率序列。
        level: 显著性水平，0.05 表示 95% 置信度下的单日最大损失。
    """
    series = _clean(returns)
    if series.empty:
        return float("nan")
    return float(series.quantile(level))


def conditional_value_at_risk(returns: pd.Series, level: float = 0.05) -> float:
    """条件 VaR（期望损失 ES），即超过 VaR 的那部分损失的平均值。"""
    series = _clean(returns)
    if series.empty:
        return float("nan")
    threshold = series.quantile(level)
    tail = series[series <= threshold]
    if tail.empty:
        return float(threshold)
    return float(tail.mean())


# ----------------------------------------------------------------------
# 风险调整收益
# ----------------------------------------------------------------------


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """夏普比率 = 年化超额收益 / 年化波动率。

    Args:
        returns: 日收益率序列。
        risk_free_rate: **年化**无风险利率，如 0.02。
        periods_per_year: 年交易日数。
    """
    series = _clean(returns)
    if _is_degenerate(series):
        return float("nan")

    # I. 年化无风险利率折算为日度
    daily_rf = risk_free_rate / periods_per_year

    # II. 年化超额收益与年化波动
    excess = series - daily_rf
    return float(excess.mean() / series.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """索提诺比率 = 年化超额收益 / 年化下行波动率。

    与夏普的区别：分母只统计**亏损**方向的标准差，对上涨波动不惩罚。
    """
    series = _clean(returns)
    if _is_degenerate(series):
        return float("nan")

    daily_rf = risk_free_rate / periods_per_year
    excess = series - daily_rf

    # I. 下行偏差：只保留负超额收益的平方
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(np.square(downside))) * np.sqrt(periods_per_year)
    if downside_dev == 0 or not np.isfinite(downside_dev):
        return float("nan")
    return float(excess.mean() * periods_per_year / downside_dev)


def calmar_ratio(equity: pd.Series) -> float:
    """卡玛比率 = 年化收益 / |最大回撤|。

    衡量"每承受 1 单位回撤，换取多少年化收益"。
    """
    annual = cagr(equity)
    mdd = max_drawdown(equity)
    if not np.isfinite(annual) or not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return float(annual / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """日度胜率 = 上涨交易日占比（不含持平）。"""
    series = _clean(returns)
    if series.empty:
        return float("nan")
    return float((series > 0).sum() / len(series))


def profit_factor(returns: pd.Series) -> float:
    """盈亏比 = 累计盈利 / 累计亏损（基于日收益）。"""
    series = _clean(returns)
    if series.empty:
        return float("nan")
    gains = series[series > 0].sum()
    losses = -series[series < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


# ----------------------------------------------------------------------
# 相对基准指标
# ----------------------------------------------------------------------


def alpha_beta(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> tuple[float, float]:
    """对基准做 OLS 回归，返回 (年化 Alpha, Beta)。

    I. 回归模型

    1. r_p - rf = alpha + beta * (r_b - rf) + eps
    2. beta = Cov(r_p, r_b) / Var(r_b)
    3. alpha 年化后即为剔除了市场暴露后的超额收益

    Returns:
        (alpha, beta)；样本不足或无波动时返回 (nan, nan)。
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 10:
        return float("nan"), float("nan")

    portfolio = aligned.iloc[:, 0]
    benchmark = aligned.iloc[:, 1]
    if np.isclose(benchmark.var(ddof=1), 0.0, atol=1e-12):
        return float("nan"), float("nan")

    daily_rf = risk_free_rate / periods_per_year
    beta = float(
        np.cov(portfolio - daily_rf, benchmark - daily_rf, ddof=1)[0, 1]
        / benchmark.var(ddof=1)
    )
    alpha_daily = float(portfolio.mean() - daily_rf - beta * (benchmark.mean() - daily_rf))
    return alpha_daily * periods_per_year, beta


def information_ratio(
    returns: pd.Series, benchmark_returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """信息比率 = 年化超额收益 / 年化跟踪误差。"""
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active.std(ddof=1)
    if tracking_error == 0 or not np.isfinite(tracking_error):
        return float("nan")
    return float(active.mean() / tracking_error * np.sqrt(periods_per_year))


def rolling_sharpe(
    returns: pd.Series,
    window: int = 252,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """滚动夏普比率，用于观察策略稳定性随时间的变化。

    若滚动夏普长期在 0 附近或正负剧烈切换，说明策略没有稳定优势。
    """
    series = _clean(returns)
    if len(series) < window:
        return pd.Series(dtype=float)

    daily_rf = risk_free_rate / periods_per_year
    mean = (series - daily_rf).rolling(window).mean()
    std = series.rolling(window).std(ddof=1)
    return mean / std * np.sqrt(periods_per_year)


# ----------------------------------------------------------------------
# 收益分解
# ----------------------------------------------------------------------


def yearly_returns(returns: pd.Series) -> pd.Series:
    """按自然年聚合的年度收益率。"""
    series = _clean(returns)
    if series.empty:
        return pd.Series(dtype=float)
    return series.groupby(series.index.year).apply(lambda r: float((1.0 + r).prod() - 1.0))


def monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """生成「年 x 月」的收益率透视表，最后一列为年度汇总。

    Returns:
        index 为年份、columns 为 1..12 加 "YTD" 的 DataFrame（值为小数）。
    """
    series = _clean(returns)
    if series.empty:
        return pd.DataFrame()

    monthly = series.groupby([series.index.year, series.index.month]).apply(
        lambda r: float((1.0 + r).prod() - 1.0)
    )
    monthly.index.names = ["year", "month"]

    table = monthly.unstack(level="month")
    table = table.reindex(columns=range(1, 13))
    table.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    table["YTD"] = series.groupby(series.index.year).apply(
        lambda r: float((1.0 + r).prod() - 1.0)
    )
    return table


# ----------------------------------------------------------------------
# 汇总
# ----------------------------------------------------------------------


def summary(
    equity: pd.Series,
    returns: pd.Series | None = None,
    benchmark: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """生成完整绩效指标汇总表。

    I. 指标分组

    1. 收益: 累计收益、年化收益
    2. 风险: 年化波动、最大回撤、最长回撤天数、日 VaR / CVaR
    3. 风险调整: Sharpe、Sortino、Calmar、盈亏比、胜率
    4. 相对基准（需传入 benchmark）: 年化超额、Alpha、Beta、信息比率

    Args:
        equity: 组合净值序列（索引为日期）。
        returns: 组合日收益序列；None 时由 equity 推导。
        benchmark: 基准净值序列；用于计算相对指标。
        risk_free_rate: 年化无风险利率。
        periods_per_year: 年交易日数。

    Returns:
        指标名 -> 指标值 的 Series（中文索引）。
    """
    equity = _clean(equity)
    rets = _clean(returns) if returns is not None else _to_returns(equity)

    metrics: dict[str, float] = {
        "累计收益": total_return(equity),
        "年化收益(CAGR)": cagr(equity),
        "年化波动率": annualized_volatility(rets, periods_per_year),
        "最大回撤": max_drawdown(equity),
        "最长回撤天数": float(max_drawdown_duration(equity)),
        "Calmar比率": calmar_ratio(equity),
        "年化Sharpe": sharpe_ratio(rets, risk_free_rate, periods_per_year),
        "年化Sortino": sortino_ratio(rets, risk_free_rate, periods_per_year),
        "日胜率": win_rate(rets),
        "盈亏比": profit_factor(rets),
        "日VaR(95%)": value_at_risk(rets, 0.05),
        "日CVaR(95%)": conditional_value_at_risk(rets, 0.05),
        "偏度": float(rets.skew()) if len(rets) > 2 else float("nan"),
        "峰度": float(rets.kurt()) if len(rets) > 3 else float("nan"),
        "交易日数": float(len(rets)),
    }

    # 相对基准指标
    if benchmark is not None:
        bench = _clean(benchmark)
        if len(bench) >= 2:
            bench_rets = _to_returns(bench)
            alpha, beta = alpha_beta(rets, bench_rets, risk_free_rate, periods_per_year)
            metrics["基准累计收益"] = total_return(bench)
            metrics["年化超额收益"] = cagr(equity) - cagr(bench)
            metrics["年化Alpha"] = alpha
            metrics["Beta"] = beta
            metrics["信息比率"] = information_ratio(rets, bench_rets, periods_per_year)

    return pd.Series(metrics, name="value")


def format_summary(table: pd.Series) -> pd.Series:
    """将指标汇总表格式化为便于阅读的字符串（自动识别比例型与数值型）。"""
    ratio_like = {
        "累计收益", "年化收益(CAGR)", "年化波动率", "最大回撤", "日胜率",
        "日VaR(95%)", "日CVaR(95%)", "基准累计收益", "年化超额收益", "年化Alpha",
    }
    formatted: dict[str, str] = {}
    for name, value in table.items():
        if not np.isfinite(value):
            formatted[name] = "N/A"
        elif name in ratio_like:
            formatted[name] = f"{value:.2%}"
        elif name in {"交易日数", "最长回撤天数"}:
            formatted[name] = f"{int(value)}"
        else:
            formatted[name] = f"{value:.3f}"
    return pd.Series(formatted, name="value")
