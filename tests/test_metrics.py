"""
绩效指标测试

I. 测试策略

使用**解析解可手算**的构造数据（常数收益、完美线性净值等），
验证指标实现与数学定义一致，而不是简单地"跑通"。

@module tests.test_metrics
@author zexuan.peng
@created 2026-09-14
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qlearn.metrics import (
    alpha_beta,
    annualized_volatility,
    cagr,
    calmar_ratio,
    conditional_value_at_risk,
    drawdown_series,
    information_ratio,
    max_drawdown,
    monthly_returns_table,
    profit_factor,
    sharpe_ratio,
    summary,
    total_return,
    value_at_risk,
    win_rate,
    yearly_returns,
)


def _equity(values: list[float], start: str = "2020-01-01", freq: str = "B") -> pd.Series:
    """按交易日频率构造净值序列。"""
    index = pd.date_range(start, periods=len(values), freq=freq)
    return pd.Series(values, index=index, dtype=float)


# ----------------------------------------------------------------------
# 基础收益指标
# ----------------------------------------------------------------------


def test_total_return():
    assert total_return(_equity([1.0, 1.5])) == pytest.approx(0.5)
    assert total_return(_equity([2.0, 1.0])) == pytest.approx(-0.5)
    # 单点序列无收益可言
    assert np.isnan(total_return(_equity([1.0])))


def test_cagr_uses_calendar_days():
    """CAGR 必须按真实日历天数年化，而不是交易日数量。"""
    index = pd.DatetimeIndex(["2020-01-01", "2021-01-01"])  # 恰好 366 天
    equity = pd.Series([1.0, 1.2], index=index)

    days = (index[1] - index[0]).days
    expected = 1.2 ** (365.25 / days) - 1.0
    assert cagr(equity) == pytest.approx(expected)

    # 两年翻倍 -> 年化约 41.4%
    two_year = pd.Series([1.0, 2.0], index=pd.DatetimeIndex(["2020-01-01", "2022-01-01"]))
    assert cagr(two_year) == pytest.approx(0.414, abs=0.01)


def test_cagr_nan_when_too_short():
    """样本不足两个点时无法计算年化收益。"""
    assert np.isnan(cagr(_equity([1.0])))


# ----------------------------------------------------------------------
# 风险指标
# ----------------------------------------------------------------------


def test_drawdown_series_and_max_drawdown():
    equity = _equity([1.0, 2.0, 1.5, 1.8])
    drawdown = drawdown_series(equity)

    # 从 2.0 跌到 1.5 = -25%
    assert drawdown.iloc[2] == pytest.approx(-0.25)
    assert max_drawdown(equity) == pytest.approx(-0.25)

    # 回撤序列永远不会为正
    assert (drawdown <= 1e-12).all()


def test_max_drawdown_is_negative_and_bounded():
    equity = _equity([1.0, 0.5, 0.25, 1.0])
    mdd = max_drawdown(equity)
    assert -1.0 <= mdd <= 0.0
    assert mdd == pytest.approx(-0.75)


def test_calmar_ratio_consistent_with_cagr_and_mdd():
    equity = _equity([1.0, 1.2, 0.9, 1.4], start="2020-01-01", freq="90D")
    expected = cagr(equity) / abs(max_drawdown(equity))
    assert calmar_ratio(equity) == pytest.approx(expected)


def test_annualized_volatility_of_constant_series_is_nan():
    """常数收益（零波动）时波动率无定义，必须返回 NaN 而不是 0 或 inf。"""
    returns = pd.Series([0.01] * 50, index=pd.date_range("2024-01-01", periods=50))
    assert np.isnan(annualized_volatility(returns))


def test_annualized_volatility_scaling():
    """年化波动率必须等于日波动率乘以 sqrt(252)。"""
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    expected = returns.std(ddof=1) * np.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_sharpe_ratio_of_known_series():
    """Sharpe 必须等于 mean/std * sqrt(252)。"""
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.normal(0.0005, 0.01, 2000))

    expected = returns.mean() / returns.std(ddof=1) * np.sqrt(252)
    assert sharpe_ratio(returns) == pytest.approx(expected)

    # 无风险利率上升时 Sharpe 必须下降
    assert sharpe_ratio(returns, risk_free_rate=0.03) < sharpe_ratio(returns)


def test_value_at_risk_and_cvar_ordering():
    """CVaR（尾部均值）必须不高于 VaR（分位数）。"""
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0, 0.02, 2000))

    var = value_at_risk(returns, 0.05)
    cvar = conditional_value_at_risk(returns, 0.05)

    assert var < 0
    assert cvar <= var + 1e-12


def test_win_rate_and_profit_factor():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    assert win_rate(returns) == pytest.approx(3 / 5)
    assert profit_factor(returns) == pytest.approx((0.01 + 0.03 + 0.02) / (0.02 + 0.01))

    # 全盈利时盈亏比为 inf
    assert profit_factor(pd.Series([0.01, 0.02])) == np.inf


# ----------------------------------------------------------------------
# 相对基准指标
# ----------------------------------------------------------------------


def test_alpha_beta_with_identical_series():
    """策略收益与基准完全相同时，beta = 1 且 alpha = 0。"""
    rng = np.random.default_rng(7)
    benchmark = pd.Series(rng.normal(0.0004, 0.012, 500))

    alpha, beta = alpha_beta(benchmark.copy(), benchmark)
    assert beta == pytest.approx(1.0, abs=1e-9)
    assert alpha == pytest.approx(0.0, abs=1e-9)


def test_alpha_beta_detects_leverage():
    """策略 = 2 倍基准时，beta 应约为 2。"""
    rng = np.random.default_rng(8)
    benchmark = pd.Series(rng.normal(0.0004, 0.012, 800))
    portfolio = benchmark * 2.0

    _alpha, beta = alpha_beta(portfolio, benchmark)
    assert beta == pytest.approx(2.0, abs=1e-6)


def test_information_ratio_zero_for_identical_series():
    """与基准完全一致时，跟踪误差为 0，信息比率无定义。"""
    rng = np.random.default_rng(9)
    benchmark = pd.Series(rng.normal(0.0004, 0.012, 300))
    assert np.isnan(information_ratio(benchmark.copy(), benchmark))


# ----------------------------------------------------------------------
# 收益分解与汇总
# ----------------------------------------------------------------------


def test_monthly_returns_table_shape_and_consistency():
    rng = np.random.default_rng(3)
    index = pd.date_range("2022-01-01", "2023-12-31", freq="B")
    returns = pd.Series(rng.normal(0.0003, 0.01, len(index)), index=index)

    table = monthly_returns_table(returns)

    assert list(table.index) == [2022, 2023]
    assert table.shape == (2, 13)
    assert "YTD" in table.columns

    # YTD 必须等于该年全部日收益的累乘
    for year in (2022, 2023):
        chunk = returns[returns.index.year == year]
        assert table.loc[year, "YTD"] == pytest.approx((1.0 + chunk).prod() - 1.0)


def test_yearly_returns():
    returns = pd.Series(
        [0.1, 0.1], index=pd.DatetimeIndex(["2022-06-01", "2023-06-01"])
    )
    yearly = yearly_returns(returns)
    assert list(yearly.index) == [2022, 2023]
    assert yearly.loc[2022] == pytest.approx(0.1)


def test_summary_contains_core_metrics_and_benchmark_block():
    rng = np.random.default_rng(4)
    index = pd.date_range("2021-01-01", "2023-12-31", freq="B")
    returns = pd.Series(rng.normal(0.0004, 0.011, len(index)), index=index)
    equity = (1.0 + returns).cumprod()

    base = summary(equity, returns=returns)
    for key in ("累计收益", "年化收益(CAGR)", "年化Sharpe", "最大回撤", "日VaR(95%)"):
        assert key in base.index

    # 传入基准后必须补充相对指标
    with_benchmark = summary(equity, returns=returns, benchmark=equity * 0.9)
    for key in ("Beta", "年化Alpha", "信息比率", "年化超额收益"):
        assert key in with_benchmark.index


def test_format_summary_outputs_strings():
    from qlearn.metrics import format_summary

    rng = np.random.default_rng(5)
    index = pd.date_range("2021-01-01", periods=400, freq="B")
    returns = pd.Series(rng.normal(0.0004, 0.011, len(index)), index=index)
    equity = (1.0 + returns).cumprod()

    formatted = format_summary(summary(equity, returns=returns))
    assert formatted.map(type).eq(str).all()
    assert formatted["累计收益"].endswith("%")


def test_metrics_handle_empty_and_nan_gracefully():
    """空序列与含 NaN/inf 的序列必须返回 NaN 而不是抛异常。"""
    empty = pd.Series(dtype=float)
    assert np.isnan(total_return(empty))
    assert np.isnan(max_drawdown(empty))
    assert np.isnan(sharpe_ratio(empty))

    dirty = pd.Series([0.01, np.nan, np.inf, -0.02])
    assert np.isfinite(annualized_volatility(dirty))
